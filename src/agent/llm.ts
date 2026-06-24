/**
 * Minimal OpenAI-compatible chat client (LM Studio / any local endpoint).
 * Extracted from the validated fixture: retries, reasoning-channel
 * fallback (thinking models leave content empty), and a hard timeout so a
 * wedged backend fails loudly instead of freezing the REPL.
 *
 * Requests are STREAMING by default (SSE). Non-streaming requests sit
 * silently until the backend finishes the whole generation before sending
 * response headers - and Node's fetch (undici) kills headerless requests
 * at ~300s (headersTimeout), surfacing as `TypeError: fetch failed` long
 * before our own AbortSignal fires. Long local generations (minutes) are
 * normal here, so streaming is the only reliable transport: headers come
 * immediately, chunks keep the socket alive, and the overall wall-clock
 * budget stays enforced by AbortSignal.timeout. Found via the 2026-06-12
 * bench run where every hard baseline call died at ~305s x 3 retries.
 * Escape hatch: RULITH_LLM_STREAM=0 or config { stream: false }.
 */

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

export interface LlmConfig {
  baseUrl?: string
  model?: string
  maxTokens?: number
  temperature?: number
  timeoutMs?: number
  stream?: boolean
  /** Hard cap on accumulated streamed chars - runaway-generation fuse. */
  maxStreamChars?: number
  /**
   * Bearer token for endpoints that require auth (cloud OpenAI-compatible
   * APIs: deepseek, dashscope, ...). Defaults to RULITH_LLM_API_KEY; pass
   * '' to send NO key even when that env var is set (used by the model-B
   * config so a different endpoint never receives model A's credentials).
   */
  apiKey?: string
  /**
   * Ask streaming backends to report token usage in the final SSE chunk
   * (`stream_options.include_usage`). Defaults on; RULITH_LLM_USAGE=0 or
   * { usageProbe: false } turns it off for backends that reject the field.
   */
  usageProbe?: boolean
  fetchImpl?: typeof fetch
}

/**
 * Token accounting for a window of calls (see consumeUsage). Cross-model
 * wall-clock comparisons are confounded by serving speed (a cloud endpoint
 * streams far more tokens/s than a local GPU) - token counts are the unit
 * that compares fairly across hardware.
 *
 * `calls` counts ATTEMPTS (retries included). When a backend reports no
 * usage - or a call dies mid-stream (timeout, stream cap) - the burn is
 * estimated at chars/4 from whatever arrived and the call is counted in
 * `estimatedCalls`, so numbers are never silently missing or zero.
 */
export interface LlmUsage {
  promptTokens: number
  completionTokens: number
  calls: number
  estimatedCalls: number
}

const EMPTY_USAGE: LlmUsage = { promptTokens: 0, completionTokens: 0, calls: 0, estimatedCalls: 0 }

type ReportedUsage = { prompt_tokens?: number; completion_tokens?: number }
type ChatOptions = { signal?: AbortSignal }

export class LlmClient {
  private readonly baseUrl: string
  private readonly model: string
  private readonly maxTokens: number
  private readonly temperature: number
  private readonly timeoutMs: number
  private readonly stream: boolean
  private readonly maxStreamChars: number
  private readonly apiKey: string
  private readonly usageProbe: boolean
  private readonly fetchImpl: typeof fetch
  private usageAcc: LlmUsage = { ...EMPTY_USAGE }

  constructor(config: LlmConfig = {}) {
    this.baseUrl = config.baseUrl ?? process.env.RULITH_LLM_BASE_URL ?? 'http://127.0.0.1:1234'
    this.model = config.model ?? process.env.RULITH_LLM_MODEL ?? 'local-model'
    this.maxTokens = config.maxTokens ?? Number(process.env.RULITH_MAX_TOKENS ?? 8000)
    this.temperature = config.temperature ?? 0.2
    this.timeoutMs = config.timeoutMs ?? Number(process.env.RULITH_LLM_TIMEOUT_MS ?? 180000)
    this.stream = config.stream ?? process.env.RULITH_LLM_STREAM !== '0'
    this.maxStreamChars =
      config.maxStreamChars ?? Number(process.env.RULITH_LLM_MAX_STREAM_CHARS ?? 8_000_000)
    this.apiKey = (config.apiKey ?? process.env.RULITH_LLM_API_KEY ?? '').trim()
    this.usageProbe = config.usageProbe ?? process.env.RULITH_LLM_USAGE !== '0'
    this.fetchImpl = config.fetchImpl ?? globalThis.fetch
  }

  /** Read-and-reset the token window: call before a unit of work, run it,
   *  call again - the second read is that unit's consumption. */
  consumeUsage(): LlmUsage {
    const out = this.usageAcc
    this.usageAcc = { ...EMPTY_USAGE }
    return out
  }

  private recordUsage(reported: ReportedUsage | undefined, messages: ChatMessage[], reply: string): void {
    this.usageAcc.calls += 1
    if (
      reported &&
      typeof reported.prompt_tokens === 'number' &&
      typeof reported.completion_tokens === 'number'
    ) {
      this.usageAcc.promptTokens += reported.prompt_tokens
      this.usageAcc.completionTokens += reported.completion_tokens
      return
    }
    // No report: estimate at chars/4 and say so. An honest approximation
    // beats a silent zero in any efficiency comparison.
    this.usageAcc.estimatedCalls += 1
    this.usageAcc.promptTokens += Math.ceil(JSON.stringify(messages).length / 4)
    this.usageAcc.completionTokens += Math.ceil(reply.length / 4)
  }

  async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<string> {
    let lastError: unknown
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        if (options.signal?.aborted) throw new DOMException('LLM request aborted', 'AbortError')
        return await this.chatOnce(messages, options.signal)
      } catch (error) {
        // Budget verdicts (our own timeout, the stream-volume fuse) are
        // final: a repetition loop retried is the same loop, three times
        // the wall clock. Only transient transport faults get retries.
        if (LlmClient.isBudgetVerdict(error)) throw error
        lastError = error
        // Narrate, or a dead backend looks exactly like a thinking model
        // from the outside (2026-06-12 wedged-server episode).
        console.error(
          `[llm] attempt ${attempt + 1}/3 failed (${String(error).slice(0, 100)}) - retrying`,
        )
        if (options.signal?.aborted) throw error
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
      }
    }
    throw lastError
  }

  private static isBudgetVerdict(error: unknown): boolean {
    if (error instanceof Error && /stream cap exceeded/i.test(error.message)) return true
    const name = (error as { name?: string } | null)?.name
    return name === 'TimeoutError' || name === 'AbortError'
  }

  /**
   * Join the base URL with the chat-completions path. Cloud OpenAI-compatible
   * bases are habitually written WITH a /v1 suffix (dashscope's
   * compatible-mode even requires it) - appending another /v1 there would
   * 404, so an existing versioned suffix is detected and kept.
   */
  private endpointUrl(): string {
    const base = this.baseUrl.replace(/\/+$/, '')
    return /\/v\d+$/.test(base) ? `${base}/chat/completions` : `${base}/v1/chat/completions`
  }

  private requestSignal(signal: AbortSignal | undefined): AbortSignal {
    const timeout = AbortSignal.timeout(this.timeoutMs)
    return signal ? AbortSignal.any([signal, timeout]) : timeout
  }

  private async chatOnce(messages: ChatMessage[], signal?: AbortSignal): Promise<string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (this.apiKey !== '') headers.Authorization = `Bearer ${this.apiKey}`
    const response = await this.fetchImpl(this.endpointUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: this.model,
        messages,
        temperature: this.temperature,
        max_tokens: this.maxTokens,
        ...(this.stream
          ? { stream: true, ...(this.usageProbe ? { stream_options: { include_usage: true } } : {}) }
          : {}),
      }),
      signal: this.requestSignal(signal),
    })
    if (!response.ok) {
      const hint =
        response.status === 400 && this.stream && this.usageProbe
          ? ' (if the backend rejects stream_options/include_usage, set RULITH_LLM_USAGE=0)'
          : ''
      throw new Error(`LLM request failed: ${response.status} ${await response.text()}${hint}`)
    }
    const contentType = response.headers.get('content-type') ?? ''
    // Servers that ignore `stream` answer with a plain JSON body.
    if (!this.stream || contentType.includes('application/json') || response.body === null) {
      const data = (await response.json()) as {
        choices: Array<{ message: { content?: string; reasoning_content?: string; reasoning?: string } }>
        usage?: ReportedUsage
      }
      const message = data.choices[0]?.message
      const content = message?.content?.trim()
      const reply =
        content && content.length > 0
          ? content
          : (message?.reasoning_content ?? message?.reasoning ?? '')
      this.recordUsage(data.usage, messages, reply)
      return reply
    }
    return this.readSse(response.body, messages)
  }

  /**
   * Aggregate an OpenAI-style SSE stream into the final message text.
   *
   * Two hard-won rules (2026-06-12 bench OOM, 4GB heap of LIVE ropes):
   * - `delta` chunks are increments and get appended; `message` chunks
   *   are the message-so-far and REPLACE when they grew (appending
   *   cumulative snapshots retains O(n^2) chars). A shrinking `message`
   *   chunk is treated as an increment (some bridges stream per-token
   *   message objects).
   * - A total-volume fuse: past maxStreamChars the call FAILS loudly
   *   (DNF for a bench arm) instead of growing until the process dies.
   */
  private async readSse(body: ReadableStream<Uint8Array>, messages: ChatMessage[]): Promise<string> {
    let content = ''
    let reasoning = ''
    let buffer = ''
    let received = 0
    let reportedUsage: ReportedUsage | undefined
    const decoder = new TextDecoder()
    const merge = (current: string, next: string): string =>
      next.length >= current.length ? next : current + next
    const consume = (rawLine: string): void => {
      const line = rawLine.trim()
      if (!line.startsWith('data:')) return
      const payload = line.slice(5).trim()
      if (payload === '' || payload === '[DONE]') return
      try {
        const chunk = JSON.parse(payload) as {
          choices?: Array<{
            delta?: { content?: string; reasoning_content?: string; reasoning?: string }
            message?: { content?: string; reasoning_content?: string; reasoning?: string }
          }>
          usage?: ReportedUsage
        }
        if (chunk.usage) reportedUsage = chunk.usage
        const delta = chunk.choices?.[0]?.delta
        const message = chunk.choices?.[0]?.message
        if (delta?.content) content += delta.content
        else if (message?.content) content = merge(content, message.content)
        const dr = delta?.reasoning_content ?? delta?.reasoning
        const mr = message?.reasoning_content ?? message?.reasoning
        if (dr) reasoning += dr
        else if (mr) reasoning = merge(reasoning, mr)
      } catch {
        // keepalive / partial junk - ignore
      }
    }
    const reader = body.getReader()
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        received += value.length
        if (received > this.maxStreamChars) {
          throw new Error(
            `LLM stream cap exceeded (${this.maxStreamChars} chars): runaway generation ` +
              `(repetition loop?). Raise RULITH_LLM_MAX_STREAM_CHARS only if the output ` +
              `is legitimately this large.`,
          )
        }
        buffer += decoder.decode(value, { stream: true })
        let nl = buffer.indexOf('\n')
        while (nl >= 0) {
          consume(buffer.slice(0, nl))
          buffer = buffer.slice(nl + 1)
          nl = buffer.indexOf('\n')
        }
      }
    } catch (error) {
      // The call died mid-stream (timeout, stream cap). The backend still
      // burned tokens generating what arrived - record the estimate so DNF
      // rows show their real cost instead of a free-looking zero.
      this.recordUsage(undefined, messages, content + reasoning)
      throw error
    } finally {
      await reader.cancel().catch(() => {})
    }
    buffer += decoder.decode()
    if (buffer.length > 0) consume(buffer)
    const trimmed = content.trim()
    const reply = trimmed.length > 0 ? trimmed : reasoning
    this.recordUsage(reportedUsage, messages, reply)
    return reply
  }
}

/**
 * Config for the cross-model comparison arms: a SECOND model/endpoint read
 * from the RULITH_LLM_*_B env family. Returns undefined unless
 * RULITH_LLM_MODEL_B is set - that one variable enables the model-B arms in
 * the benches (e.g. the same LM Studio instance serving a second model).
 * Add RULITH_LLM_BASE_URL_B / RULITH_LLM_API_KEY_B for a remote endpoint
 * (deepseek, dashscope compatible-mode, ...).
 *
 * Credential isolation: when a DIFFERENT base URL is configured without its
 * own key, apiKey is forced to '' so model A's RULITH_LLM_API_KEY is never
 * sent to model B's endpoint. Without a base URL B (same endpoint as A),
 * the client's normal key fallback applies.
 */
export function modelBConfigFromEnv(env: NodeJS.ProcessEnv = process.env): LlmConfig | undefined {
  const model = env.RULITH_LLM_MODEL_B?.trim()
  if (model === undefined || model === '') return undefined
  const baseUrl = env.RULITH_LLM_BASE_URL_B?.trim()
  const apiKey = env.RULITH_LLM_API_KEY_B ?? (baseUrl ? '' : undefined)
  return {
    model,
    ...(baseUrl ? { baseUrl } : {}),
    ...(apiKey !== undefined ? { apiKey } : {}),
  }
}

/**
 * Conservative JSON repair for the one common weak-model slip: a doubled / stray
 * closing bracket MID-structure — e.g. `"desired":[...]]` (an extra `]` after a
 * valid array close, audit p6), including a batch that doubles `}` on EVERY op
 * (audit p1). A string-aware stack scan drops only closers that do NOT match the
 * top of the stack, and returns a candidate ONLY if it dropped at least one stray,
 * the strays are a minority (≤ the well-formed closers, floor 4 — so the heal scales
 * with batch size instead of forking at a flat count), AND the result is fully
 * balanced. It never adds missing closers
 * (a truncated object stays unrecovered), never touches brackets inside strings,
 * and leaves legitimately-nested arrays (`[[...]]` match correctly) alone — so it
 * cannot force-repair genuinely scrambled input; JSON.parse + the "tool" check is
 * the final gate.
 */
function repairStrayClosers(s: string): string | undefined {
  const stack: string[] = []
  const out: string[] = []
  let inString = false
  let escaped = false
  let removed = 0
  let kept = 0
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i]
    if (inString) {
      out.push(ch)
      if (escaped) escaped = false
      else if (ch === '\\') escaped = true
      else if (ch === '"') inString = false
      continue
    }
    if (ch === '"') {
      inString = true
      out.push(ch)
      continue
    }
    if (ch === '{' || ch === '[') {
      stack.push(ch)
      out.push(ch)
      continue
    }
    if (ch === '}' || ch === ']') {
      const want = ch === '}' ? '{' : '['
      if (stack.length > 0 && stack[stack.length - 1] === want) {
        stack.pop()
        out.push(ch)
        kept += 1
      } else {
        removed += 1 // stray / mismatched closer — drop it
      }
      continue
    }
    out.push(ch)
  }
  // Strays must be a MINORITY of the closers — at most as many as the well-formed ones
  // (small-input floor of 4). So a batch that uniformly doubles "}" on every op heals at
  // any length (audit p1: ~6 ops, ~6 strays), while genuinely scrambled input where strays
  // dominate is still refused. (860a15f recovered ONE doubled "}"; a flat cap of 4 forked it
  // from six identical ones — same slip, same heal, no fork.)
  if (removed === 0 || removed > Math.max(4, kept) || stack.length !== 0) return undefined
  return out.join('')
}

/**
 * Extract the first balanced JSON object carrying a "tool" key, string-aware
 * so braces inside string values (regex patterns, code quotes) don't break
 * the scan. Lifted from the fixture (verification #11 fix). A recovered call
 * (stray bracket stripped/dropped) is flagged `repaired` so the harness can
 * count the help instead of silently scoring a malformed reply as a clean pass.
 */
export function parseToolCall(
  reply: string,
): { tool: string; args?: Record<string, unknown>; note?: string; repaired?: boolean } | undefined {
  const text = reply.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/```(?:json)?/g, '')
  for (let start = text.indexOf('{'); start >= 0; start = text.indexOf('{', start + 1)) {
    let depth = 0
    let inString = false
    let escaped = false
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i]
      if (inString) {
        if (escaped) escaped = false
        else if (ch === '\\') escaped = true
        else if (ch === '"') inString = false
        continue
      }
      if (ch === '"') {
        inString = true
        continue
      }
      if (ch === '{') depth += 1
      if (ch === '}') {
        depth -= 1
        if (depth === 0) {
          const slice = text.slice(start, i + 1)
          // A weak model slips in a STRAY closing bracket the brace-only scan slices in, so JSON.parse
          // throws and the driver loops blind on an empty board — mis-scored as a driving collapse that
          // inflates the measured driving floor. Two conservative recoveries run ONLY after the raw slice
          // fails (so valid input, which parses first, is never touched): (1) strip a trailing stray
          // ]/whitespace before the final } ("...}]}"); (2) drop a doubled/stray closer MID-structure
          // ("desired":[...]] — audit p6) via repairStrayClosers. A recovered call is flagged `repaired`
          // so the harness can SEE the help (count a repair rate) rather than score a malformed reply clean.
          const stripped = slice.replace(/[\]\s]+}$/, '}')
          const deduped = repairStrayClosers(slice)
          // A stray `]` leaves the brace scan's boundary intact (brackets don't move brace depth), so the
          // full object is sliced and `deduped` heals it. A stray `}` (audit p1, qwen3.5) instead drops the
          // brace depth to 0 EARLY, so `slice` is only a partial prefix that cannot re-balance. Widen to the
          // last `}` in the reply and dedup THAT span, so a doubled `}` recovers just like a doubled `]`
          // (closing the brace-vs-bracket asymmetry — same slip, same heal).
          const lastBrace = text.lastIndexOf('}')
          const widened = lastBrace > i ? repairStrayClosers(text.slice(start, lastBrace + 1)) : undefined
          const candidates: Array<{ json: string; repaired: boolean }> = [{ json: slice, repaired: false }]
          if (stripped !== slice) candidates.push({ json: stripped, repaired: true })
          if (deduped !== undefined) candidates.push({ json: deduped, repaired: true })
          if (widened !== undefined && widened !== deduped) candidates.push({ json: widened, repaired: true })
          for (const candidate of candidates) {
            try {
              const parsed = JSON.parse(candidate.json) as { tool?: unknown }
              if (typeof parsed.tool === 'string') {
                const call = parsed as {
                  tool: string
                  args?: Record<string, unknown>
                  note?: string
                  repaired?: boolean
                }
                if (candidate.repaired) call.repaired = true
                return call
              }
            } catch {
              // try the next candidate, then keep scanning
            }
          }
          break
        }
      }
    }
  }
  return undefined
}
