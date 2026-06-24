/**
 * Shared arm selection for the A/B benches (bench-arith / bench-audit /
 * bench-repair), extended with the CROSS-MODEL comparison arms.
 *
 * Classic arms run model A (RULITH_LLM_MODEL):
 *   baseline    model A bare (plain chat, no board)
 *   board       model A driving the reasoning board
 * Cross-model arms run model B (RULITH_LLM_MODEL_B, see modelBConfigFromEnv):
 *   baseline_b  model B bare - the "first-tier model, no board" reference
 *   board_b     model B driving the board (completes the 2x2)
 *
 * RULITH_BENCH_ARM values:
 *   both (default)  baseline + board; PLUS baseline_b when model B is
 *                   configured - so setting RULITH_LLM_MODEL_B is the only
 *                   step needed for the experiment's central three-way
 *                   comparison (A+board vs B bare vs A bare: can a
 *                   second-tier model with the board match a first-tier
 *                   model without it?)
 *   all             the full 2x2 (requires model B)
 *   <comma list>    explicit arms, e.g. "baseline_b" alone to add a model-B
 *                   column next to an existing log without re-running the
 *                   model A arms (same seed = identical problems, rows
 *                   merge by id)
 */
import type { LlmUsage } from '../agent/llm.js'

export type BenchArm = 'baseline' | 'board' | 'baseline_b' | 'board_b'

/**
 * Token tally for one arm. Wall-clock comparisons across endpoints are
 * confounded by serving speed (cloud streams far more tokens/s than a local
 * GPU); tokens are the unit that compares fairly across hardware, and they
 * quantify the central efficiency claim: the board replaces long
 * chain-of-thought with a few small tool calls.
 */
export type UsageTally = { in: number; out: number; calls: number; est: number }

export const emptyUsageTally = (): UsageTally => ({ in: 0, out: 0, calls: 0, est: 0 })

export function addUsage(t: UsageTally, u: LlmUsage): void {
  t.in += u.promptTokens
  t.out += u.completionTokens
  t.calls += u.calls
  t.est += u.estimatedCalls
}

/** Per-problem JSONL row fragment: raw integers, estimation flagged only when present. */
export function usageRow(u: LlmUsage): Record<string, number> {
  return {
    in: u.promptTokens,
    out: u.completionTokens,
    calls: u.calls,
    ...(u.estimatedCalls > 0 ? { estimatedCalls: u.estimatedCalls } : {}),
  }
}

/** Console summary fragment for an arm's token totals. */
export function fmtUsage(t: UsageTally): string {
  const base = `tokens in=${t.in} out=${t.out} calls=${t.calls}`
  return t.est > 0 ? `${base} (${t.est} estimated)` : base
}

/**
 * Transcript-capture mode (RULITH_BENCH_TRANSCRIPT). The transcript is the per-turn model replies,
 * kept for post-mortem. For the DRIVING benches (arith / audit / coding) the full firehose is rarely
 * what you want, so `onfail` still captures but EMITS only for a failed board arm. `=1` / `all` stays
 * full capture — the TRUST benches rely on it to see attempts the board CORRECTLY blocked (a blocked
 * attack is not a "fail", so on-fail would hide exactly what you want to inspect there).
 *   off    (unset / 0)              no capture
 *   all    (1 / all)               capture every arm  (back-compat; the trust-bench mode)
 *   onfail (onfail / fail / error) capture always, EMIT only when the board arm failed
 */
export type TranscriptMode = 'off' | 'all' | 'onfail'

export function transcriptMode(env: NodeJS.ProcessEnv = process.env): TranscriptMode {
  const v = (env.RULITH_BENCH_TRANSCRIPT ?? '').trim().toLowerCase()
  if (v === '1' || v === 'all') return 'all'
  if (v === 'onfail' || v === 'fail' || v === 'error') return 'onfail'
  return 'off'
}

/** Capture the transcript at all? (both `all` and `onfail` capture; only the emit step differs). */
export const captureTranscript = (mode: TranscriptMode): boolean => mode !== 'off'

/** Emit the captured transcript into the row, given the board arm's outcome (failed = emit on onfail). */
export const emitTranscript = (mode: TranscriptMode, failed: boolean): boolean =>
  mode === 'all' || (mode === 'onfail' && failed)

const ALL_ARMS: readonly BenchArm[] = ['baseline', 'board', 'baseline_b', 'board_b']

export function resolveArms(spec: string | undefined, hasModelB: boolean): Set<BenchArm> {
  const trimmed = (spec ?? '').trim().toLowerCase()
  const requireB = (wanted: string): void => {
    if (!hasModelB) {
      throw new Error(
        `RULITH_BENCH_ARM=${wanted} needs a second model: set RULITH_LLM_MODEL_B ` +
          `(and RULITH_LLM_BASE_URL_B / RULITH_LLM_API_KEY_B for a remote endpoint).`,
      )
    }
  }
  if (trimmed === '' || trimmed === 'both') {
    const arms = new Set<BenchArm>(['baseline', 'board'])
    if (hasModelB) arms.add('baseline_b')
    return arms
  }
  if (trimmed === 'all') {
    requireB('all')
    return new Set(ALL_ARMS)
  }
  const arms = new Set<BenchArm>()
  for (const token of trimmed.split(',').map((t) => t.trim()).filter((t) => t !== '')) {
    if (!(ALL_ARMS as readonly string[]).includes(token)) {
      throw new Error(
        `unknown RULITH_BENCH_ARM "${token}" - valid: ${ALL_ARMS.join(', ')}, both, all ` +
          `(comma-separate to combine).`,
      )
    }
    if (token.endsWith('_b')) requireB(token)
    arms.add(token as BenchArm)
  }
  if (arms.size === 0) {
    throw new Error('RULITH_BENCH_ARM resolved to no arms - nothing would run.')
  }
  return arms
}
