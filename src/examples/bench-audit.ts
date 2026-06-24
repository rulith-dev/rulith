/**
 * bench-audit — A/B benchmark for the board's ERROR-FINDING value.
 *
 * Synthetic ledgers where some rows carry a wrong claimed total
 * (unit × qty ≠ claimed). The task: flag EXACTLY the bad rows — find
 * every seeded error, raise no false alarms. Some problems contain ZERO
 * errors: saying "all clean" when it is clean is part of the discipline
 * being measured (a hallucinated finding is precisely the failure mode
 * the board's derivation gate exists to kill).
 *
 *   baseline  plain chat; model checks mentally, answers
 *             "ANSWER: r2,r5" or "ANSWER: NONE"
 *   board     board-driven; a guard rule (mul + neq) DERIVES bad(row)
 *             for mismatching rows; scored from [derived] bad facts only
 *
 * Metrics per arm: exact-set solve rate, plus row-level precision /
 * recall aggregated over all problems (false positives = hallucinated
 * errors, false negatives = missed errors).
 *
 * Usage mirrors bench-arith:
 *   tsx src/examples/bench-audit.ts --selftest
 *   RULITH_BENCH_N=20 RULITH_BENCH_SEED=7
 *   RULITH_BENCH_ARM=both|all|<comma list>   arm selection, see bench-arms.ts
 *   RULITH_BENCH_ROWS=6-10   rows per ledger (default)
 *   RULITH_LLM_BASE_URL / RULITH_LLM_MODEL / RULITH_LLM_TIMEOUT_MS
 *   RULITH_LLM_MODEL_B       model B: enables the cross-model arms
 *     (+ RULITH_LLM_BASE_URL_B / RULITH_LLM_API_KEY_B for a remote endpoint);
 *     default "both" then runs the three-way comparison A+board vs B bare vs A bare
 */
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { getLogicContext } from '../engine/logic-context.js'
import { runAgentTask, type ChatModel } from '../agent/task-loop.js'
import { ToolRegistry } from '../agent/tools.js'
import { LlmClient, modelBConfigFromEnv, type ChatMessage } from '../agent/llm.js'
import { addUsage, captureTranscript, emitTranscript, emptyUsageTally, fmtUsage, resolveArms, transcriptMode, usageRow, type BenchArm, type UsageTally } from './bench-arms.js'
import type { ClientFactory } from './bench-pool.js'
import { pathToFileURL } from 'node:url'

// ---------------------------------------------------------------- problems

type Row = { id: string; unit: number; qty: number; claimed: number }
export type Problem = { id: number; rows: Row[]; badIds: string[]; text: string }

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const ROW_RANGE = /^(\d+)-(\d+)$/.exec(process.env.RULITH_BENCH_ROWS ?? '6-10')
const MIN_ROWS = Math.max(2, Number(ROW_RANGE?.[1] ?? 6))
const MAX_ROWS = Math.max(MIN_ROWS, Number(ROW_RANGE?.[2] ?? 10))

export function generateProblem(rng: () => number, id: number): Problem {
  const rowCount = MIN_ROWS + Math.floor(rng() * (MAX_ROWS - MIN_ROWS + 1))
  // 0..3 seeded errors; ~1 in 5 ledgers is fully clean (tests false-alarm discipline).
  const errorCount = rng() < 0.2 ? 0 : 1 + Math.floor(rng() * 3)
  const badIndices = new Set<number>()
  while (badIndices.size < Math.min(errorCount, rowCount)) {
    badIndices.add(Math.floor(rng() * rowCount))
  }
  const rows: Row[] = []
  for (let i = 0; i < rowCount; i += 1) {
    const unit = 10_007 + Math.floor(rng() * 89_000) // 5-digit price
    const qty = 11 + Math.floor(rng() * 880) // 2-3 digit quantity
    const trueTotal = unit * qty
    let claimed = trueTotal
    if (badIndices.has(i)) {
      // Plausible corruption: small offset or a swapped digit pair - the
      // kind a skimming eye accepts and exact recomputation catches.
      const mode = rng()
      if (mode < 0.5) {
        claimed = trueTotal + (1 + Math.floor(rng() * 90)) * (rng() < 0.5 ? -1 : 1)
      } else {
        const s = String(trueTotal).split('')
        const k = Math.floor(rng() * (s.length - 1))
        if (s[k] !== s[k + 1]) {
          ;[s[k], s[k + 1]] = [s[k + 1]!, s[k]!]
          claimed = Number(s.join(''))
        } else {
          claimed = trueTotal + 10
        }
      }
    }
    rows.push({ id: `r${i + 1}`, unit, qty, claimed })
  }
  const badIds = rows.filter((_, i) => badIndices.has(i)).map((r) => r.id)
  const text =
    `A ledger claims these line totals: ` +
    rows.map((r) => `${r.id}: ${r.qty} x ${r.unit} = ${r.claimed}`).join('; ') +
    `. Some claimed totals may be WRONG. Identify exactly which rows are wrong ` +
    `(it is possible that none are).`
  return { id, rows, badIds, text }
}

// ---------------------------------------------------------------- baseline arm

function baselineMessages(p: Problem): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You are a meticulous auditor. Recompute every row exactly, then give your ' +
        'verdict on the last line as: ANSWER: <comma-separated wrong row ids> ' +
        'or ANSWER: NONE if every row is correct.',
    },
    { role: 'user', content: p.text },
  ]
}

export function parseBaselineAnswer(reply: string): string[] | undefined {
  const tagged = [...reply.matchAll(/ANSWER:\s*([^\n]+)/gi)].pop()?.[1]
  if (!tagged) return undefined
  if (/\bnone\b/i.test(tagged)) return []
  const ids = tagged.match(/r\d+/gi)
  return ids ? [...new Set(ids.map((s) => s.toLowerCase()))] : undefined
}

// ---------------------------------------------------------------- board arm

function boardGoal(p: Problem): string {
  return (
    `${p.text} Assert one row(id, unit, qty, claimed) fact per row, then add ONE rule ` +
    `that recomputes each row with the mul built-in and derives bad(id) ONLY for rows ` +
    `whose claimed total differs (use neq on the recomputed vs claimed value - do NOT ` +
    `check any arithmetic yourself). The bad(...) facts must be DERIVED. Then record ` +
    `the result listing the bad rows (or that none are bad).`
  )
}

type BoardScore = {
  ok: boolean
  flagged: string[]
  falsePositives: number
  falseNegatives: number
  /** bad(...) facts that were asserted rather than derived (laundering attempt). */
  assertedBad: number
  /** The board derived NOTHING (ops never landed / rules never fired) — a true driving collapse. NOT
   *  the same as flagging the WRONG rows (which derives bad facts): that is a modeling error, and the
   *  driving-floor metric must not count it as an empty board. */
  emptyBoard: boolean
}

function scoreBoard(store: MemorySpaceStore, spaceId: string, p: Problem): BoardScore {
  const facts = getLogicContext(store, spaceId).facts
  const badFacts = facts.filter((f) => f.atom.predicate === 'bad')
  const flagged = [
    ...new Set(
      badFacts
        .filter((f) => f.derived)
        .flatMap((f) => Object.values(f.atom.args ?? {}).map(String))
        .filter((v) => /^r\d+$/i.test(v))
        .map((v) => v.toLowerCase()),
    ),
  ]
  const assertedBad = badFacts.filter((f) => !f.derived).length
  const truth = new Set(p.badIds)
  const falsePositives = flagged.filter((id) => !truth.has(id)).length
  const falseNegatives = p.badIds.filter((id) => !flagged.includes(id)).length
  return {
    ok: falsePositives === 0 && falseNegatives === 0 && assertedBad === 0,
    flagged,
    falsePositives,
    falseNegatives,
    assertedBad,
    // empty = the closure derived nothing; flagging the WRONG rows derives bad facts (modeling error), not empty.
    emptyBoard: !facts.some((f) => f.derived),
  }
}

// Optional per-turn transcript capture (RULITH_BENCH_TRANSCRIPT=1), same as bench-arith — so an
// audit board failure can be post-mortemed from the model's actual replies (this model is ~100% on
// audit, so any miss is worth the replies).
const capReply = (reply: string, cap = 4000): string =>
  reply.length > cap ? `${reply.slice(0, cap)} ...[+${reply.length - cap} chars]` : reply

async function runBoardArm(
  llm: ChatModel,
  p: Problem,
  maxTurns: number,
): Promise<BoardScore & { turns: number; transcript?: string[] }> {
  const store = new MemorySpaceStore()
  let spaceId = ''
  let turns = 0
  const transcript: string[] = []
  const tapped: ChatModel = captureTranscript(transcriptMode())
    ? { chat: async (m) => { const r = await llm.chat(m); transcript.push(capReply(r)); return r } }
    : llm
  await runAgentTask({
    store,
    llm: tapped,
    reg: new ToolRegistry(),
    rootDir: process.cwd(),
    goal: boardGoal(p),
    maxTurns,
    onContext: (info) => {
      spaceId = info.spaceId
    },
    onTurn: () => {
      turns += 1
    },
  })
  const score = scoreBoard(store, spaceId, p)
  return { ...score, turns, ...(emitTranscript(transcriptMode(), !score.ok) ? { transcript } : {}) }
}

function scoreBaseline(answer: string[] | undefined, p: Problem): {
  ok: boolean
  falsePositives: number
  falseNegatives: number
} {
  if (answer === undefined) {
    return { ok: false, falsePositives: 0, falseNegatives: p.badIds.length }
  }
  const truth = new Set(p.badIds)
  const falsePositives = answer.filter((id) => !truth.has(id)).length
  const falseNegatives = p.badIds.filter((id) => !answer.includes(id)).length
  return { ok: falsePositives === 0 && falseNegatives === 0, falsePositives, falseNegatives }
}

type BaselineOutcome = {
  ok: boolean
  falsePositives: number
  falseNegatives: number
  dnf: boolean
  row: Record<string, unknown>
}

/** One bare-chat audit - shared by the model A and model B bare arms. */
async function runBaselineArm(client: ChatModel, p: Problem): Promise<BaselineOutcome> {
  const t0 = Date.now()
  try {
    const reply = await client.chat(baselineMessages(p))
    const answer = parseBaselineAnswer(reply)
    const s = scoreBaseline(answer, p)
    return { ...s, dnf: false, row: { ...s, answer, ms: Date.now() - t0 } }
  } catch (error) {
    return {
      ok: false,
      falsePositives: 0,
      falseNegatives: 0,
      dnf: true,
      row: { ok: false, dnf: true, error: String(error).slice(0, 120), ms: Date.now() - t0 },
    }
  }
}

// ---------------------------------------------------------------- selftest

/** Scripted model: asserts the rows, writes the neq guard rule, records, done. */
class ScriptedAuditModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) {
      const ops: unknown[] = this.p.rows.map((r) => ({
        op: 'assert_fact',
        id: `F_${r.id}`,
        predicate: 'row',
        args: { id: r.id, unit: r.unit, qty: r.qty, claimed: r.claimed },
      }))
      ops.push({
        op: 'add_axiom',
        id: 'ax_bad',
        label: 'bad row: recomputed total differs from claimed',
        when: [
          { predicate: 'row', args: { id: '?r', unit: '?u', qty: '?q', claimed: '?c' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
          { predicate: 'neq', args: { left: '?t', right: '?c' } },
        ],
        then: [{ predicate: 'bad', args: { id: '?r' } }],
      })
      return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops }, note: 'recompute all rows' })
    }
    if (this.step === 2) {
      return JSON.stringify({
        tool: 'update_working_memory',
        args: {
          operations: [
            { op: 'record_result', id: 'res', label: 'audit done', summary: 'bad rows derived by guard rule' },
          ],
        },
        note: 'record',
      })
    }
    return JSON.stringify({ tool: 'done', args: { summary: 'audit complete' } })
  }
}

async function selftest(): Promise<void> {
  // Find a seeded problem WITH errors and one WITHOUT, prove exact scoring on both.
  const rng = mulberry32(7)
  const pool = Array.from({ length: 30 }, (_, i) => generateProblem(rng, i + 1))
  const withErrors = pool.find((p) => p.badIds.length > 0)
  const clean = pool.find((p) => p.badIds.length === 0)
  if (!withErrors || !clean) throw new Error('selftest: generator did not produce both kinds')

  for (const p of [withErrors, clean]) {
    const score = await runBoardArm(new ScriptedAuditModel(p), p, 6)
    if (!score.ok || score.falsePositives !== 0 || score.falseNegatives !== 0) {
      throw new Error(`selftest board failed on problem ${p.id}: ${JSON.stringify(score)}`)
    }
  }

  // Parser: ids, NONE, garbage.
  if (JSON.stringify(parseBaselineAnswer('blah\nANSWER: r2, R5')) !== JSON.stringify(['r2', 'r5'])) {
    throw new Error('selftest: id parse failed')
  }
  if (JSON.stringify(parseBaselineAnswer('ANSWER: NONE')) !== '[]') {
    throw new Error('selftest: NONE parse failed')
  }
  if (parseBaselineAnswer('no verdict given') !== undefined) {
    throw new Error('selftest: missing-answer parse failed')
  }

  // Baseline scorer: a false positive must fail the problem.
  const fp = scoreBaseline([...withErrors.badIds, 'r999'], withErrors)
  if (fp.ok || fp.falsePositives !== 1) throw new Error('selftest: FP scoring failed')

  // Bare-arm helper (shared by the model A and model B bare arms): scripted
  // exact verdicts score on both problem kinds; a dead endpoint is a DNF row.
  for (const p of [withErrors, clean]) {
    const verdict = `ANSWER: ${p.badIds.length === 0 ? 'NONE' : p.badIds.join(', ')}`
    const r = await runBaselineArm({ chat: async () => verdict }, p)
    if (!r.ok || r.dnf) throw new Error(`selftest: bare arm failed on problem ${p.id}`)
  }
  const deadBare = await runBaselineArm(
    {
      chat: async () => {
        throw new Error('endpoint down')
      },
    },
    withErrors,
  )
  if (deadBare.ok || !deadBare.dnf) throw new Error('selftest: bare arm must record a DNF row')

  console.log('bench-audit selftest PASSED (derived-only flags; FP/FN accounting + bare arm sane)')
}

// ---------------------------------------------------------------- per-problem

export type AuditTally = { ok: number; fp: number; fn: number; laundered: number; dnf: number; tok: UsageTally }
export const emptyAuditTally = (): AuditTally => ({ ok: 0, fp: 0, fn: 0, laundered: 0, dnf: 0, tok: emptyUsageTally() })

/**
 * Run ONE audit problem across the active arms; return its log row and mutate
 * the shared tally. Shared by the serial main() and the pool runner.
 * `makeClient` supplies the model (shared in serial, per-task in the pool).
 */
export async function runAuditProblemRow(
  p: Problem,
  ctx: { active: readonly BenchArm[]; maxTurns: number; tally: Map<BenchArm, AuditTally>; makeClient: ClientFactory },
): Promise<Record<string, unknown>> {
  const { active, maxTurns, tally, makeClient } = ctx
  const row: Record<string, unknown> = { id: p.id, badIds: p.badIds }
  for (const arm of active) {
    const t = tally.get(arm)!
    const client = makeClient(arm)
    client.consumeUsage?.() // open this arm's token window
    console.error(`[audit p${p.id}] ${arm} arm started`)
    let armRow: Record<string, unknown>
    if (arm === 'baseline' || arm === 'baseline_b') {
      const outcome = await runBaselineArm(client, p)
      if (outcome.ok) t.ok += 1
      if (outcome.dnf) t.dnf += 1
      t.fp += outcome.falsePositives
      t.fn += outcome.falseNegatives
      armRow = outcome.row
    } else {
      const t0 = Date.now()
      try {
        const s = await runBoardArm(client, p, maxTurns)
        if (s.ok) t.ok += 1
        t.fp += s.falsePositives
        t.fn += s.falseNegatives
        t.laundered += s.assertedBad
        armRow = { ...s, ms: Date.now() - t0 }
      } catch (error) {
        t.dnf += 1
        armRow = { ok: false, dnf: true, error: String(error).slice(0, 120), ms: Date.now() - t0 }
      }
    }
    const u = client.consumeUsage?.()
    if (u) {
      if (u.calls > 0) armRow.tokens = usageRow(u)
      addUsage(t.tok, u)
    }
    row[arm] = armRow
  }
  return row
}

// ---------------------------------------------------------------- main

async function main(): Promise<void> {
  if (process.argv.includes('--selftest')) {
    await selftest()
    return
  }

  const N = Number(process.env.RULITH_BENCH_N ?? 20)
  const seed = Number(process.env.RULITH_BENCH_SEED ?? 7)
  const bConfig = modelBConfigFromEnv()
  const arms = resolveArms(process.env.RULITH_BENCH_ARM, bConfig !== undefined)
  const maxTurns = Number(process.env.RULITH_BENCH_TURNS ?? 10)
  const skip = Number(process.env.RULITH_BENCH_SKIP ?? 0)
  // TAKE: window length after skip (partitioned concurrency — worker k runs ids [skip+1 .. skip+take]).
  // Unset/0 ⇒ run through the end. Same seed across workers keeps ids 1..N aligned.
  const take = Number(process.env.RULITH_BENCH_TAKE ?? 0)
  const rng = mulberry32(seed)
  const problems = Array.from({ length: N }, (_, i) => generateProblem(rng, i + 1)).slice(skip, take > 0 ? skip + take : undefined)

  const llm = new LlmClient()
  const llmB = bConfig ? new LlmClient(bConfig) : undefined
  mkdirSync('logs', { recursive: true })
  const logPath = join('logs', `bench-audit-${Date.now()}.jsonl`)
  const log = (entry: unknown): void => appendFileSync(logPath, `${JSON.stringify(entry)}\n`)
  // Attribution header: model identity from env, not from any UI banner
  // (lesson of validation #27 - environment labels are assertions too).
  log({
    type: 'config',
    bench: 'audit',
    startedAt: new Date().toISOString(),
    model: process.env.RULITH_LLM_MODEL ?? '(client default: local-model)',
    note: process.env.RULITH_BENCH_NOTE,
    baseUrl: process.env.RULITH_LLM_BASE_URL ?? '(client default: http://127.0.0.1:1234)',
    stream: process.env.RULITH_LLM_STREAM !== '0',
    timeoutMs: Number(process.env.RULITH_LLM_TIMEOUT_MS ?? 180000),
    maxTokens: Number(process.env.RULITH_MAX_TOKENS ?? 8000),
    arms: [...arms].join(','),
    modelB: bConfig?.model ?? null,
    baseUrlB: bConfig ? (bConfig.baseUrl ?? '(model A endpoint)') : null,
    seed,
    n: N,
    skip,
    maxTurns,
  })

  const ORDER: readonly BenchArm[] = ['baseline', 'baseline_b', 'board', 'board_b']
  const active = ORDER.filter((a) => arms.has(a))
  const makeClient: ClientFactory = (a) => (a.endsWith('_b') ? llmB! : llm)
  const tally = new Map<BenchArm, AuditTally>(active.map((a) => [a, emptyAuditTally()]))
  const ran = problems.length

  for (const p of problems) {
    const row = await runAuditProblemRow(p, { active, maxTurns, tally, makeClient })
    log(row)
    console.log(JSON.stringify(row))
  }

  console.log('---')
  const label = (a: BenchArm): string =>
    a.endsWith('_b') ? `${a} [${bConfig?.model}]` : `${a} [${process.env.RULITH_LLM_MODEL ?? 'local-model'}]`
  for (const arm of active) {
    const t = tally.get(arm)!
    console.log(
      `${label(arm)}: ${t.ok}/${ran} exact-set; false alarms ${t.fp}, missed ${t.fn} (${t.dnf} DNF)`,
    )
    if (arm === 'board' || arm === 'board_b') {
      console.log(`${label(arm)}: ${t.laundered} bad(...) facts asserted instead of derived (counted as FAIL)`)
    }
    if (t.tok.calls > 0) console.log(`${label(arm)}: ${fmtUsage(t.tok)}`)
  }
  if (skip > 0) console.log(`(resumed at problem ${skip + 1})`)
  console.log(`log: ${logPath}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}