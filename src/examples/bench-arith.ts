/**
 * bench-arith — A/B benchmark for the board's exact-arithmetic value.
 *
 * GSM-Hard-style big-number invoice problems (7-digit unit prices ×
 * 3-digit quantities, 2–4 lines, exact grand total) solved by the SAME
 * local model two ways:
 *
 *   baseline  plain chat, model computes mentally, answers "ANSWER: <n>"
 *   board     board-driven task loop; costs and total must end as
 *             closure-DERIVED facts (mul/add built-ins do the arithmetic)
 *
 * Scoring is from ground truth (exact integers, all within 2^53 so the
 * kernel's exact-or-fail contract applies). The board arm is scored from
 * the BOARD, not from prose: a problem counts as solved only if every
 * per-line cost AND the grand total appear as [derived] facts with the
 * exact values. This measures the claim that matters: not "the model
 * said a number" but "the closure stands behind the number".
 *
 * Usage:
 *   tsx src/examples/bench-arith.ts --selftest     # scripted model, no LLM needed
 *   tsx src/examples/bench-arith.ts                # all configured arms, real model
 *   RULITH_BENCH_ARM=both|all|<comma list>         # arm selection, see bench-arms.ts
 *   RULITH_BENCH_N=20 RULITH_BENCH_SEED=7          # problem count / seed
 *   RULITH_LLM_BASE_URL / RULITH_LLM_MODEL         # model A, as in the other fixtures
 *   RULITH_LLM_MODEL_B                             # model B: enables the cross-model
 *     arms (+ RULITH_LLM_BASE_URL_B / RULITH_LLM_API_KEY_B for a remote endpoint).
 *     With model B set, the default "both" runs the THREE-WAY comparison
 *     (A+board vs B bare vs A bare) - the capability-escalation question:
 *     does a second-tier model with the board match a first-tier model without?
 *
 * Results: console summary + JSONL per-problem log under logs/.
 */
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { formatLogicContextAsText, getLogicContext } from '../engine/logic-context.js'
import { groundingOf } from '../engine/premise-provenance.js'
import { runAgentTask, type ChatModel } from '../agent/task-loop.js'
import { ToolRegistry } from '../agent/tools.js'
import { LlmClient, modelBConfigFromEnv, type ChatMessage } from '../agent/llm.js'
import { addUsage, captureTranscript, emitTranscript, emptyUsageTally, fmtUsage, resolveArms, transcriptMode, usageRow, type BenchArm, type UsageTally } from './bench-arms.js'
import type { ClientFactory } from './bench-pool.js'
import { pathToFileURL } from 'node:url'

// ---------------------------------------------------------------- problems

type Line = { item: string; unit: number; qty: number }
export type Problem = { id: number; lines: Line[]; costs: number[]; total: number; text: string }

/** Deterministic PRNG so runs are reproducible and arms see identical problems. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const ITEM_POOL = [
  'turbine', 'gasket', 'manifold', 'sensor', 'actuator', 'bearing',
  'compressor', 'valve', 'rotor', 'flange', 'coupling', 'injector',
]

/**
 * Difficulty knobs (mental arithmetic cost explodes with digit width and
 * line count; the kernel is indifferent as long as every product and the
 * total stay within 2^53 — guarded below):
 *   RULITH_BENCH_UNIT_DIGITS  unit price digits, 2..8 (default 7)
 *   RULITH_BENCH_QTY_DIGITS   quantity digits, 1..5 (default 3)
 *   RULITH_BENCH_LINES        line-count range, e.g. "5-8" (default "2-4")
 */
const UNIT_DIGITS = Math.min(8, Math.max(2, Number(process.env.RULITH_BENCH_UNIT_DIGITS ?? 7)))
const QTY_DIGITS = Math.min(5, Math.max(1, Number(process.env.RULITH_BENCH_QTY_DIGITS ?? 3)))
const LINE_RANGE = /^(\d+)-(\d+)$/.exec(process.env.RULITH_BENCH_LINES ?? '2-4')
const MIN_LINES = Math.max(1, Number(LINE_RANGE?.[1] ?? 2))
const MAX_LINES = Math.min(ITEM_POOL.length, Math.max(MIN_LINES, Number(LINE_RANGE?.[2] ?? 4)))

function randomWithDigits(rng: () => number, digits: number): number {
  const lo = 10 ** (digits - 1)
  return lo + Math.floor(rng() * (9 * lo - 2)) + 1
}

export function generateProblem(rng: () => number, id: number): Problem {
  const lineCount = MIN_LINES + Math.floor(rng() * (MAX_LINES - MIN_LINES + 1))
  const names = [...ITEM_POOL].sort(() => rng() - 0.5).slice(0, lineCount)
  const lines: Line[] = names.map((item) => ({
    item,
    unit: randomWithDigits(rng, UNIT_DIGITS),
    qty: randomWithDigits(rng, QTY_DIGITS),
  }))
  const costs = lines.map((l) => l.unit * l.qty)
  const total = costs.reduce((a, b) => a + b, 0)
  // Exactness guard: the kernel's contract is exact-or-fail within 2^53;
  // problems must never leave that range or the board arm would (rightly)
  // refuse. Max with 8-digit unit × 5-digit qty × 12 lines ≈ 1.2e14, safe.
  if (!Number.isSafeInteger(total)) {
    throw new Error(`generated total ${total} exceeds 2^53; lower the difficulty knobs`)
  }
  const text =
    `An invoice has ${lineCount} line items: ` +
    lines.map((l) => `${l.qty} units of "${l.item}" at ${l.unit} cents each`).join('; ') +
    `. Compute the EXACT cost of each line in cents and the EXACT grand total in cents.`
  return { id, lines, costs, total, text }
}

// ---------------------------------------------------------------- baseline arm

function baselineMessages(p: Problem): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You are a precise accountant. Work step by step, then give the grand total ' +
        'on the last line in exactly this form: ANSWER: <integer>',
    },
    { role: 'user', content: p.text },
  ]
}

export function parseBaselineAnswer(reply: string): number | undefined {
  const tagged = [...reply.matchAll(/ANSWER:\s*(-?[\d,]+)/gi)].pop()
  const raw = tagged?.[1] ?? [...reply.matchAll(/-?\d[\d,]{2,}/g)].pop()?.[0]
  if (!raw) return undefined
  const n = Number(raw.replaceAll(',', ''))
  return Number.isFinite(n) ? n : undefined
}

type BaselineOutcome = { ok: boolean; dnf: boolean; row: Record<string, unknown> }

/** One bare-chat attempt - shared by the model A and model B bare arms.
 *  A model that cannot produce an answer within the call budget is a DNF
 *  (did not finish): recorded as a failure row, never a crashed run. */
async function runBaselineArm(client: ChatModel, p: Problem): Promise<BaselineOutcome> {
  const t0 = Date.now()
  try {
    const reply = await client.chat(baselineMessages(p))
    const answer = parseBaselineAnswer(reply)
    const ok = answer === p.total
    return {
      ok,
      dnf: false,
      row: {
        ok,
        answer,
        ms: Date.now() - t0,
        ...(emitTranscript(transcriptMode(), !ok) ? { reply: capReply(reply, 6000) } : {}),
      },
    }
  } catch (error) {
    return {
      ok: false,
      dnf: true,
      row: { ok: false, dnf: true, error: String(error).slice(0, 120), ms: Date.now() - t0 },
    }
  }
}

// ---------------------------------------------------------------- board arm

function boardGoal(p: Problem): string {
  // NOTE: kept deliberately SHORT. An earlier verbose variant (a pile of "do NOT assert cost /
  // it will conflict / taint the total" prohibitions) measurably HURT weak-model driving (empty
  // boards, e.g. arith p2 on 27b-mtp) for zero benefit: bare-asserted-cost pollution is already
  // caught at the BOARD layer (functional_dependency conflict + derive_aggregate refusal), which
  // does not depend on the model reading any prompt. Treat pollution at the board, not by piling
  // prohibitions onto the driver.
  return (
    `${p.text} Assert one line(item, unit, qty) fact per line item, then make the BOARD ` +
    `do all arithmetic with the mul/add built-ins in rule bodies (do NOT compute any ` +
    `product or sum yourself): derive cost(item, total) for every line and a single ` +
    `grand_total(value) fact. The costs and grand_total must be DERIVED facts (closure-` +
    `computed), then record the result.`
  )
}

type BoardScore = {
  ok: boolean
  lineHits: number
  lineCount: number
  totalDerived: boolean
  /** A fact with the right number exists but only as a bare assertion. */
  totalAssertedOnly: boolean
  /** The board derived NOTHING (ops never landed, or rules never fired) — a true driving collapse.
   *  DISTINCT from a full board with WRONG values (which derives facts, just not the right ones). The
   *  driving-floor metric must count THIS, not "0 correct hits" — the latter conflates modeling errors
   *  (the model drove fine but wrote a wrong rule) into the driving-collapse count. */
  emptyBoard: boolean
  /** Final board text, attached ONLY on failure (post-mortem evidence). */
  boardDump?: string
  /** Structured board self-critique (functional conflicts, contradictions, self-sealed
   *  goals, asserted findings, vacuous/unfirable rules, ...), attached on failure. Unlike
   *  the capped boardDump - whose critique/conflicts tail the 6000-char cap can clip on a
   *  big board - this is the full structured list, so ONE failure row reverse-engineers
   *  MANY problems at once (the arith p10 lesson). */
  diagnostics?: string[]
  /** Raw per-turn model replies, when RULITH_BENCH_TRANSCRIPT is on (=1/all always; =onfail only on a failed arm). */
  transcript?: string[]
  /** Set when the arm died mid-task (turn-level budget exhaustion etc.). */
  dnfError?: string
}

function capReply(reply: string, cap = 4000): string {
  return reply.length > cap ? `${reply.slice(0, cap)} ...[+${reply.length - cap} chars]` : reply
}

function scoreBoard(store: MemorySpaceStore, spaceId: string, p: Problem): BoardScore {
  const facts = getLogicContext(store, spaceId).facts
  const values = (args: Record<string, unknown> | undefined): unknown[] =>
    Object.values(args ?? {})
  let lineHits = 0
  for (let i = 0; i < p.lines.length; i += 1) {
    const line = p.lines[i]!
    const want = p.costs[i]!
    const hit = facts.some(
      (f) =>
        f.derived &&
        values(f.atom.args).includes(line.item) &&
        values(f.atom.args).includes(want),
    )
    if (hit) lineHits += 1
  }
  const totalDerived = facts.some((f) => f.derived && values(f.atom.args).includes(p.total))
  const totalAssertedOnly =
    !totalDerived && facts.some((f) => values(f.atom.args).includes(p.total))
  return {
    ok: lineHits === p.lines.length && totalDerived,
    lineHits,
    lineCount: p.lines.length,
    totalDerived,
    totalAssertedOnly,
    // empty = the closure produced nothing; a full-but-wrong board (derived costs/total, just wrong
    // values) is NOT empty — it is a modeling error, not a driving collapse.
    emptyBoard: !facts.some((f) => f.derived),
  }
}

async function runBoardArm(
  llm: ChatModel,
  p: Problem,
  maxTurns: number,
  tag: 'board' | 'board_b' = 'board',
): Promise<BoardScore & { turns: number }> {
  const store = new MemorySpaceStore()
  let spaceId = ''
  let turns = 0
  const transcript: string[] = []
  // ONE timed-out turn is absorbed (real P5: a single runaway generation
  // killed the whole arm at 991s). The loop's parse-failure nudge keeps
  // the task alive; a second budget death aborts for real.
  let budgetDeaths = 0
  const isBudgetShaped = (error: unknown): boolean =>
    (error as { name?: string } | null)?.name === 'TimeoutError' ||
    /stream cap exceeded/i.test(String(error))
  const tolerant: ChatModel = {
    chat: async (messages) => {
      try {
        return await llm.chat(messages)
      } catch (error) {
        if (isBudgetShaped(error) && budgetDeaths === 0) {
          budgetDeaths = 1
          console.error(`[p${p.id}] ${tag} turn timed out - tolerated once, nudging`)
          return (
            'TURN BUDGET EXCEEDED - the previous generation never finished ' +
            '(repetition loop?). Reply now with ONE SMALL JSON tool call.'
          )
        }
        throw error
      }
    },
  }
  const tapped: ChatModel = captureTranscript(transcriptMode())
    ? {
        chat: async (messages) => {
          const reply = await tolerant.chat(messages)
          transcript.push(capReply(reply))
          return reply
        },
      }
    : tolerant
  let dnfError: string | undefined
  try {
    await runAgentTask({
      store,
      llm: tapped,
      reg: new ToolRegistry(),
      rootDir: process.cwd(),
      goal: boardGoal(p),
      maxTurns,
      // NOTE: deliberately NO functional_dependency seed. An earlier ① landing seeded
      // functional_dependency(cost,item) so p10-style pollution (a bare-asserted cost vs the
      // derived one) would surface as a conflict — but on this already-stable model (~98% solo)
      // that unfamiliar config fact in every board input CORRELATED WITH EMPTY-BOARD DRIVING
      // FAILURES for zero benefit (the model never bare-asserts cost). Same trap as an over-long
      // prompt: do not feed the stable driver extra it does not need. The ①/② kernel machinery
      // still exists for domains that explicitly declare a functional dependency; arith doesn't.
      onContext: (info) => {
        spaceId = info.spaceId
      },
      onTurn: () => {
        turns += 1
        console.error(`[p${p.id}] ${tag} turn ${turns}`)
      },
    })
  } catch (error) {
    // Crash forensics (real P5): score whatever the board already holds
    // instead of throwing the evidence away with the exception.
    dnfError = String(error).slice(0, 160)
  }
  const score = scoreBoard(store, spaceId, p)
  if (dnfError !== undefined) {
    score.ok = false
    score.dnfError = dnfError
  }
  if (!score.ok) {
    // Keep the evidence: without the board itself, a failure row cannot be
    // post-mortemed (learned from the 2026-06-12 run, problem 3). Capped so
    // a pathological board cannot flood the JSONL log.
    const failCtx = getLogicContext(store, spaceId)
    // Structured, uncapped diagnostics so one failure row reverse-engineers many problems at
    // once (arith p10), independent of the capped boardDump whose critique tail the cap can clip.
    // Three lenses: (1) the board's standing self-critique (functional conflicts, contradictions,
    // self-sealed, asserted findings, ...); (2) disputed = what a conflict tainted; (3) grounding
    // floor of the headline total = how trustworthy it actually is (its weakest ground premise).
    const diag = failCtx.critique.map((c) => `[${c.kind}] ${c.nodeId}: ${c.message}`)
    const disputed = failCtx.facts.filter((f) => f.disputed).map((f) => f.nodeId)
    if (disputed.length > 0) diag.push(`[disputed] tainted by a conflict: ${disputed.join(', ')}`)
    const total = failCtx.facts.find((f) => f.atom.predicate === 'grand_total')
    if (total) {
      const g = groundingOf(failCtx.facts, total.nodeId)
      diag.push(`[grounding] grand_total floor=${g.weakestTier}${g.ungrounded ? ' (ungrounded)' : ''}`)
    }
    // empty-board / driving failure: NO rules and NO derived facts means the model's ops never
    // landed - the #1 bench failure mode (driving floor), on which all three lenses above are
    // silent (there is nothing on the board to critique). Name it explicitly with how to dig in.
    if (score.emptyBoard) {
      diag.push(
        `[empty-board] driving failure: ${turns} turn(s) produced NO derived facts ` +
          `(ops never landed - malformed JSON, a rejected op, non-tool replies - or a rule that never fired). ` +
          `This is a true driving collapse, NOT a wrong-value board (which would derive facts). ` +
          `Re-run with RULITH_BENCH_TRANSCRIPT=onfail to capture the replies of just the failed arms.`,
      )
    }
    score.diagnostics = diag
    score.boardDump = formatLogicContextAsText(failCtx).slice(0, 6000)
  }
  if (emitTranscript(transcriptMode(), !score.ok)) score.transcript = transcript
  return { ...score, turns }
}

// ---------------------------------------------------------------- selftest

/** Scripted model that drives the board correctly for any generated problem. */
class ScriptedBoardModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) {
      const ops: unknown[] = this.p.lines.map((l, i) => ({
        op: 'assert_fact',
        id: `L${i}`,
        predicate: 'line',
        args: { item: l.item, unit: l.unit, qty: l.qty },
      }))
      ops.push({
        op: 'add_axiom',
        id: 'ax_cost',
        label: 'cost = unit*qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      })
      // Grand total: one rule chaining add over the per-line costs. Written
      // deliberately with the adds BEFORE some cost literals they depend on -
      // the matcher's dependency ordering must absorb that.
      const when: unknown[] = []
      const costVars = this.p.lines.map((l, i) => {
        when.push({ predicate: 'cost', args: { item: l.item, total: `?c${i}` } })
        return `?c${i}`
      })
      let acc = costVars[0]!
      for (let i = 1; i < costVars.length; i += 1) {
        const next = i === costVars.length - 1 ? '?sum' : `?s${i}`
        when.push({ predicate: 'add', args: { left: acc, right: costVars[i]!, result: next } })
        acc = next
      }
      const sumVar = costVars.length === 1 ? costVars[0]! : '?sum'
      ops.push({
        op: 'add_axiom',
        id: 'ax_total',
        label: 'grand total = sum of costs',
        when,
        then: [{ predicate: 'grand_total', args: { value: sumVar } }],
      })
      return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops }, note: 'model the invoice' })
    }
    if (this.step === 2) {
      return JSON.stringify({
        tool: 'update_working_memory',
        args: {
          operations: [
            {
              op: 'record_result',
              id: 'res',
              label: 'totals derived',
              summary: 'all costs and the grand total are closure-derived',
            },
          ],
        },
        note: 'record',
      })
    }
    return JSON.stringify({ tool: 'done', args: { summary: 'derived exact totals' } })
  }
}

/** Shared first-turn ops: line facts + cost rule (+ optionally the sum rule). */
function buildModelingCall(p: Problem, includeTotalRule: boolean): string {
  const ops: unknown[] = p.lines.map((l, i) => ({
    op: 'assert_fact',
    id: `L${i}`,
    predicate: 'line',
    args: { item: l.item, unit: l.unit, qty: l.qty },
  }))
  ops.push({
    op: 'add_axiom',
    id: 'ax_cost',
    label: 'cost = unit*qty',
    when: [
      { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
      { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
    ],
    then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
  })
  if (includeTotalRule) {
    const when: unknown[] = []
    const costVars = p.lines.map((l, i) => {
      when.push({ predicate: 'cost', args: { item: l.item, total: `?c${i}` } })
      return `?c${i}`
    })
    let acc = costVars[0]!
    for (let i = 1; i < costVars.length; i += 1) {
      const next = i === costVars.length - 1 ? '?sum' : `?s${i}`
      when.push({ predicate: 'add', args: { left: acc, right: costVars[i]!, result: next } })
      acc = next
    }
    ops.push({
      op: 'add_axiom',
      id: 'ax_total',
      label: 'grand total = sum of costs',
      when,
      then: [{ predicate: 'grand_total', args: { value: costVars.length === 1 ? costVars[0]! : '?sum' } }],
    })
  }
  return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops }, note: 'model the invoice' })
}

/**
 * Scripted model that derives every line cost but never sums - the
 * failure shape of real run 2026-06-12 problem 3 (lineHits full,
 * totalDerived false). Exists to pin the forensics contract: a failed
 * board arm must come back with a board dump for post-mortems.
 */
class StallingModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) return buildModelingCall(this.p, false)
    return JSON.stringify({ tool: 'done', args: { summary: 'leaving without the total' } })
  }
}

/**
 * Models the lines, then every later turn times out - the failure shape
 * of real run 2026-06-12 problem 5 (board arm died mid-task at 991s).
 * Pins the crash-forensics contract: a turn-level budget death must
 * still score the partial board and carry the dump.
 */
class TimeoutAfterModelingModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) return buildModelingCall(this.p, false)
    throw new DOMException('The operation was aborted due to timeout', 'TimeoutError')
  }
}

/** Times out ONCE mid-task, then finishes properly - tolerance must let it. */
class RecoversAfterTimeoutModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) return buildModelingCall(this.p, true)
    if (this.step === 2) throw new DOMException('The operation was aborted due to timeout', 'TimeoutError')
    if (this.step === 3) {
      return JSON.stringify({
        tool: 'update_working_memory',
        args: { operations: [{ op: 'record_result', id: 'res', label: 'totals derived', summary: 'derived after a hiccup' }] },
      })
    }
    return JSON.stringify({ tool: 'done', args: { summary: 'recovered' } })
  }
}

async function selftest(): Promise<void> {
  const rng = mulberry32(7)
  const p = generateProblem(rng, 1)

  // Board arm with a correct scripted model must score ok with all lines derived.
  const board = await runBoardArm(new ScriptedBoardModel(p), p, 6)
  if (!board.ok || !board.totalDerived || board.lineHits !== board.lineCount) {
    throw new Error(`selftest board arm failed: ${JSON.stringify(board)}`)
  }

  // Baseline parser: tagged answers, comma-grouped, and trailing-number fallback.
  if (parseBaselineAnswer('thinking...\nANSWER: 1,234,567') !== 1234567) {
    throw new Error('selftest: tagged parse failed')
  }
  if (parseBaselineAnswer('the total is 99887766 cents') !== 99887766) {
    throw new Error('selftest: fallback parse failed')
  }
  if (parseBaselineAnswer('no numbers here') !== undefined) {
    throw new Error('selftest: empty parse failed')
  }

  // Scoring guards: a wrong total must not pass.
  const wrong: Problem = { ...p, total: p.total + 1 }
  const store = new MemorySpaceStore()
  const space = store.createSpace({ title: 'wrong' })
  if (scoreBoard(store, space.id, wrong).ok) {
    throw new Error('selftest: empty board must not score ok')
  }

  // Failure forensics: a failed board arm must carry the board itself.
  const stalled = await runBoardArm(new StallingModel(p), p, 4)
  if (stalled.ok) {
    throw new Error('selftest: stalling model must not score ok')
  }
  if (
    stalled.boardDump === undefined ||
    !stalled.boardDump.includes('cost(') ||
    stalled.boardDump.includes('grand_total(')
  ) {
    throw new Error(
      `selftest: failed board arm must include a board dump showing derived costs and no grand_total (got: ${String(stalled.boardDump).slice(0, 200)})`,
    )
  }
  // Hermetic about RULITH_BENCH_TRANSCRIPT: a dev shell usually has it set (=1 is
  // how you capture transcripts), but the off/all/onfail assertions below drive
  // the var themselves. Clear our own copy so the checks hold regardless of the
  // ambient env (this does NOT touch the parent shell's variable). The =1 and
  // onfail blocks each set + delete it, so the section stays self-contained.
  delete process.env.RULITH_BENCH_TRANSCRIPT
  const fine = await runBoardArm(new ScriptedBoardModel(p), p, 6)
  if (fine.boardDump !== undefined) {
    throw new Error('selftest: successful board arm must not carry a dump (log bloat)')
  }
  if (fine.transcript !== undefined) {
    throw new Error('selftest: transcript must stay off unless RULITH_BENCH_TRANSCRIPT=1')
  }
  process.env.RULITH_BENCH_TRANSCRIPT = '1'
  try {
    const taped = await runBoardArm(new ScriptedBoardModel(p), p, 6)
    if (!taped.transcript || taped.transcript.length < 2 || !taped.transcript[0]!.includes('add_axiom')) {
      throw new Error(`selftest: transcript capture failed (got ${JSON.stringify(taped.transcript?.length)})`)
    }
  } finally {
    delete process.env.RULITH_BENCH_TRANSCRIPT
  }

  // onfail: CAPTURE always, EMIT only for a FAILED arm. A passing arm carries none; a stalled one does.
  process.env.RULITH_BENCH_TRANSCRIPT = 'onfail'
  try {
    const okRun = await runBoardArm(new ScriptedBoardModel(p), p, 6)
    if (okRun.transcript !== undefined) {
      throw new Error('selftest: onfail must NOT emit a transcript for a passing board arm')
    }
    const failRun = await runBoardArm(new StallingModel(p), p, 4)
    if (failRun.ok || !failRun.transcript || failRun.transcript.length < 1) {
      throw new Error('selftest: onfail must emit a transcript for a FAILED board arm')
    }
  } finally {
    delete process.env.RULITH_BENCH_TRANSCRIPT
  }

  // Crash forensics (real P5): a turn-level budget death must not throw
  // away the board - partial derivations scored, dump + error attached.
  const crashed = await runBoardArm(new TimeoutAfterModelingModel(p), p, 6)
  if (crashed.ok) {
    throw new Error('selftest: crashed arm must not score ok')
  }
  if (!crashed.dnfError || !/timeout/i.test(crashed.dnfError)) {
    throw new Error(`selftest: crash must record dnfError (got ${JSON.stringify(crashed.dnfError)})`)
  }
  if (!crashed.boardDump?.includes('cost(')) {
    throw new Error('selftest: crash path must still dump the partial board')
  }
  if (crashed.lineHits !== crashed.lineCount) {
    throw new Error('selftest: partial derivations must still be scored after a crash')
  }

  // Tolerance: ONE timed-out turn is absorbed (nudge + continue), so a
  // model that recovers afterwards still solves the problem.
  const recovered = await runBoardArm(new RecoversAfterTimeoutModel(p), p, 8)
  if (!recovered.ok || recovered.dnfError !== undefined) {
    throw new Error(
      `selftest: one tolerated timeout should still allow a full solve (got ${JSON.stringify({
        ok: recovered.ok,
        dnfError: recovered.dnfError,
        totalDerived: recovered.totalDerived,
      })})`,
    )
  }

  // Bare-arm helper (shared by the model A and model B bare arms): a correct
  // ANSWER scores, a wrong one is rejected, a dead endpoint is a DNF row.
  const right = await runBaselineArm({ chat: async () => `sure.\nANSWER: ${p.total}` }, p)
  if (!right.ok || right.dnf) throw new Error('selftest: bare arm must score a correct ANSWER')
  const wrongAnswer = await runBaselineArm({ chat: async () => 'ANSWER: 1' }, p)
  if (wrongAnswer.ok) throw new Error('selftest: bare arm must reject a wrong ANSWER')
  const dead = await runBaselineArm(
    {
      chat: async () => {
        throw new Error('endpoint down')
      },
    },
    p,
  )
  if (dead.ok || !dead.dnf) throw new Error('selftest: a dead endpoint must be a DNF row, not a crash')

  console.log('bench-arith selftest PASSED (board arm scored from derived facts; parser + scorer + failure dump + bare arm sane)')
}

// ---------------------------------------------------------------- per-problem

export type ArithTally = { ok: number; dnf: number; assertedOnly: number; tok: UsageTally }
export const emptyArithTally = (): ArithTally => ({ ok: 0, dnf: 0, assertedOnly: 0, tok: emptyUsageTally() })

/**
 * Run ONE problem across the active arms and return its log row, mutating the
 * shared per-arm tally. The single source of truth for both the standalone
 * serial main() and the unified pool runner (bench-pool.ts).
 *
 * `makeClient(arm)` supplies the model: serial main passes a shared LlmClient;
 * the pool passes a fresh per-task LlmClient (so consumeUsage windows stay
 * isolated under concurrency); selftests pass scripted models.
 */
export async function runArithProblemRow(
  p: Problem,
  ctx: { active: readonly BenchArm[]; maxTurns: number; tally: Map<BenchArm, ArithTally>; makeClient: ClientFactory },
): Promise<Record<string, unknown>> {
  const { active, maxTurns, tally, makeClient } = ctx
  const row: Record<string, unknown> = { id: p.id, total: p.total }
  for (const arm of active) {
    const t = tally.get(arm)!
    const client = makeClient(arm)
    client.consumeUsage?.() // open this arm's token window
    console.error(`[arith p${p.id}] ${arm} arm started`)
    let armRow: Record<string, unknown>
    if (arm === 'baseline' || arm === 'baseline_b') {
      const outcome = await runBaselineArm(client, p)
      if (outcome.ok) t.ok += 1
      if (outcome.dnf) t.dnf += 1
      armRow = outcome.row
    } else {
      const t0 = Date.now()
      try {
        const score = await runBoardArm(client, p, maxTurns, arm)
        if (score.ok) t.ok += 1
        if (score.dnfError !== undefined) t.dnf += 1
        if (score.totalAssertedOnly) t.assertedOnly += 1
        armRow = { ...score, ms: Date.now() - t0 }
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
  // Resume support: skip the first K problems (same seed keeps ids aligned),
  // so a crashed or interrupted run can continue where it stopped.
  const skip = Number(process.env.RULITH_BENCH_SKIP ?? 0)
  // TAKE: window length after skip (for partitioned concurrency — worker k runs ids [skip+1 .. skip+take]).
  // Unset/0 ⇒ run through the end (serial / resume). Same seed across workers keeps ids 1..N aligned.
  const take = Number(process.env.RULITH_BENCH_TAKE ?? 0)
  const rng = mulberry32(seed)
  const problems = Array.from({ length: N }, (_, i) => generateProblem(rng, i + 1)).slice(skip, take > 0 ? skip + take : undefined)

  const llm = new LlmClient()
  const llmB = bConfig ? new LlmClient(bConfig) : undefined
  mkdirSync('logs', { recursive: true })
  const logPath = join('logs', `bench-arith-${Date.now()}.jsonl`)
  const log = (entry: unknown): void => appendFileSync(logPath, `${JSON.stringify(entry)}\n`)
  // Attribution header: model identity from env, not from any UI banner
  // (lesson of validation #27 - environment labels are assertions too).
  log({
    type: 'config',
    bench: 'arith',
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
    unitDigits: UNIT_DIGITS,
    qtyDigits: QTY_DIGITS,
    lines: `${MIN_LINES}-${MAX_LINES}`,
  })

  // One tally per arm; rows carry one column per arm so a multi-arm run
  // reads as a side-by-side table and single-arm runs merge by problem id.
  const ORDER: readonly BenchArm[] = ['baseline', 'baseline_b', 'board', 'board_b']
  const active = ORDER.filter((a) => arms.has(a))
  const makeClient: ClientFactory = (a) => (a.endsWith('_b') ? llmB! : llm)
  const tally = new Map<BenchArm, ArithTally>(active.map((a) => [a, emptyArithTally()]))
  const ran = problems.length

  for (const p of problems) {
    const row = await runArithProblemRow(p, { active, maxTurns, tally, makeClient })
    log(row)
    console.log(JSON.stringify(row))
  }

  console.log('---')
  const label = (a: BenchArm): string =>
    a.endsWith('_b') ? `${a} [${bConfig?.model}]` : `${a} [${process.env.RULITH_LLM_MODEL ?? 'local-model'}]`
  for (const arm of active) {
    const t = tally.get(arm)!
    if (arm === 'baseline' || arm === 'baseline_b') {
      console.log(`${label(arm)}: ${t.ok}/${ran} exact (${t.dnf} DNF/timeout)`)
    } else {
      console.log(`${label(arm)}: ${t.ok}/${ran} exact AND closure-derived (${t.dnf} DNF/timeout)`)
      console.log(
        `${label(arm)}: ${t.assertedOnly}/${ran} had the right number but only as a bare assertion (counted as FAIL)`,
      )
    }
    if (t.tok.calls > 0) console.log(`${label(arm)}: ${fmtUsage(t.tok)}`)
  }
  if (skip > 0) console.log(`(resumed at problem ${skip + 1}; merge with the earlier log for full totals)`)
  console.log(`log: ${logPath}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
