/**
 * bench-revision — A/B benchmark for the board's BELIEF-REVISION value.
 *
 * The one capability a chat model + tools almost never has: truth maintenance.
 * Retract a premise and the conclusions that depended on it withdraw, exactly
 * and automatically, along the evidence chain — while conclusions that still
 * have a surviving support REMAIN. Models (even frontier, even with "think")
 * anchor on the old conclusion or over-retract; the board's JTMS gets it right.
 *
 * Domain: access-control / eligibility (no arithmetic — this whole task has not
 * a single number, which is the point: "general deductive engine, not a
 * calculator"). Ground facts: has_role(user,R), grants(R,P), enables(P,A).
 * The model authors a 2-hop chain of rules:
 *     has_role(u,R) & grants(R,P)  -> has_perm(u,P)
 *     has_perm(u,P) & enables(P,A) -> can_do(u,A)
 * The closure derives can_do(user,A) facts with provenance to the ground facts.
 *
 * Each case queries can_do(user, A_i) conclusions in THREE classes:
 *   single   derivable ONLY through the to-be-retracted fact P -> must be WITHDRAWN
 *   indep    derivable through facts unrelated to P            -> must STAY
 *   redundant derivable through P AND an independent 2nd path  -> must STAY
 * The `redundant` class is the discriminator: a model typically drops it with P
 * (it touched P!), but the alternative support survives, so the truth is STAY.
 * The board removes the conclusion then re-derives it from the surviving path.
 *
 * Two phases: establish (assert facts + 2 rules, conclusions derived) then
 * revise (retract P, ask which conclusions still hold). Ground truth is the
 * exact set of queried can_do that remain derivable after P is gone, computed
 * deterministically from the generated graph — which IS the board's post-retract
 * state.
 *
 *   baseline  plain chat; model is given the facts/rules/queries, answers the
 *             establish set, is told the retraction, then answers
 *             "STILL: <ids> / GONE: <ids>" for the queried actions
 *   board     board-driven; the model asserts facts + the two rules, retracts P
 *             with retract_node (physical delete + evidence cascade, NOT
 *             consume/archive), and the surviving [derived] can_do facts are
 *             the answer
 *
 * Metric per arm: exact-set solve rate over the queried conclusions (survivors
 * kept AND withdrawn ones dropped), plus action-level wrong-survivor /
 * wrong-withdrawal counts (a "kept-something-that-should-have-died" is the
 * over-anchor failure; a "dropped-something-that-should-have-stayed" is the
 * over-retract failure — the redundant class lives here).
 *
 * Usage mirrors bench-audit:
 *   tsx src/examples/bench-revision.ts --selftest
 *   RULITH_BENCH_N=20 RULITH_BENCH_SEED=7
 *   RULITH_BENCH_ARM=both|all|<comma list>   arm selection, see bench-arms.ts
 *   RULITH_BENCH_ACTIONS=6-9   queried can_do conclusions per case (default)
 *   RULITH_BENCH_RETRACTS=1-2  ground facts retracted per case (default)
 *   RULITH_LLM_BASE_URL / RULITH_LLM_MODEL / RULITH_LLM_TIMEOUT_MS
 *   RULITH_LLM_MODEL_B       model B: enables the cross-model arms
 *   RULITH_BENCH_TRANSCRIPT=1  capture model replies for post-mortem
 */
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { getLogicContext } from '../engine/logic-context.js'
import { runAgentTask, type ChatModel } from '../agent/task-loop.js'
import { ToolRegistry } from '../agent/tools.js'
import { LlmClient, modelBConfigFromEnv, type ChatMessage } from '../agent/llm.js'
import type { WorkingMemoryOperation } from '../engine/working-memory.js'
import { addUsage, emptyUsageTally, fmtUsage, resolveArms, usageRow, type BenchArm, type UsageTally } from './bench-arms.js'
import type { ClientFactory } from './bench-pool.js'

// ---------------------------------------------------------------- problems

/** One ground access-control fact (the EDB the model asserts and partly retracts). */
type Fact =
  | { kind: 'has_role'; id: string; user: string; role: string }
  | { kind: 'grants'; id: string; role: string; perm: string }
  | { kind: 'enables'; id: string; perm: string; action: string }

/** A queried can_do(user, action) conclusion with its design intent. */
type Query = { action: string; cls: 'single' | 'indep' | 'redundant' }

export type Problem = {
  id: number
  user: string
  facts: Fact[]
  /** Ground-fact node ids to retract in the revise phase (the premises P). */
  retractIds: string[]
  queries: Query[]
  /** Exact set of queried actions whose can_do STILL holds after the retraction. */
  survivors: string[]
}

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

const ACTION_RANGE = /^(\d+)-(\d+)$/.exec(process.env.RULITH_BENCH_ACTIONS ?? '6-9')
const MIN_ACTIONS = Math.max(3, Number(ACTION_RANGE?.[1] ?? 6))
const MAX_ACTIONS = Math.max(MIN_ACTIONS, Number(ACTION_RANGE?.[2] ?? 9))
const RETRACT_RANGE = /^(\d+)-(\d+)$/.exec(process.env.RULITH_BENCH_RETRACTS ?? '1-2')
const MIN_RETRACTS = Math.max(1, Number(RETRACT_RANGE?.[1] ?? 1))
const MAX_RETRACTS = Math.max(MIN_RETRACTS, Number(RETRACT_RANGE?.[2] ?? 2))

/**
 * Build one case. Construction guarantees the three conclusion classes are all
 * realizable against ONE retracted role:
 *
 *   - role_p (the retract target) grants perm p_p.
 *   - role_q (kept) grants perm p_q.
 *   - single    action: enabled ONLY by p_p  -> dies with role_p.
 *   - indep     action: enabled ONLY by p_q  -> survives.
 *   - redundant action: enabled by BOTH p_p AND p_q -> survives via p_q.
 *
 * Ground truth is computed by reachability over the graph minus the retracted
 * fact(s) — the same answer the board reaches by retract + re-closure.
 */
export function generateProblem(rng: () => number, id: number): Problem {
  const user = `u${id}`
  const facts: Fact[] = []
  const queries: Query[] = []
  const fid = (s: string): string => `F_${id}_${s}`

  // Two role->perm chains: role_p (retracted) -> p_p, role_q (kept) -> p_q.
  facts.push({ kind: 'has_role', id: fid('role_p'), user, role: 'role_p' })
  facts.push({ kind: 'has_role', id: fid('role_q'), user, role: 'role_q' })
  facts.push({ kind: 'grants', id: fid('g_p'), role: 'role_p', perm: 'p_p' })
  facts.push({ kind: 'grants', id: fid('g_q'), role: 'role_q', perm: 'p_q' })

  // Distractor: a perm the user does NOT hold (no has_role/grants reaches it),
  // enabling some actions — a chain whose can_do never holds (over-claim trap).
  facts.push({ kind: 'grants', id: fid('g_d'), role: 'role_unheld', perm: 'p_d' })

  const total = MIN_ACTIONS + Math.floor(rng() * (MAX_ACTIONS - MIN_ACTIONS + 1))
  // Guarantee at least one of each class; fill the rest with a random class.
  const classes: Query['cls'][] = ['single', 'indep', 'redundant']
  for (let i = classes.length; i < total; i += 1) {
    classes.push(['single', 'indep', 'redundant'][Math.floor(rng() * 3)] as Query['cls'])
  }
  // Shuffle so the witnessed classes are not always first (Fisher–Yates).
  for (let i = classes.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1))
    ;[classes[i], classes[j]] = [classes[j]!, classes[i]!]
  }

  classes.forEach((cls, i) => {
    const action = `act_${i + 1}`
    if (cls === 'single') {
      facts.push({ kind: 'enables', id: fid(`e_${i + 1}_p`), perm: 'p_p', action })
    } else if (cls === 'indep') {
      facts.push({ kind: 'enables', id: fid(`e_${i + 1}_q`), perm: 'p_q', action })
    } else {
      facts.push({ kind: 'enables', id: fid(`e_${i + 1}_p`), perm: 'p_p', action })
      facts.push({ kind: 'enables', id: fid(`e_${i + 1}_q`), perm: 'p_q', action })
    }
    queries.push({ action, cls })
  })

  // A couple of distractor enables on the unheld perm — can_do never holds for
  // these and they are NOT queried (they only thicken the establish picture).
  facts.push({ kind: 'enables', id: fid('e_d1'), perm: 'p_d', action: 'act_unheld_1' })
  facts.push({ kind: 'enables', id: fid('e_d2'), perm: 'p_d', action: 'act_unheld_2' })

  // Retract target(s): always role_p; optionally also grants(role_p, p_p) to
  // exercise multi-retract. Both remove the same support (the p_p chain), so the
  // ground truth is identical — what changes is the model's bookkeeping load.
  const retractIds = [fid('role_p')]
  const nRetract = MIN_RETRACTS + Math.floor(rng() * (MAX_RETRACTS - MIN_RETRACTS + 1))
  if (nRetract >= 2) retractIds.push(fid('g_p'))

  const survivors = computeSurvivors(facts, retractIds, user, queries)
  return { id, user, facts, retractIds, queries, survivors }
}

/** Reachability ground truth: which queried can_do(user,action) hold after the
 *  retracted ground facts are removed. Mirrors the kernel's role->perm->action
 *  closure exactly (this is what retract + re-closure yields). */
function computeSurvivors(facts: Fact[], retractIds: string[], user: string, queries: Query[]): string[] {
  const dead = new Set(retractIds)
  const live = facts.filter((f) => !dead.has(f.id))
  const roles = new Set(live.filter((f): f is Extract<Fact, { kind: 'has_role' }> => f.kind === 'has_role' && f.user === user).map((f) => f.role))
  const perms = new Set<string>()
  for (const f of live) {
    if (f.kind === 'grants' && roles.has(f.role)) perms.add(f.perm)
  }
  const actions = new Set<string>()
  for (const f of live) {
    if (f.kind === 'enables' && perms.has(f.perm)) actions.add(f.action)
  }
  return queries.filter((q) => actions.has(q.action)).map((q) => q.action)
}

/** The two rules the board arm must author (and that ground truth mirrors). */
const RULE_OPS: WorkingMemoryOperation[] = [
  {
    op: 'add_axiom',
    id: 'ax_perm',
    label: 'a role the user holds that grants a permission gives the user that permission',
    when: [
      { predicate: 'has_role', args: { user: '?u', role: '?r' } },
      { predicate: 'grants', args: { role: '?r', perm: '?p' } },
    ],
    then: [{ predicate: 'has_perm', args: { user: '?u', perm: '?p' } }],
  },
  {
    op: 'add_axiom',
    id: 'ax_can',
    label: 'a permission the user holds that enables an action lets the user do that action',
    when: [
      { predicate: 'has_perm', args: { user: '?u', perm: '?p' } },
      { predicate: 'enables', args: { perm: '?p', action: '?a' } },
    ],
    then: [{ predicate: 'can_do', args: { user: '?u', action: '?a' } }],
  },
]

/** assert_fact ops for every ground fact in a problem. */
function factOps(p: Problem): WorkingMemoryOperation[] {
  return p.facts.map((f): WorkingMemoryOperation => {
    if (f.kind === 'has_role') {
      return { op: 'assert_fact', id: f.id, predicate: 'has_role', args: { user: f.user, role: f.role } }
    }
    if (f.kind === 'grants') {
      return { op: 'assert_fact', id: f.id, predicate: 'grants', args: { role: f.role, perm: f.perm } }
    }
    return { op: 'assert_fact', id: f.id, predicate: 'enables', args: { perm: f.perm, action: f.action } }
  })
}

// ---------------------------------------------------------------- baseline arm

function describeFacts(p: Problem): string {
  const roles = p.facts.filter((f) => f.kind === 'has_role').map((f) => (f as Extract<Fact, { kind: 'has_role' }>).role)
  const grants = p.facts.filter((f) => f.kind === 'grants').map((f) => {
    const g = f as Extract<Fact, { kind: 'grants' }>
    return `role ${g.role} grants permission ${g.perm}`
  })
  const enables = p.facts.filter((f) => f.kind === 'enables').map((f) => {
    const e = f as Extract<Fact, { kind: 'enables' }>
    return `permission ${e.perm} enables action ${e.action}`
  })
  return (
    `User ${p.user} holds roles: ${roles.join(', ')}.\n` +
    `Role grants: ${grants.join('; ')}.\n` +
    `Permission enables: ${enables.join('; ')}.\n` +
    `Rules: (1) if the user holds a role that grants a permission, the user has that permission; ` +
    `(2) if the user has a permission that enables an action, the user can do that action.`
  )
}

function retractDescription(p: Problem): string {
  const items = p.retractIds.map((id) => {
    const f = p.facts.find((x) => x.id === id)!
    if (f.kind === 'has_role') return `the user no longer holds role ${f.role} (has_role removed)`
    if (f.kind === 'grants') return `role ${f.role} no longer grants permission ${f.perm} (grants removed)`
    return `permission ${f.perm} no longer enables action ${f.action} (enables removed)`
  })
  return items.join('; ')
}

function baselineMessages(p: Problem): ChatMessage[] {
  const queried = p.queries.map((q) => q.action)
  return [
    {
      role: 'system',
      content:
        'You track which actions a user can perform under access-control rules, and you keep that ' +
        'judgment correct when a fact is REVOKED. A conclusion remains true if it still has ANY ' +
        'valid derivation; it becomes false only when EVERY derivation is gone. ' +
        'After the revocation, give your verdict on the LAST line exactly as: ' +
        'STILL: <comma-separated actions still doable> | GONE: <comma-separated actions no longer doable>. ' +
        'Use NONE for an empty side. Judge only the queried actions.',
    },
    {
      role: 'user',
      content:
        `${describeFacts(p)}\n\n` +
        `Queried actions: ${queried.join(', ')}.\n\n` +
        `REVOCATION: ${retractDescription(p)}.\n\n` +
        `After this revocation, for EACH queried action decide whether the user can STILL do it ` +
        `(it has a surviving derivation) or it is now GONE (no derivation remains). ` +
        `Answer with the STILL:/GONE: line.`,
    },
  ]
}

/** Parse "STILL: a,b | GONE: c" into the set of still-doable actions, restricted
 *  to the queried universe. undefined => no parseable verdict. */
export function parseBaselineAnswer(reply: string, queried: string[]): string[] | undefined {
  const line = [...reply.matchAll(/STILL:\s*([^\n|]*)(?:\|\s*GONE:\s*([^\n]*))?/gi)].pop()
  if (!line) return undefined
  const universe = new Set(queried)
  const pick = (s: string | undefined): string[] =>
    (s ?? '').match(/act_\w+/gi)?.map((x) => x.toLowerCase()).filter((x) => universe.has(x)) ?? []
  const still = new Set(pick(line[1]))
  // If GONE names a queried action, it is authoritatively not-still (covers a
  // model that lists the same action on both sides — GONE wins, deletion is the
  // claim under test).
  for (const g of pick(line[2])) still.delete(g)
  return [...still]
}

// ---------------------------------------------------------------- board arm

function boardGoal(p: Problem): string {
  const queried = p.queries.map((q) => q.action).join(', ')
  const retractList = p.retractIds.join(', ')
  return (
    `Model this access-control board, then revise it after a revocation.\n` +
    `STEP 1 — assert one fact per ground fact: has_role(user, role) for each role the user holds, ` +
    `grants(role, perm) for each grant, enables(perm, action) for each enable. ` +
    `Use EXACTLY these node ids so they can be revoked later: ${p.facts.map((f) => f.id).join(', ')}.\n` +
    `STEP 2 — add TWO rules: ` +
    `has_role(?u,?r) & grants(?r,?p) => has_perm(?u,?p); and ` +
    `has_perm(?u,?p) & enables(?p,?a) => can_do(?u,?a). ` +
    `The closure will DERIVE can_do(...) facts; do NOT assert can_do yourself.\n` +
    `STEP 3 — revoke the named fact(s) by retracting the node(s): ${retractList}. ` +
    `Use retract_node (NOT consume/archive): retraction physically removes the fact and withdraws ` +
    `every conclusion that depended on it, while conclusions with another surviving support remain.\n` +
    `STEP 4 — after the revocation, the board's remaining can_do(${p.user}, ...) facts are the answer. ` +
    `Then record_result noting which of these queried actions are still doable: ${queried}.\n` +
    `The facts:\n${describeFacts(p)}`
  )
}

type RevisionScore = {
  ok: boolean
  survived: string[]
  /** queried actions kept that should have been withdrawn (over-anchor). */
  wrongSurvivors: number
  /** queried actions dropped that should have stayed (over-retract; redundant class). */
  wrongWithdrawals: number
  /** can_do facts that were ASSERTED rather than derived (laundering attempt). */
  assertedCanDo: number
}

/** Score the board's post-retract state: the [derived] can_do(user,action) facts
 *  for queried actions vs the survivor ground truth. Asserted can_do is a fail. */
function scoreBoard(store: MemorySpaceStore, spaceId: string, p: Problem): RevisionScore {
  const facts = getLogicContext(store, spaceId).facts
  const canDo = facts.filter((f) => f.atom.predicate === 'can_do' && String(f.atom.args?.user) === p.user)
  const queried = new Set(p.queries.map((q) => q.action))
  const survived = [
    ...new Set(
      canDo
        .filter((f) => f.derived)
        .map((f) => String(f.atom.args?.action).toLowerCase())
        .filter((a) => queried.has(a)),
    ),
  ]
  const assertedCanDo = canDo.filter((f) => !f.derived && queried.has(String(f.atom.args?.action).toLowerCase())).length
  const truth = new Set(p.survivors)
  const wrongSurvivors = survived.filter((a) => !truth.has(a)).length
  const wrongWithdrawals = p.survivors.filter((a) => !survived.includes(a)).length
  return {
    ok: wrongSurvivors === 0 && wrongWithdrawals === 0 && assertedCanDo === 0,
    survived,
    wrongSurvivors,
    wrongWithdrawals,
    assertedCanDo,
  }
}

const WANT_TRANSCRIPT = (): boolean => process.env.RULITH_BENCH_TRANSCRIPT === '1'
const capReply = (reply: string, cap = 4000): string =>
  reply.length > cap ? `${reply.slice(0, cap)} ...[+${reply.length - cap} chars]` : reply

async function runBoardArm(
  llm: ChatModel,
  p: Problem,
  maxTurns: number,
): Promise<RevisionScore & { turns: number; transcript?: string[] }> {
  const store = new MemorySpaceStore()
  let spaceId = ''
  let turns = 0
  const transcript: string[] = []
  const tapped: ChatModel = WANT_TRANSCRIPT()
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
  return { ...score, turns, ...(WANT_TRANSCRIPT() ? { transcript } : {}) }
}

function scoreBaseline(answer: string[] | undefined, p: Problem): {
  ok: boolean
  wrongSurvivors: number
  wrongWithdrawals: number
} {
  if (answer === undefined) {
    // No verdict: count every survivor as missed (a non-answer is a failure).
    return { ok: false, wrongSurvivors: 0, wrongWithdrawals: p.survivors.length }
  }
  const truth = new Set(p.survivors)
  const wrongSurvivors = answer.filter((a) => !truth.has(a)).length
  const wrongWithdrawals = p.survivors.filter((a) => !answer.includes(a)).length
  return { ok: wrongSurvivors === 0 && wrongWithdrawals === 0, wrongSurvivors, wrongWithdrawals }
}

type BaselineOutcome = {
  ok: boolean
  wrongSurvivors: number
  wrongWithdrawals: number
  dnf: boolean
  row: Record<string, unknown>
}

/** One bare-chat revision — shared by the model A and model B bare arms. */
async function runBaselineArm(client: ChatModel, p: Problem): Promise<BaselineOutcome> {
  const t0 = Date.now()
  const queried = p.queries.map((q) => q.action)
  try {
    const reply = await client.chat(baselineMessages(p))
    const answer = parseBaselineAnswer(reply, queried)
    const s = scoreBaseline(answer, p)
    return { ...s, dnf: false, row: { ...s, answer, ms: Date.now() - t0 } }
  } catch (error) {
    return {
      ok: false,
      wrongSurvivors: 0,
      wrongWithdrawals: 0,
      dnf: true,
      row: { ok: false, dnf: true, error: String(error).slice(0, 120), ms: Date.now() - t0 },
    }
  }
}

// ---------------------------------------------------------------- selftest

/**
 * Scripted model that drives the BOARD arm correctly: asserts every ground
 * fact, adds the two chain rules, retracts the named premise(s), records.
 * This is the scripted PROOF that the board does belief revision right.
 */
class ScriptedRevisionModel implements ChatModel {
  private step = 0
  constructor(private readonly p: Problem) {}

  async chat(_messages: ChatMessage[]): Promise<string> {
    this.step += 1
    if (this.step === 1) {
      const ops: WorkingMemoryOperation[] = [...factOps(this.p), ...RULE_OPS]
      return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops }, note: 'establish' })
    }
    if (this.step === 2) {
      const ops: WorkingMemoryOperation[] = this.p.retractIds.map((nodeId) => ({
        op: 'retract_node',
        nodeId,
        reason: 'revoked',
      }))
      return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops }, note: 'revise' })
    }
    if (this.step === 3) {
      return JSON.stringify({
        tool: 'update_working_memory',
        args: {
          operations: [
            { op: 'record_result', id: 'res', label: 'revision complete', summary: 'surviving can_do are the answer' },
          ],
        },
        note: 'record',
      })
    }
    return JSON.stringify({ tool: 'done', args: { summary: 'revision complete' } })
  }
}

async function selftest(): Promise<void> {
  const rng = mulberry32(7)
  const pool = Array.from({ length: 12 }, (_, i) => generateProblem(rng, i + 1))

  // Pick a case that genuinely contains all three classes (the generator
  // guarantees it, but assert it so the proof rests on a real mixed case).
  const p = pool.find(
    (q) =>
      q.queries.some((x) => x.cls === 'single') &&
      q.queries.some((x) => x.cls === 'indep') &&
      q.queries.some((x) => x.cls === 'redundant'),
  )
  if (!p) throw new Error('selftest: generator did not produce a case with all three classes')

  const single = p.queries.filter((q) => q.cls === 'single').map((q) => q.action)
  const indep = p.queries.filter((q) => q.cls === 'indep').map((q) => q.action)
  const redundant = p.queries.filter((q) => q.cls === 'redundant').map((q) => q.action)

  // Ground-truth invariants of the generated case (independent of the board):
  //   single    -> WITHDRAWN (not a survivor)
  //   indep     -> STAYS
  //   redundant -> STAYS  (this is the JTMS discriminator)
  for (const a of single) {
    if (p.survivors.includes(a)) throw new Error(`selftest: single-support ${a} must NOT survive in ground truth`)
  }
  for (const a of [...indep, ...redundant]) {
    if (!p.survivors.includes(a)) throw new Error(`selftest: ${a} (indep/redundant) must survive in ground truth`)
  }

  // Drive the BOARD correctly and assert the board's post-retract state matches:
  const board = await runBoardArm(new ScriptedRevisionModel(p), p, 6)
  if (!board.ok) {
    throw new Error(`selftest: correct board driver did not score perfect: ${JSON.stringify(board)}`)
  }
  // The board WITHDREW every single-support conclusion...
  for (const a of single) {
    if (board.survived.includes(a)) {
      throw new Error(`selftest: board kept single-support ${a} after retraction — cascade failed`)
    }
  }
  // ...KEPT every independent conclusion...
  for (const a of indep) {
    if (!board.survived.includes(a)) {
      throw new Error(`selftest: board dropped independent ${a} — over-retraction`)
    }
  }
  // ...and KEPT every redundant-support conclusion (proves "withdrawn IFF ALL
  // supports gone", not naive "anything touching P dies").
  for (const a of redundant) {
    if (!board.survived.includes(a)) {
      throw new Error(`selftest: board dropped redundant-support ${a} — JTMS is over-retracting (cascade is not iff-all-supports-gone)`)
    }
  }
  if (board.survived.slice().sort().join(',') !== p.survivors.slice().sort().join(',')) {
    throw new Error(`selftest: board survivor set ${JSON.stringify(board.survived)} != ground truth ${JSON.stringify(p.survivors)}`)
  }

  // Exact-set scorer discriminates: the correct answer scores perfect; a
  // deliberately-wrong answer (drop a survivor + keep a withdrawn one) does NOT.
  const queried = p.queries.map((q) => q.action)
  const correct = scoreBaseline(p.survivors, p)
  if (!correct.ok) throw new Error('selftest: scorer rejected the correct survivor set')
  const wrong = scoreBaseline(
    // drop the first redundant survivor (over-retract) AND add a single-support
    // withdrawn action (over-anchor) — the two real failure modes.
    [...p.survivors.filter((a) => a !== redundant[0]), single[0]!],
    p,
  )
  if (wrong.ok || wrong.wrongWithdrawals !== 1 || wrong.wrongSurvivors !== 1) {
    throw new Error(`selftest: scorer failed to penalize a deliberately-wrong answer: ${JSON.stringify(wrong)}`)
  }

  // Baseline answer parser: STILL/GONE line, NONE side, restriction to queried.
  const parsed = parseBaselineAnswer(
    `reasoning...\nSTILL: ${indep[0]}, ${redundant[0]} | GONE: ${single[0]}`,
    queried,
  )
  if (parsed === undefined || !parsed.includes(indep[0]!) || !parsed.includes(redundant[0]!) || parsed.includes(single[0]!)) {
    throw new Error(`selftest: STILL/GONE parse failed: ${JSON.stringify(parsed)}`)
  }
  if (JSON.stringify(parseBaselineAnswer('STILL: NONE | GONE: act_1', queried)) !== '[]') {
    throw new Error('selftest: NONE parse failed')
  }
  if (parseBaselineAnswer('no verdict here', queried) !== undefined) {
    throw new Error('selftest: missing-verdict parse failed')
  }

  // Bare-arm helper: a scripted exact verdict scores; a dead endpoint is a DNF.
  const verdict = `STILL: ${p.survivors.join(', ') || 'NONE'} | GONE: ${p.queries
    .map((q) => q.action)
    .filter((a) => !p.survivors.includes(a))
    .join(', ') || 'NONE'}`
  const bare = await runBaselineArm({ chat: async () => verdict }, p)
  if (!bare.ok || bare.dnf) throw new Error(`selftest: bare arm failed on the correct verdict: ${JSON.stringify(bare.row)}`)
  const deadBare = await runBaselineArm({ chat: async () => { throw new Error('endpoint down') } }, p)
  if (deadBare.ok || !deadBare.dnf) throw new Error('selftest: bare arm must record a DNF row')

  console.log(
    `bench-revision selftest PASSED — JTMS belief revision proven on a mixed case ` +
      `(single→withdrawn ${JSON.stringify(single)}, indep→stay ${JSON.stringify(indep)}, ` +
      `redundant→stay ${JSON.stringify(redundant)}); exact-set scorer discriminates correct vs wrong; bare arm sane`,
  )
}

// ---------------------------------------------------------------- per-problem

export type RevisionTally = { ok: number; wrongSurv: number; wrongWdr: number; laundered: number; dnf: number; tok: UsageTally }
export const emptyRevisionTally = (): RevisionTally => ({ ok: 0, wrongSurv: 0, wrongWdr: 0, laundered: 0, dnf: 0, tok: emptyUsageTally() })

/**
 * Run ONE revision problem across the active arms; return its log row and
 * mutate the shared tally. Shared by the serial main() and the pool runner.
 */
export async function runRevisionProblemRow(
  p: Problem,
  ctx: { active: readonly BenchArm[]; maxTurns: number; tally: Map<BenchArm, RevisionTally>; makeClient: ClientFactory },
): Promise<Record<string, unknown>> {
  const { active, maxTurns, tally, makeClient } = ctx
  const row: Record<string, unknown> = {
    id: p.id,
    queries: p.queries.map((q) => `${q.action}:${q.cls}`),
    retract: p.retractIds,
    survivors: p.survivors,
  }
  for (const arm of active) {
    const t = tally.get(arm)!
    const client = makeClient(arm)
    client.consumeUsage?.()
    console.error(`[revision p${p.id}] ${arm} arm started`)
    let armRow: Record<string, unknown>
    if (arm === 'baseline' || arm === 'baseline_b') {
      const outcome = await runBaselineArm(client, p)
      if (outcome.ok) t.ok += 1
      if (outcome.dnf) t.dnf += 1
      t.wrongSurv += outcome.wrongSurvivors
      t.wrongWdr += outcome.wrongWithdrawals
      armRow = outcome.row
    } else {
      const t0 = Date.now()
      try {
        const s = await runBoardArm(client, p, maxTurns)
        if (s.ok) t.ok += 1
        t.wrongSurv += s.wrongSurvivors
        t.wrongWdr += s.wrongWithdrawals
        t.laundered += s.assertedCanDo
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
  const take = Number(process.env.RULITH_BENCH_TAKE ?? 0)
  const rng = mulberry32(seed)
  const problems = Array.from({ length: N }, (_, i) => generateProblem(rng, i + 1)).slice(
    skip,
    take > 0 ? skip + take : undefined,
  )

  const llm = new LlmClient()
  const llmB = bConfig ? new LlmClient(bConfig) : undefined
  mkdirSync('logs', { recursive: true })
  const logPath = join('logs', `bench-revision-${Date.now()}.jsonl`)
  const log = (entry: unknown): void => appendFileSync(logPath, `${JSON.stringify(entry)}\n`)
  log({
    type: 'config',
    bench: 'revision',
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
    actions: `${MIN_ACTIONS}-${MAX_ACTIONS}`,
    retracts: `${MIN_RETRACTS}-${MAX_RETRACTS}`,
  })

  const ORDER: readonly BenchArm[] = ['baseline', 'baseline_b', 'board', 'board_b']
  const active = ORDER.filter((a) => arms.has(a))
  const makeClient: ClientFactory = (a) => (a.endsWith('_b') ? llmB! : llm)
  const tally = new Map<BenchArm, RevisionTally>(active.map((a) => [a, emptyRevisionTally()]))
  const ran = problems.length

  for (const p of problems) {
    const row = await runRevisionProblemRow(p, { active, maxTurns, tally, makeClient })
    log(row)
    console.log(JSON.stringify(row))
  }

  console.log('---')
  const label = (a: BenchArm): string =>
    a.endsWith('_b') ? `${a} [${bConfig?.model}]` : `${a} [${process.env.RULITH_LLM_MODEL ?? 'local-model'}]`
  for (const arm of active) {
    const t = tally.get(arm)!
    console.log(
      `${label(arm)}: ${t.ok}/${ran} exact-set; over-anchor (kept-dead) ${t.wrongSurv}, ` +
        `over-retract (dropped-live) ${t.wrongWdr} (${t.dnf} DNF)`,
    )
    if (arm === 'board' || arm === 'board_b') {
      console.log(`${label(arm)}: ${t.laundered} can_do(...) facts asserted instead of derived (counted as FAIL)`)
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
