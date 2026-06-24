import {
  applyWorkingMemoryOperations,
  type ApplyOptions,
  type WorkingMemoryOperation,
} from '../engine/working-memory.js'
import { formatLogicContextAsText, getLogicContext, type LogicContext, type LogicContextFact } from '../engine/logic-context.js'
import { diffFacts, formatDelta } from '../engine/board-diff.js'
import { trustInversions, formatTrustInversions } from './evidence-policy.js'
import { simulateActionEffects } from '../engine/simulate.js'
import { validatePlan } from '../engine/validate-plan.js'
import { applyPlan } from '../engine/apply-plan.js'
import { suggestPlanRepairs } from '../engine/plan-repair.js'
import { planToGoal } from '../engine/plan-search.js'
import { solveConstraintsOnBoard } from '../engine/constraint-solve.js'
import type { FdValue, FdVariable } from '../engine/finite-domain.js'
import { explain, formatExplanation } from '../engine/explain.js'
import { classifyFailure, type FrictionProfile } from './failure-taxonomy.js'
import { frictionPreamble } from './friction-log.js'
import { deriveActionEffects } from '../engine/semantic-derivation.js'
import { formatAtom } from '../kernel/predicate.js'
import type { PredicateAtom } from '../model/types.js'
import type { SpaceStore } from '../storage/space-store.js'
import { parseToolCall, type ChatMessage } from './llm.js'
import { renderHistoryDelta } from './delta-transcript.js'
import type { ToolRegistry, ToolContext } from './tools.js'

/** Minimal LLM surface the loop needs (so tests can inject a scripted model). */
export interface ChatModel {
  chat(messages: ChatMessage[], options?: { signal?: AbortSignal }): Promise<string>
}

export interface RunAgentTaskOptions {
  store: SpaceStore
  llm: ChatModel
  reg: ToolRegistry
  rootDir: string
  goal: string
  maxTurns: number
  /** Optional rules/vocab seeded from the experience layer. */
  seedOperations?: WorkingMemoryOperation[]
  /** Operator/client inputs that should enter the board as trusted environment facts. */
  systemSeedOperations?: WorkingMemoryOperation[]
  /**
   * Windowed transcript: keep only the most recent N turn exchanges
   * (assistant reply + its result) instead of the full chat history.
   * The board is the single source of truth and every turn's result
   * already embeds the current board, so older snapshots are pure
   * redundancy — windowing turns an O(N^2) prompt into O(board size)
   * with no loss of the global view. 0/undefined keeps the validated
   * unbounded behaviour. Env default: RULITH_AGENT_WINDOW.
   */
  contextWindow?: number
  /** Receives the task's space id + process-fact recorder for follow-up (e.g. /remember). */
  onContext?: (info: { spaceId: string }, recordProcessFact: ProcessFactRecorder) => void
  /** Test hook: capture each turn's tool + result. */
  onTurn?: (turn: number, tool: string, result: string) => void
  /** UI hook: the board changed and should be re-read/rendered. */
  onBoard?: (spaceId: string, reason: string) => void
  /** Control arm (self-audit bench): hide the standing critique section from
   *  the model's view of the board. The board still COMPUTES it (end-state
   *  scoring is unaffected) - this only measures what the nudge is worth.
   *  Env default: RULITH_AGENT_NO_CRITIQUE=1. */
  suppressCritique?: boolean
  /** Which tools the system prompt advertises (a view over the registry, not a
   *  fork). Env default: RULITH_TOOL_PROFILE. Defaults to 'expert' (full surface). */
  toolProfile?: ToolProfile
  /** Machine-attested predicates the MODEL may not assert (test_result, edited,
   *  build_status, ...). The harness writes them; the model deriving conclusions
   *  from them is fine, claiming them by fiat is rejected. */
  attestedPredicates?: string[]
  /** Rules producing a guarded conclusion must read the machine evidence (see
   *  ApplyOptions.attestedDerivations) - e.g. a finding(kind=fixed) rule must
   *  read edited + test_result(status=pass). */
  attestedDerivations?: ApplyOptions['attestedDerivations']
  /** Per-model friction profile (P3 failure learning): a frictionPreamble is
   *  injected up front, and every guard failure this run hits is recorded into
   *  it (the caller persists it across runs via friction-log). */
  friction?: { profile: FrictionProfile; model: string }
  /** Model-facing board view arm. 'highlight' adds turn-over-turn board deltas;
   *  default/env plain preserves existing bench controls. */
  boardView?: BoardViewMode
  /** Optional task-class protocol injected into the SYSTEM message (e.g.
   *  proof-carrying mode). Keeps the user goal focused on the actual work while
   *  the loop contract remains stable across turns. */
  taskProtocol?: string
  /** Run this focused pass on an EXISTING board instead of creating a fresh one
   *  (staged orchestration: several passes share one board, state carried over). */
  spaceId?: string
  /** Optional operator/client cancellation. Checked between turns and passed
   *  through to LLM clients that can abort in-flight generation. */
  abortSignal?: AbortSignal
  /** ②③ hooks injected by the driven harness (`task-loop-driven.ts`). Bare task-loop
   *  is ①-clean: when undefined the routing-pack hint is omitted and board-derived
   *  control (driveState halt/done early-stop) is skipped. Both are no-ops for benches
   *  and plain tasks (routingHint is '' on a board with no pack predicates; driveState's
   *  early-stop is gated on a derived `root`), so leaving them undefined is behaviour-
   *  preserving for ① usage. The ②③ runtime injects the real implementations. */
  routingHint?: (facts: LogicContextFact[]) => string
  driveState?: (facts: LogicContextFact[]) => { state: string }
}

/** Records a harness-attested process fact onto the board (edited/build_status/game probes/etc.). */
export type ProcessFactRecorder = (kind: string, args: Record<string, string>) => void

/**
 * Tool profile = a VIEW over the single tool registry, not a fork. It only
 * controls which tools the system prompt advertises; the engine and every
 * dispatch path are unchanged, so a "core" agent that names a hidden tool
 * still has it handled. This is the progressive-disclosure surface (and the
 * control instrument for measuring whether the larger menu raises the bar):
 *   core     - the minimal loop: drive the board, read it, finish.
 *   planning - core + the plan tools (simulate/validate/apply/repair).
 *   audit    - core + the inspection tools (critique/explain).
 *   expert   - everything (default; the validated full surface).
 */
export type ToolProfile = 'core' | 'planning' | 'audit' | 'expert'
export type BoardViewMode = 'plain' | 'highlight' | 'delta-history'

type ToolMenuEntry = { profiles: ToolProfile[]; line: string }

const ALL: ToolProfile[] = ['core', 'planning', 'audit', 'expert']
const PLAN: ToolProfile[] = ['planning', 'expert']
const AUDIT: ToolProfile[] = ['audit', 'expert']

/**
 * Board ops a model may emit as a STANDALONE tool call. Goal/hint prose reads
 * like "call record_result", so models routinely send {"tool":"record_result",
 * args:{...}} directly instead of wrapping it in update_working_memory.operations.
 * The loop normalizes any such call to the op form so the board adjudicates it
 * (same derivation gate, same #32 atomicity) rather than letting it fall through
 * to reg.invoke → `error: unknown tool` and burn the run. record_result is the
 * common case (the finish move), but this MUST list EVERY op in the
 * WorkingMemoryOperation union — any missing one silently forks back to the
 * unknown-tool failure for that op alone (derive_aggregate was the gap, and it is
 * named verbatim in the aggregate gap hint, so models emit it standalone too).
 * Keep in sync with the union in engine/working-memory.ts. None of these names
 * collide with a dispatched tool or a registry (file/web/write) tool.
 */
const BOARD_OP_TOOLS = new Set<string>([
  'declare_goal',
  'assert_fact',
  'declare_hypothesis',
  'add_axiom',
  'derive_aggregate',
  'define_action',
  'record_result',
  'record_conflict',
  'retract_node',
  'revise_fact',
])

const TOOL_MENU: ToolMenuEntry[] = [
  { profiles: ALL, line: '- update_working_memory {operations:[...]}  // drive the board (incl. define_action)' },
  { profiles: PLAN, line: '- simulate_action {actionNodeId}            // preview; returns boardRevision token' },
  { profiles: PLAN, line: '- validate_plan {actionNodeIds:[...]}       // dry-run a whole ordered plan on a clone; reports first blocked step, goal reached, and the shortest prefix that already meets the goal (prune the rest)' },
  { profiles: PLAN, line: '- apply_action {actionNodeId, expectedRevision?}  // commit; pass simulate revision to reject a stale preview' },
  { profiles: PLAN, line: '- apply_plan {actionNodeIds:[...], requireGoals?}  // validate then commit a WHOLE plan in order, each step drift-guarded; nothing commits if it fails to validate' },
  { profiles: PLAN, line: '- suggest_plan_repairs {actionNodeIds:[...]}  // when a plan fails to validate, propose copyable repaired sequences (inserts the producer actions for the unmet precondition, multi-hop); re-validate before trusting' },
  { profiles: PLAN, line: '- plan_to_goal {maxDepth?,maxBeam?}  // let the board SEARCH its actions for a validated plan reaching the declared goals; returns a copyable plan to apply_plan' },
  { profiles: PLAN, line: '- solve {variables:[{name,domain:[...]}],conflictPredicate?}  // finite-domain constraint SOLVER: first add_axiom a rule deriving conflict(...) over assignment(var,value); the board SEARCHES an assignment with NO conflict, the closure certifies it and commits assignment(var,value) facts (else reports unsat/budget). Do not assign values yourself.' },
  { profiles: ALL, line: '- get_logic_context {}                       // re-read the board' },
  { profiles: AUDIT, line: '- board_critique {}                          // standing problems (self-sealed/unreachable/conflicting goals, asserted findings, vacuous rules, contradictions, dead actions)' },
  { profiles: AUDIT, line: '- explain_fact {factNodeId}                  // why a derived fact is true: rule chain back to asserted leaves' },
  { profiles: ALL, line: '- done {summary}                             // finish' },
]

export function resolveToolProfile(value: string | undefined): ToolProfile {
  return value === 'core' || value === 'planning' || value === 'audit' || value === 'expert' ? value : 'expert'
}

export function resolveBoardView(value: string | undefined): BoardViewMode {
  const v = String(value || '').trim().toLowerCase()
  if (v === 'highlight') return 'highlight'
  if (v === 'delta-history' || v === 'delta') return 'delta-history'
  return 'plain'
}

/** The tool lines a given profile advertises - a projection of the single
 *  registry, exported so the menu filter can be asserted directly. */
export function toolMenuFor(profile: ToolProfile): string[] {
  return TOOL_MENU.filter((entry) => entry.profiles.includes(profile)).map((entry) => entry.line)
}

function existingBoardContinuationProtocol(): string[] {
  return [
    'Existing-board continuation override:',
    '- This run is continuing an existing Rulith board; the current board is the source of truth.',
    '- First call get_logic_context and read the visible facts, goals, gaps, results, and critique before changing anything.',
    '- Do not start a parallel proof contract by re-declaring existing goals, root goal_node, acceptance_slot, or record_result ids.',
    '- Reuse visible node ids; add only the missing acceptance_slot, evidence_metric, slot_met, rules, or result refs needed to close the exposed gap.',
    '- If the gap is receipt freshness or result coverage, refresh record_result evidenceRefs against the current board instead of changing the acceptance bar.',
  ]
}

function taskSystemPrompt(reg: ToolRegistry, profile: ToolProfile = 'expert', taskProtocol = '', continuation = false): string {
  const protocol = taskProtocol.trim()
  return [
    'You are rulith-agent running a board-driven task. Reply with EXACTLY one JSON object per turn:',
    '  {"tool": "<name>", "args": {...}, "note": "<one short why>"}',
    '',
    'You have a reasoning board (working memory + rule engine). Drive it with update_working_memory.',
    'Record observations as facts (cite evidence), derive conclusions via rules, finish with done.',
    'Board ops in update_working_memory.operations: declare_goal{id,label,desired:[atom]},',
    '  assert_fact{id,predicate,args,evidenceRefs?}, add_axiom{id,label,when:[atom],then:[atom]},',
    '  derive_aggregate{id,kind?:"sum"|"count"|"min"|"max"|"avg",source:{predicate,valueArg?},into:{predicate,valueArg},where?,group_by?}',
    '  declare_hypothesis{id,predicate,args}, record_result{id,label,summary,evidenceRefs}, retract_node{nodeId}.',
    'record_result.evidenceRefs must be exact node ids visible in get_logic_context (facts, findings, axioms, goals, actions, or conflicts). Do not cite prose labels or invented ids.',
    'Atoms: {"predicate":"p","args":{"k":"v"}}; variables are "?x" (rules only).',
    '',
    'Built-ins in rule BODIES (not heads): compare eq/neq/lt/lte/gt/gte{left,right}, between{value,low,high} (closed interval), contains{left,right} (string substring);',
    'EXACT arithmetic add/sub/mul/div/mod/idiv/imod/pow/min/max{left,right,result} and neg/abs/sqrt/ln/exp{left,result}',
    '(idiv=floor division, imod=mathematical modulo non-negative for positive divisors; mod=JS sign-of-dividend remainder; sqrt/ln/exp=transcendental, IEEE best-effort, FAIL on domain/overflow e.g. sqrt(-1)/ln(0))',
    'string producer concat{left,right,result}: join two string/integer parts (integers stringified canonically, 5->"5"); chain for 3+ parts: concat(a,b,?ab) then concat(?ab,c,?abc)',
    '— do not multiply in your head, let the board compute it. The result becomes a derived fact',
    'with an evidence chain (retract an input and it disappears). Copyable template:',
    '  {"op":"add_axiom","id":"ax_cost","label":"cost = unit*qty",',
    '   "when":[{"predicate":"line","args":{"item":"?i","unit":"?u","qty":"?q"}},',
    '           {"predicate":"mul","args":{"left":"?u","right":"?q","result":"?t"}}],',
    '   "then":[{"predicate":"cost","args":{"item":"?i","total":"?t"}}]}',
    'To TOTAL many facts do NOT hand-write the chain and NEVER use a rule whose head feeds its own',
    'body (monotonic closures cannot run accumulator loops). One op expands to an exact chain rule:',
    '  {"op":"derive_aggregate","id":"agg_total","source":{"predicate":"cost","valueArg":"total"},',
    '   "into":{"predicate":"grand_total","valueArg":"value"}}   // kind:"count" needs no valueArg;',
    'it aggregates the facts present NOW - re-run it (same id) after facts change.',
    '  add a where filter to aggregate a subset: ...,"where":{"arg":"region","equals":"east"}',
    '  group_by buckets by a distinct arg, one result per group: ...,"group_by":"region"',
    '',
    'For CONSUME/PRODUCE transformations (a reactant is used up, a product appears) use an ACTION,',
    'not a rule (rules are monotonic - they never delete). define_action then apply_action; a',
    'negated effect DELETES the matching fact (consumption), a positive effect asserts it',
    '(production). Preconditions bind variables (incl. arithmetic) usable in effects. Template:',
    '  {"op":"define_action","id":"burn","action":"combust",',
    '   "preconditions":[{"predicate":"have","args":{"species":"H2"}},{"predicate":"have","args":{"species":"O2"}}],',
    '   "effects":[{"predicate":"have","args":{"species":"H2"},"negated":true},',
    '              {"predicate":"have","args":{"species":"H2O"}}]}',
    'Then call apply_action{actionNodeId:"burn"} (or simulate_action first to preview).',
    'Open goals list "producible via action X" when a defined action could produce the missing',
    'atom - that is the cue to simulate/apply that action (NOT to assert the product yourself).',
    '',
    'For COUNTED amounts (stoichiometry, budgets, inventory) do NOT consume the whole fact -',
    'bind the current amount, guard it with gte, COMPUTE the new amount with sub/add in the',
    'preconditions, then swap old amount for new in the effects. Template (consume 2 H2, produce 2 H2O):',
    '  {"op":"define_action","id":"burn1","action":"combust_once",',
    '   "preconditions":[{"predicate":"amount","args":{"species":"H2","mol":"?h"}},',
    '                    {"predicate":"gte","args":{"left":"?h","right":2}},',
    '                    {"predicate":"sub","args":{"left":"?h","right":2,"result":"?h2"}},',
    '                    {"predicate":"amount","args":{"species":"H2O","mol":"?w"}},',
    '                    {"predicate":"add","args":{"left":"?w","right":2,"result":"?w2"}}],',
    '   "effects":[{"predicate":"amount","args":{"species":"H2","mol":"?h"},"negated":true},',
    '              {"predicate":"amount","args":{"species":"H2","mol":"?h2"}},',
    '              {"predicate":"amount","args":{"species":"H2O","mol":"?w"},"negated":true},',
    '              {"predicate":"amount","args":{"species":"H2O","mol":"?w2"}}]}',
    '(extend the same pattern per species; gte guards make the action refuse when amounts run short)',
    '',
    'To SOLVE a constraint problem (choose a value for each variable so no constraint is',
    'violated) let the board SEARCH instead of guessing. First add_axiom a rule that derives',
    'conflict(...) from the bridge facts assignment(var,value); then call solve. The board tries',
    'assignments, the CLOSURE rejects any that derive conflict, and the certified solution is',
    'committed as assignment(var,value) facts. Template (schedule meetings into slots, no two',
    'in one slot):',
    '  {"op":"add_axiom","id":"ax_clash","label":"two in one slot clash",',
    '   "when":[{"predicate":"assignment","args":{"var":"?a","value":"?s"}},',
    '           {"predicate":"assignment","args":{"var":"?b","value":"?s"}},',
    '           {"predicate":"neq","args":{"left":"?a","right":"?b"}}],',
    '   "then":[{"predicate":"conflict","args":{"a":"?a","b":"?b","slot":"?s"}}]}',
    'then: {"tool":"solve","args":{"variables":[{"name":"M1","domain":["s1","s2"]},',
    '         {"name":"M2","domain":["s1","s2"]}],"conflictPredicate":"conflict"}}',
    '(do NOT assign the values yourself - the board searches and the closure certifies them)',
    '',
    ...(protocol ? [
      'Task protocol for this run:',
      protocol,
      '',
    ] : []),
    ...(continuation ? [
      ...existingBoardContinuationProtocol(),
      '',
    ] : []),
    '',
    'Tools:',
    reg.promptSection('task'),
    ...toolMenuFor(profile),
  ].join('\n')
}

/**
 * Board-driven task loop for the agent (v0.1). Same shape as the validated
 * fixture loop - turn budget, board persistence, stuck-action breaker - but
 * tool dispatch goes through the registry, and update_working_memory is
 * handled inline so the kernel invariants (derivation gate, attestation
 * warnings, idempotence) apply unchanged.
 */
export async function runAgentTask(options: RunAgentTaskOptions): Promise<string> {
  const { store, llm, reg, rootDir, goal, maxTurns } = options
  // Reuse an existing board when spaceId is given (staged orchestration runs
  // several focused passes on ONE shared board); otherwise create a fresh space.
  const space =
    options.spaceId !== undefined
      ? store.getSpace(options.spaceId)
      : store.createSpace({ id: `space:agent-${Date.now()}`, title: goal.slice(0, 80) })
  const ctx: ToolContext = { rootDir, mode: 'task', evidenceLog: new Map(), metrics: {} }

  // Machine-attested process facts (edited/build_status): write tools call
  // this so fixed(...) is DERIVED from facts the harness vouches for, not
  // claimed by the model. Mirrors the validated repair-mode design.
  let buildStatusOnBoard = false
  let editSeq = 0
  let testSeq = 0
  const factSeq = new Map<string, number>()
  const notifyBoard = (reason: string): void => options.onBoard?.(space.id, reason)
  const nextHarnessFactId = (kind: string): string => {
    const safe = kind.replace(/[^a-zA-Z0-9_]/g, '_') || 'fact'
    const next = (factSeq.get(safe) ?? 0) + 1
    factSeq.set(safe, next)
    return `me_${safe}_${next}`
  }
  // Process facts come from the TOOLS themselves — edit_file applied a change, the runner emitted TAP —
  // so they enter through the trusted channel: createdBy:'tool' makes grounding/floor/premise-provenance
  // see them as attested, matching their "Harness-attested" summaries. (source:'system' alone only
  // relaxes the batch attestation gate; it does NOT set createdBy, which would default to 'agent' and
  // floor a real machine fact to a bare model assert — the tier-trap that under-trusted real verdicts.)
  const recordProcessFact: ProcessFactRecorder = (kind, args) => {
    try {
      if (kind === 'edited') {
        editSeq += 1
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: `me_edit_${editSeq}`, predicate: 'edited', args, summary: 'Harness-attested: edit applied' },
        ], { source: 'system', createdBy: 'tool' })
      } else if (kind === 'test_result') {
        testSeq += 1
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: `me_test_${testSeq}`, predicate: 'test_result', args, summary: 'Harness-attested: test runner result' },
        ], { source: 'system', createdBy: 'tool' })
      } else if (kind === 'build_status') {
        applyWorkingMemoryOperations(store, space.id, [
          buildStatusOnBoard
            ? { op: 'revise_fact', nodeId: 'build_status', id: 'build_status', predicate: 'build_status', args, summary: 'Harness-attested build status' }
            : { op: 'assert_fact', id: 'build_status', predicate: 'build_status', args, summary: 'Harness-attested build status' },
        ], { source: 'system', createdBy: 'tool' })
        buildStatusOnBoard = true
      } else {
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: nextHarnessFactId(kind), predicate: kind, args, summary: `Harness-attested: ${kind}` },
        ], { source: 'system', createdBy: 'tool' })
      }
      notifyBoard(`system fact: ${kind}`)
    } catch {
      // bookkeeping must never block a tool result
    }
  }
  options.onContext?.({ spaceId: space.id }, recordProcessFact)
  if (options.systemSeedOperations && options.systemSeedOperations.length > 0) {
    // Operator/client/environment inputs enter as TRUSTED facts (the documented contract):
    // createdBy:'system' stamps them attested, so e.g. a host-declared required_floor policy or a
    // machine baseline is honored by grounding/evidence-policy — not floored to a bare model assertion.
    // (source:'system' alone only relaxes the batch gate; it does NOT set the node's createdBy.)
    applyWorkingMemoryOperations(store, space.id, options.systemSeedOperations, {
      format: 'text',
      source: 'system',
      createdBy: 'system',
    })
    notifyBoard('system seed')
  }
  if (options.seedOperations && options.seedOperations.length > 0) {
    applyBoardOps(store, space.id, options.seedOperations)
    notifyBoard('experience seed')
  }

  // The first two messages (system + goal) are PINNED; everything after is
  // the running transcript that windowing may trim.
  // Friction preamble (P3): if a per-model friction profile is supplied, remind
  // the model up front of the guards it keeps tripping (board-derived, not a new
  // judgment). Empty when the model has no recorded friction.
  const systemPrompt = (() => {
    const base = taskSystemPrompt(
      reg,
      options.toolProfile ?? resolveToolProfile(process.env.RULITH_TOOL_PROFILE),
      options.taskProtocol,
      options.spaceId !== undefined,
    )
    const preamble = options.friction ? frictionPreamble(options.friction.profile, options.friction.model) : ''
    return preamble ? `${preamble}\n\n${base}` : base
  })()
  const pinned: ChatMessage[] = [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content: options.spaceId !== undefined
        ? `Task: ${goal}\nYou are continuing an existing Rulith board. Start by calling get_logic_context, then close only the missing evidence or derivation gaps. Reply with one JSON tool call.`
        : `Task: ${goal}\nStart by declaring a goal and a plan. Reply with one JSON tool call.`,
    },
  ]
  const transcript: ChatMessage[] = []
  const window = options.contextWindow ?? Number(process.env.RULITH_AGENT_WINDOW ?? 0)
  const hideCritique = options.suppressCritique ?? process.env.RULITH_AGENT_NO_CRITIQUE === '1'
  const boardView = options.boardView ?? resolveBoardView(process.env.RULITH_BENCH_VIEW ?? process.env.RULITH_BOARD_VIEW)
  // delta-history (docs/product.md §3, append-only): history board-view turns append their delta
  // (never rewritten); the full CURRENT board is appended fresh at the TAIL of each send (not in history)
  // -> byte-stable prefix (prefix-cache hits), O(N), simpler. lastBoardSnap = previous board-view snapshot.
  const deltaHistory = boardView === 'delta-history'
  let lastBoardSnap: LogicContextFact[] | undefined
  const view = (text: string): string => (hideCritique ? stripCritiqueSection(text) : text)
  let previousBoardNodes: LogicContextFact[] | undefined
  const factSnapshot = (ctx: LogicContext): LogicContextFact[] =>
    [...ctx.facts, ...ctx.findings].map((f) => ({
      ...f,
      atom: structuredClone(f.atom),
      evidenceRefs: [...f.evidenceRefs],
    }))
  const boardText = (ctx: LogicContext, opts: { routing?: boolean } = {}): string => {
    const delta =
      boardView === 'highlight'
        ? formatDelta(diffFacts(previousBoardNodes, [...ctx.facts, ...ctx.findings]), {
            firstTurn: previousBoardNodes === undefined,
          })
        : ''
    previousBoardNodes = factSnapshot(ctx)
    const body = formatLogicContextAsText(ctx) + (opts.routing === false || !options.routingHint ? '' : options.routingHint(ctx.facts))
    // Evidence-policy critique (first wiring of the tested machinery into the live loop): surface
    // any satisfied obligation grounded weaker than a TRUSTED required_floor / human-tier constraint.
    // Inert (empty) without policy facts, so boards/benches that declare none are byte-for-byte unchanged.
    // Hidden under the suppressCritique control arm, like the standing critique it sits beside.
    const inversions = hideCritique
      ? ''
      : formatTrustInversions(trustInversions(ctx.facts, { attestedPredicates: options.attestedPredicates }))
    return view([delta, body, inversions].filter(Boolean).join('\n'))
  }
  // Each turn contributes 2 messages (assistant + result); keep 2*window.
  const trim = (): void => {
    if (deltaHistory) return // append-only delta-history keeps the full trajectory; never trims
    if (window > 0 && transcript.length > 2 * window) {
      transcript.splice(0, transcript.length - 2 * window)
    }
  }

  let finished = false
  let lastFailing = ''
  let failRepeat = 0
  const abortSummary = (): string => `aborted: stopped by operator. ${boardSummary(store, space.id)}`

  for (let turn = 0; turn < maxTurns && !finished; turn += 1) {
    if (options.abortSignal?.aborted) return abortSummary()
    // Board-derived control (driveState): read the loop's control decisions off the board.
    //  • a TRUSTED halt_requested (createdBy tool/system) stops immediately — a model-asserted halt is
    //    untrusted ⇒ ignored, so plain runs are unchanged.
    //  • a self-driven GOAL-TREE task (a derived root on the board) that has driven to 'done' (receipt
    //    closed) stops without waiting for the model to call done or burning turns. Gated on a derived
    //    root so empty / non-tree boards (every bench, every plain task) are untouched — an empty board
    //    reads 'done' vacuously and must NOT stop.
    //  • 'parked'/'stuck' are deliberately NOT terminal here: driveState's agentFrontier does not model
    //    "mark a leaf done / add constraint_met", so they can read prematurely while the model still has
    //    a legitimate move. Extending that frontier + measuring convergence is the real-machine round
    //    (docs/theory.md).
    if (options.driveState) {
      const driveFacts = getLogicContext(store, space.id).facts
      const drive = options.driveState(driveFacts).state
      if (drive === 'halted') {
        return `halted: stopped by a trusted halt_requested. ${boardSummary(store, space.id)}`
      }
      if (drive === 'done' && driveFacts.some((f) => f.derived && f.atom.predicate === 'root')) {
        return `done: ${boardSummary(store, space.id)}`
      }
    }
    let reply: string
    // delta-history (append-only): history is already deltas; the authoritative CURRENT board is appended
    // FRESH at the TAIL (the only per-turn-varying piece) so the prefix stays byte-stable for the cache.
    const messages = deltaHistory
      ? [...pinned, ...transcript, { role: 'user' as const, content: `[CURRENT BOARD \u2014 full]\n${boardText(getLogicContext(store, space.id))}` }]
      : [...pinned, ...transcript]
    try {
      reply = await llm.chat(messages, { signal: options.abortSignal })
    } catch (error) {
      if (options.abortSignal?.aborted) return abortSummary()
      throw error
    }
    if (options.abortSignal?.aborted) return abortSummary()
    transcript.push({ role: 'assistant', content: reply })
    const parsed = parseToolCall(reply)
    if (!parsed) {
      transcript.push({
        role: 'user',
        content:
          'No valid JSON tool call parsed. Output EXACTLY ONE balanced JSON object ' +
          '{"tool":...,"args":...} and check your brackets — a stray or missing "]" / "}" is the ' +
          'usual cause. If the reply was long it may have been truncated; send a smaller batch.',
      })
      trim()
      continue
    }
    // Tolerate `operations` emitted at the TOP LEVEL (a sibling of "tool") instead of nested under
    // "args" — a weak-model shape slip. Without this the dispatch reads call.args.operations → [],
    // applies an EMPTY batch silently, and the driver loops to the turn limit with no signal that its
    // ops were ignored (audit p3: turns:10, flagged:[]). Hoist top-level operations into args.
    const topLevelOps = (parsed as { operations?: unknown }).operations
    const shaped =
      parsed.tool === 'update_working_memory' && !Array.isArray(parsed.args?.operations) && Array.isArray(topLevelOps)
        ? { ...parsed, args: { ...(parsed.args ?? {}), operations: topLevelOps } }
        : parsed
    // Tolerate a board op emitted as a STANDALONE tool, e.g. {"tool":"record_result", args:{...}}.
    // Goal/hint prose like "call record_result" reads as a tool name, so models send the op directly
    // instead of wrapping it in update_working_memory.operations. Route it to the op form so the board
    // adjudicates it (same gate, same atomicity) rather than falling through to reg.invoke → unknown tool.
    const call = BOARD_OP_TOOLS.has(shaped.tool)
      ? { tool: 'update_working_memory', args: { operations: [{ ...(shaped.args ?? {}), op: shaped.tool }] }, note: shaped.note }
      : shaped

    let result: string
    if (call.tool === 'done') {
      const doneError = validateDone(store, space.id, { enforceAudit: !hideCritique, attestedPredicates: options.attestedPredicates })
      if (doneError) {
        result = doneError
      } else {
        finished = true
        result = `done: ${String(call.args?.summary ?? '')}`
      }
    } else if (call.tool === 'update_working_memory') {
      const ops = (call.args?.operations ?? []) as WorkingMemoryOperation[]
      if (ops.length === 0) {
        // Fail visibly instead of applying an empty batch and looping: name the expected shape so the
        // driver fixes it. The "error:" prefix also arms the stuck-call breaker if it keeps repeating.
        result =
          'error: update_working_memory received no operations. Nest them under args.operations — ' +
          '{"tool":"update_working_memory","args":{"operations":[ ... ]}}.'
      } else {
        result = applyBoardOps(
          store,
          space.id,
          ops,
          options.attestedPredicates,
          options.attestedDerivations,
          options.friction ? (kind) => options.friction!.profile.record(options.friction!.model, kind) : undefined,
          boardView === 'highlight' ? () => boardText(getLogicContext(store, space.id), { routing: false }) : undefined,
        )
        result = view(result)
      }
    } else if (call.tool === 'get_logic_context') {
      const ctx = getLogicContext(store, space.id)
      result = boardText(ctx)
    } else if (call.tool === 'board_critique') {
      const items = hideCritique ? [] : getLogicContext(store, space.id).critique
      result =
        items.length === 0
          ? 'critique: none - the board is healthy.'
          : ['critique:', ...items.map((item) => `- [${item.kind}] ${item.nodeId}: ${item.message}`)].join('\n')
    } else if (call.tool === 'simulate_action') {
      result = runSimulateAction(store, space.id, String(call.args?.actionNodeId ?? ''))
    } else if (call.tool === 'apply_action') {
      result = runApplyAction(store, space.id, String(call.args?.actionNodeId ?? ''))
    } else if (call.tool === 'validate_plan') {
      result = runValidatePlan(store, space.id, (call.args?.actionNodeIds ?? []) as string[])
    } else if (call.tool === 'apply_plan') {
      result = runApplyPlan(
        store,
        space.id,
        (call.args?.actionNodeIds ?? []) as string[],
        call.args?.requireGoals === true,
      )
    } else if (call.tool === 'suggest_plan_repairs') {
      result = runSuggestPlanRepairs(store, space.id, (call.args?.actionNodeIds ?? []) as string[])
    } else if (call.tool === 'plan_to_goal') {
      const r = planToGoal(store, space.id, {
        maxDepth: typeof call.args?.maxDepth === 'number' ? call.args.maxDepth : undefined,
        maxBeam: typeof call.args?.maxBeam === 'number' ? call.args.maxBeam : undefined,
      })
      result = r.found
        ? `plan_to_goal: found a validated plan: {"actionNodeIds":${JSON.stringify(r.plan)}} - apply_plan it.`
        : `plan_to_goal: no plan${r.note ? ` - ${r.note}` : ''}`
    } else if (call.tool === 'explain_fact') {
      try {
        result = formatExplanation(explain(store, space.id, String(call.args?.factNodeId ?? '')))
      } catch (error) {
        result = `error: ${error instanceof Error ? error.message : String(error)}`
      }
    } else if (call.tool === 'solve') {
      result = runSolve(store, space.id, call.args ?? {})
    } else {
      result = await reg.invoke(call.tool, call.args ?? {}, ctx)
    }
    if (options.abortSignal?.aborted) return abortSummary()

    options.onTurn?.(turn, call.tool, result)
    notifyBoard(call.tool)

    // Stuck-action breaker (fixture lesson): identical failing call repeated.
    const key = JSON.stringify(call)
    if (result.startsWith('error:')) {
      failRepeat = key === lastFailing ? failRepeat + 1 : 1
      lastFailing = key
      if (failRepeat >= 6) {
        return `aborted: model wedged on a failing "${call.tool}" call. ${boardSummary(store, space.id)}`
      }
    } else {
      failRepeat = 0
      lastFailing = ''
    }

    // delta-history (append-only): a board-view turn appends its DELTA from the previous snapshot; a
    // FAILED/no-op turn (empty delta) keeps the full result so its teaching error survives (arith fix).
    if (deltaHistory && (call.tool === 'update_working_memory' || call.tool === 'get_logic_context')) {
      const curr = factSnapshot(getLogicContext(store, space.id))
      const d = diffFacts(lastBoardSnap, curr)
      const changed = d.added.length > 0 || d.changed.length > 0 || d.removedIds.length > 0
      transcript.push({
        role: 'user',
        content: changed
          ? `[turn ${turn + 1} \u0394]\n${renderHistoryDelta(lastBoardSnap, curr)}`
          : `[turn ${turn + 1}/${maxTurns}]\n${result.slice(0, 6000)}`,
      })
      lastBoardSnap = curr
    } else {
      transcript.push({ role: 'user', content: `[turn ${turn + 1}/${maxTurns}]\n${result.slice(0, 6000)}` })
    }
    trim()
  }

  return finished
    ? boardSummary(store, space.id)
    : `max turns reached. ${boardSummary(store, space.id)}`
}

function describePreconditionFailure(r: {
  failedPrecondition?: PredicateAtom
  unsatisfiedPreconditions: PredicateAtom[]
}): string {
  const lines: string[] = []
  if (r.failedPrecondition) {
    lines.push(`first failing precondition: ${formatAtom(r.failedPrecondition)}`)
    if (r.failedPrecondition.negated === true) {
      const positive = formatAtom({ ...r.failedPrecondition, negated: undefined })
      lines.push(
        `hint: "negated":true is STRONG negation - it only matches an explicit not-${positive} fact. ` +
          `To require the ABSENCE of ${positive}, use "naf":true in the precondition instead.`,
      )
    }
  }
  lines.push(
    `missing facts: ${r.unsatisfiedPreconditions.map(formatAtom).join(', ') || 'none (a guard/arithmetic literal failed, see above)'}`,
  )
  return lines.join('\n')
}

function describeBinding(binding: Record<string, unknown>, candidates: number): string {
  const text = Object.entries(binding)
    .map(([name, value]) => `?${name}=${String(value)}`)
    .join(', ')
  if (!text) return ''
  const ambiguity =
    candidates > 1
      ? ` (WARNING: ${candidates} candidate bindings matched - the first is used; add preconditions to pin the instance you mean)`
      : ''
  return `binding: ${text}${ambiguity}`
}

/** solve tool: run the finite-domain constraint solver with the closure as the
 *  adjudicator (慢轨 #2). The model declares the constraint rule (deriving the
 *  conflict predicate over assignment(var,value)); this searches the domains, the
 *  closure certifies each candidate, and a satisfying assignment is committed.
 *  Malformed args fail visibly. */
function runSolve(store: SpaceStore, spaceId: string, args: Record<string, unknown>): string {
  const rawVars = args.variables
  if (!Array.isArray(rawVars) || rawVars.length === 0) {
    return 'error: solve needs args.variables = a non-empty array of {name, domain:[...]}'
  }
  const variables: FdVariable[] = []
  for (const raw of rawVars) {
    const v = (raw ?? {}) as { name?: unknown; domain?: unknown }
    const name = typeof v.name === 'string' ? v.name : ''
    if (!name) return 'error: each solve variable needs a string "name"'
    const domain = Array.isArray(v.domain) ? (v.domain as unknown[]) : []
    if (domain.length === 0) return `error: variable "${name}" needs a non-empty "domain" array of candidate values`
    const scalars: FdValue[] = []
    for (const d of domain) {
      if (typeof d === 'string' || typeof d === 'number' || typeof d === 'boolean') scalars.push(d)
      else return `error: variable "${name}" domain values must be string/number/boolean (got ${typeof d})`
    }
    variables.push({ name, domain: scalars })
  }
  const conflictPredicate = typeof args.conflictPredicate === 'string' ? args.conflictPredicate : undefined
  const assignPredicate = typeof args.assignPredicate === 'string' ? args.assignPredicate : undefined
  const maxNodes = typeof args.maxNodes === 'number' ? args.maxNodes : undefined
  const cp = conflictPredicate ?? 'conflict'
  try {
    const r = solveConstraintsOnBoard(store, spaceId, { variables, conflictPredicate, assignPredicate, maxNodes })
    if (r.sat) {
      const pairs = Object.entries(r.assignment).map(([k, val]) => `${k}=${String(val)}`).join(', ')
      return `solved (closure-certified, committed as assignment facts): ${pairs}  [searched ${r.nodes} candidate(s)]`
    }
    if (r.reason === 'unsat') {
      return `unsat: no assignment over the given domains avoids ${cp} (searched ${r.nodes}). Widen a domain or relax the constraint rule.`
    }
    return `budget: search hit its node cap before finding a solution (searched ${r.nodes}); raise maxNodes or shrink the problem.`
  } catch (error) {
    return `error: ${error instanceof Error ? error.message : String(error)}`
  }
}

function runValidatePlan(store: SpaceStore, spaceId: string, actionNodeIds: string[]): string {
  const v = validatePlan(store, spaceId, actionNodeIds)
  const lines = [
    `validate_plan: ${v.ok ? 'OK - every step applies and all goals are reached' : 'NOT OK'}`,
    ...v.steps.map((step) =>
      step.applicable
        ? `  #${step.index} ${step.actionNodeId}: applies`
        : `  #${step.index} ${step.actionNodeId}: BLOCKED${step.error ? ` (${step.error})` : ''}${
            step.failedPrecondition ? ` - first unmet: ${formatAtom(step.failedPrecondition)}` : ''
          }`,
    ),
  ]
  if (v.unmetGoalIds.length > 0) lines.push(`goals NOT reached: ${v.unmetGoalIds.join(', ')}`)
  if (v.shortestPrefixLength !== undefined && v.redundantStepIndices.length > 0) {
    lines.push(
      `shortest prefix: first ${v.shortestPrefixLength} step(s) reach every goal; ` +
        `steps #${v.redundantStepIndices.join(', #')} are redundant - drop them.`,
    )
  }
  return lines.join('\n')
}

function runApplyPlan(
  store: SpaceStore,
  spaceId: string,
  actionNodeIds: string[],
  requireGoals: boolean,
): string {
  const r = applyPlan(store, spaceId, actionNodeIds, { requireGoals })
  if (!r.applied) {
    return [
      `apply_plan: NOT applied${r.failedIndex !== undefined ? ` (stopped at step #${r.failedIndex})` : ''}`,
      r.failureReason ?? '',
      r.appliedActionNodeIds.length > 0
        ? `committed before the failure: ${r.appliedActionNodeIds.join(', ')}`
        : 'nothing was committed.',
    ]
      .filter(Boolean)
      .join('\n')
  }
  return [
    `apply_plan: applied ${r.appliedActionNodeIds.length} step(s): ${r.appliedActionNodeIds.join(' -> ') || 'none'}`,
    `final revision: ${r.finalRevision}`,
    formatLogicContextAsText(getLogicContext(store, spaceId)),
  ].join('\n')
}

function runSuggestPlanRepairs(store: SpaceStore, spaceId: string, actionNodeIds: string[]): string {
  const r = suggestPlanRepairs(store, spaceId, actionNodeIds)
  if (r.repairs.length === 0) {
    return `suggest_plan_repairs: no repair${r.note ? ` - ${r.note}` : ''}`
  }
  const lines = [
    `suggest_plan_repairs: ${r.repairs.length} candidate(s) for the gap at step #${r.failedIndex}` +
      (r.failedPrecondition ? ` (unmet: ${formatAtom(r.failedPrecondition)})` : ''),
    ...r.repairs.map(
      (rep, i) =>
        `  [${i}]${rep.validates ? ' (reaches goal)' : ' (runs past failure)'} insert ${rep.insertedActionNodeIds.join(' -> ')}: ` +
        `try {"actionNodeIds":${JSON.stringify(rep.actionNodeIds)}}`,
    ),
    're-run validate_plan / apply_plan on a candidate before trusting it.',
  ]
  return lines.join('\n')
}

function runSimulateAction(store: SpaceStore, spaceId: string, actionNodeId: string): string {
  if (!actionNodeId) return 'error: actionNodeId required (the id of a define_action node)'
  try {
    const r = simulateActionEffects(store, spaceId, actionNodeId)
    if (!r.applicable) {
      return `simulate ${actionNodeId}: NOT applicable\n${describePreconditionFailure(r)}`
    }
    return [
      `simulate ${actionNodeId}: applicable`,
      describeBinding(r.binding, r.bindingCandidates),
      `would add: ${r.addedAtoms.map(formatAtom).join(', ') || 'none'}`,
      `would remove: ${r.removedAtoms.map(formatAtom).join(', ') || 'none'}`,
      `new derived: ${r.newDerivedAtoms.map(formatAtom).join(', ') || 'none'}`,
      `lost derived: ${r.lostDerivedAtoms.map(formatAtom).join(', ') || 'none'}`,
      `would satisfy goals: ${r.wouldSatisfyGoalIds.join(', ') || 'none'}`,
      r.predicateConflicts.length > 0
        ? `WARNING: would introduce ${r.predicateConflicts.length} predicate conflict(s)`
        : '',
    ]
      .filter(Boolean)
      .join('\n')
  } catch (error) {
    return `error: ${error instanceof Error ? error.message : String(error)}`
  }
}

function runApplyAction(store: SpaceStore, spaceId: string, actionNodeId: string): string {
  if (!actionNodeId) return 'error: actionNodeId required (the id of a define_action node)'
  try {
    const r = deriveActionEffects(store, spaceId, actionNodeId)
    if (!r.applied) {
      return `apply ${actionNodeId}: blocked\n${describePreconditionFailure(r)}`
    }
    const header = [
      `applied ${actionNodeId}: +${r.addedFactNodeIds.length} fact(s), -${r.removedFactNodeIds.length} fact(s) (consumed facts are archived, see event ${r.eventNodeId ?? ''})`,
      describeBinding(r.binding, r.bindingCandidates),
    ]
      .filter(Boolean)
      .join('\n')
    return `${header}\n${formatLogicContextAsText(getLogicContext(store, spaceId))}`
  } catch (error) {
    return `error: ${error instanceof Error ? error.message : String(error)}`
  }
}

function applyBoardOps(
  store: SpaceStore,
  spaceId: string,
  ops: WorkingMemoryOperation[],
  attestedPredicates?: string[],
  attestedDerivations?: ApplyOptions['attestedDerivations'],
  onFailure?: (kind: ReturnType<typeof classifyFailure>['kind']) => void,
  renderBoard?: () => string,
): string {
  try {
    // Model-sourced: the attested-predicate + attested-derivation guards apply
    // (the harness writes machine facts via its own system-sourced path).
    const result = applyWorkingMemoryOperations(store, spaceId, ops, {
      format: 'text',
      source: 'model',
      attestedPredicates,
      attestedDerivations,
    })
    const warnings = result.warnings.map((w) => `warning: ${w}`).join('\n')
    return [warnings, renderBoard ? renderBoard() : result.workingMemoryText ?? ''].filter(Boolean).join('\n')
  } catch (error) {
    // Tag the failure with its stable kind + the fix to try, so the model gets
    // a categorized, actionable signal (and a friction profile can be built from
    // the same taxonomy). Pure classification of an existing signal - no new rule.
    const message = error instanceof Error ? error.message : String(error)
    const { kind, fix } = classifyFailure(error)
    onFailure?.(kind)
    return `error: ${message}\n[failure: ${kind}] ${fix}`
  }
}

/** Strip ALL teaching signals the critique system surfaces (control arm):
 *  the standing 'critique:' block AND the inline '! self-sealed' goal line. */
function stripCritiqueSection(text: string): string {
  const idx = text.indexOf('\ncritique:')
  const withoutBlock = idx >= 0 ? text.slice(0, idx) : text
  return withoutBlock
    .split('\n')
    .filter((line) => !line.includes('! self-sealed'))
    .join('\n')
}

/**
 * done-gate (absorbed from Codex codex/rulith-agent-infra a52d8ed): a non-empty
 * board must carry a recorded result before the agent may finish - otherwise
 * the model can do real work and walk away without an attested conclusion
 * (the loop-level twin of the record_result derivation gate). An empty board
 * is exempt (nothing to conclude).
 */
function validateDone(
  store: SpaceStore,
  spaceId: string,
  opts: { enforceAudit?: boolean; attestedPredicates?: string[] } = {},
): string | undefined {
  const board = getLogicContext(store, spaceId)
  const recordedResults = board.results.filter((result) =>
    result.createdBy !== 'system' && !/^applied action:/i.test(result.label))
  const evidenceNodeIds = new Set([
    ...board.goals.map((node) => node.nodeId),
    ...board.facts.map((node) => node.nodeId),
    ...board.findings.map((node) => node.nodeId),
    ...board.hypotheses.map((node) => node.nodeId),
    ...board.axioms.map((node) => node.nodeId),
    ...board.actions.map((node) => node.nodeId),
    ...board.conflicts.map((node) => node.nodeId),
  ])
  const backedResults = recordedResults.filter((result) =>
    !result.disputed &&
    result.evidenceRefs.length > 0 &&
    result.evidenceRefs.every((ref) => evidenceNodeIds.has(ref)))
  const unresolvedResultRefs = recordedResults
    .flatMap((result) => result.evidenceRefs.filter((ref) => !evidenceNodeIds.has(ref)))
  const hasBoardWork =
    board.goals.length +
      board.facts.length +
      board.findings.length +
      board.hypotheses.length +
      board.axioms.length +
      board.actions.length +
      board.conflicts.length +
      board.predicateConflicts.length >
    0
  if (!hasBoardWork) return undefined // empty board: nothing to conclude

  // Audit gate: a finish must not rest on fabrication. A goal satisfied only by
  // a bare assertion (self_sealed_goal) or a finding asserted instead of derived
  // (asserted_finding) means the conclusion is claimed, not earned - exactly
  // what the bench showed a model does when it cannot legitimately reach the
  // goal. Block done until those are derived or retracted. Off under the
  // suppressCritique control arm (which measures what the critique is worth).
  if (opts.enforceAudit !== false) {
    // self_sealed_goal: claimed by bare assertion. asserted_finding: claimed
    // not derived. vacuous_rule: "derived" via a body that just renames the
    // head - a zero-power rule that launders a claim into a derived fact so it
    // dodges the self_sealed check (the bench's 4th cheat). All three mean the
    // conclusion is not earned; block done until they are gone.
    const blocking = board.critique.filter(
      (item) =>
        item.kind === 'self_sealed_goal' ||
        item.kind === 'asserted_finding' ||
        item.kind === 'vacuous_rule',
    )
    if (blocking.length > 0) {
      return (
        `error: done rejected: ${blocking.length} conclusion(s) are claimed, not earned — ` +
        blocking.map((item) => `[${item.kind}] ${item.nodeId}: ${item.message}`).join('; ') +
        `. Each must be EARNED: assert the primitive observation you verified, add_axiom a rule that ` +
        `derives the goal/finding from it, and let the closure produce it (or retract_node it if it does not hold). ` +
        `Then call done again. board_critique lists these any time.`
      )
    }

    // Evidence-policy ENFORCEMENT (not just surfacing): a satisfied obligation grounded WEAKER than a
    // TRUSTED required_floor / human-tier constraint cannot honestly close — the trust floor is policy,
    // and a bare assertion does not clear it. Inert without a trusted policy fact (so plain boards are
    // unaffected). Mirrors the surfaced `trust_inversions` critique; off under the suppressCritique arm.
    const inversions = trustInversions(board.facts, { attestedPredicates: opts.attestedPredicates })
    if (inversions.length > 0) {
      return (
        `error: done rejected: ${inversions.length} obligation(s) are satisfied BELOW their required evidence floor — ` +
        inversions.map((t) => `[${t.source}] ${t.scope}: grounded "${t.actualTier}" < required "${t.requiredTier}"`).join('; ') +
        `. A trusted policy sets this bar; a bare assertion does not clear it. Observe the evidence through a ` +
        `trusted channel (tool/system) or a passing machine check, or have the policy revised — then call done again.`
      )
    }
  }

  if (backedResults.length > 0) return undefined
  if (unresolvedResultRefs.length > 0) {
    const sample = [...new Set(unresolvedResultRefs)].slice(0, 5).join(', ')
    return (
      `error: done rejected: this board has record_result evidenceRefs that do not resolve to current board nodes: ${sample}. ` +
      'Cite actual node ids from get_logic_context (facts, findings, axioms, goals, actions, or conflicts), ' +
      'or first assert/derive the missing evidence on the board. Then record_result with resolvable evidenceRefs and call done again.'
    )
  }
  if (recordedResults.length > 0) {
    return (
      'error: done rejected: this board has an agent record_result, but no usable recorded result cites evidenceRefs. ' +
      'Replace or revise the conclusion: call update_working_memory with record_result{id,label,summary,evidenceRefs} ' +
      'where evidenceRefs cite the observations, derived facts, checks, or rules that back the conclusion. ' +
      'If an existing result is wrong or disputed, retract_node it first; then call done again.'
    )
  }
  return (
    'error: done rejected: this board has working state but no agent-recorded result backed by evidenceRefs. ' +
    'First call update_working_memory with record_result{id,label,summary,evidenceRefs}. ' +
    'If the current board state is wrong, use retract_node/revise_fact before record_result; then call done again.'
  )
}

function boardSummary(store: SpaceStore, spaceId: string): string {
  const board = getLogicContext(store, spaceId)
  const results = board.results.map((r) => `- ${r.label}: ${r.summary}`).join('\n')
  return results || `(${board.stats.facts} facts, ${board.stats.findings} findings, no recorded result)`
}
