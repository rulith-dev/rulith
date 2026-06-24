import type { PredicateAtom, SpaceNode } from '../model/types.js'
import type { PremiseTier } from './premise-provenance.js'
import { detectPredicateConflicts, detectFunctionalConflicts, type FunctionalConflict, type FunctionalDependency } from '../kernel/conflict.js'
import {
  atomHasVariables,
  atomHolds,
  formatAtoms as formatAtomList,
  type PredicateFact,
  type RuleDefinition,
} from '../kernel/predicate.js'
import type { SpaceStore } from '../storage/space-store.js'
import {
  abduceMissingFacts,
  abduceProducingActions,
  collectActionDefinitions,
  type AbductionHint,
  type ActionDefinition,
  type ActionHint,
} from './abduction.js'
import { logicallyUsableNodes } from './semantic-active.js'
import { critiqueBoard, formatCritique, type BoardCritique } from './board-critique.js'
import { formatAtom } from './semantic-derivation.js'
import { collectPredicateVocabulary, formatVocabulary } from './vocabulary.js'

/** Facts with this predicate are task outputs and listed under findings. */
export const FINDING_PREDICATE = 'finding'

export type LogicContextGoal = {
  nodeId: string
  label: string
  desired: PredicateAtom[]
  satisfied: boolean
  /** Satisfied ONLY by bare assertions - no derived/effect fact participates
   *  (notebook mode: the model rubber-stamped its own claim). Soft signal. */
  selfSealed?: boolean
  /** For unsatisfied goals: which facts would close the gap, per rule. */
  hints: AbductionHint[]
  /** For unsatisfied goals: which defined actions could produce the missing atoms. */
  actionHints: ActionHint[]
  status: SpaceNode['status']
  confidence: number
}

export type LogicContextFact = {
  nodeId: string
  atom: PredicateAtom
  label?: string
  summary?: string
  status: SpaceNode['status']
  confidence: number
  evidenceRefs: string[]
  /** Who put this fact on the board: 'system'/'tool' = a trusted channel,
   *  'agent'/'user' = a free assertion. Read by premise-provenance grounding. */
  createdBy: SpaceNode['createdBy']
  /** Typed trust tier when the fact entered via a trusted channel (else undefined,
   *  and grounding falls back to the createdBy default). Mirror of SpaceNode.trustTier. */
  trustTier?: PremiseTier
  /** The rule closure stands behind this fact. */
  derived: boolean
  /** Asserted by applying an action (transformation product). */
  effect?: boolean
  /** Evidence chain touches a contradicted fact (p and not-p both active). */
  disputed?: boolean
}

export type LogicContextAxiom = {
  nodeId: string
  label: string
  when: PredicateAtom[]
  then: PredicateAtom[]
  status: SpaceNode['status']
  confidence: number
}

export type LogicContextAction = {
  nodeId: string
  label: string
  action: string
  preconditions: PredicateAtom[]
  effects: PredicateAtom[]
  status: SpaceNode['status']
  confidence: number
}

export type LogicContextNote = {
  nodeId: string
  label: string
  summary: string
  status: SpaceNode['status']
  confidence: number
  evidenceRefs: string[]
  createdBy: SpaceNode['createdBy']
  disputed?: boolean
}

export type LogicContextPredicateConflict = {
  atom: PredicateAtom
  positiveFactId: string
  negativeFactId: string
}

export type LogicContextHypothesis = {
  nodeId: string
  atom: PredicateAtom
  status: 'open' | 'supported' | 'refuted'
  /** For open hypotheses: which facts would decide them, per rule. */
  hints: AbductionHint[]
  /** For pattern hypotheses: the instances supporting them. */
  instances?: PredicateAtom[]
  label: string
  confidence: number
}

export type LogicContext = {
  spaceId: string
  title: string
  goals: LogicContextGoal[]
  facts: LogicContextFact[]
  findings: LogicContextFact[]
  hypotheses: LogicContextHypothesis[]
  axioms: LogicContextAxiom[]
  actions: LogicContextAction[]
  results: LogicContextNote[]
  conflicts: LogicContextNote[]
  predicateConflicts: LogicContextPredicateConflict[]
  functionalConflicts: FunctionalConflict[]
  vocabulary: string[]
  /** Standing board-health problems re-derived from state (see board-critique). */
  critique: BoardCritique[]
  stats: {
    goals: number
    facts: number
    findings: number
    hypotheses: number
    openHypotheses: number
    axioms: number
    actions: number
    results: number
    conflicts: number
    predicateConflicts: number
    functionalConflicts: number
  }
}

export function getLogicContext(store: SpaceStore, spaceId: string): LogicContext {
  const space = store.getSpace(spaceId)
  const allNodes = store.listNodes(spaceId)
  const nodes = logicallyUsableNodes(allNodes)
  const allFacts = nodes.flatMap(toLogicFact)
  const facts = allFacts.filter((fact) => fact.atom.predicate !== FINDING_PREDICATE)
  const findings = allFacts.filter((fact) => fact.atom.predicate === FINDING_PREDICATE)
  const axioms = nodes.flatMap(toLogicAxiom)

  // Abduction inputs: active rules and the full active atom set.
  const ruleDefinitions: RuleDefinition[] = axioms.map((axiom) => ({
    id: axiom.nodeId,
    when: axiom.when,
    then: axiom.then,
  }))
  const activeAtoms: PredicateFact[] = allFacts.map((fact) => ({
    id: fact.nodeId,
    atom: fact.atom,
  }))
  const activeAtomList = allFacts.map((fact) => fact.atom)
  const hintsFor = (atom: PredicateAtom): AbductionHint[] =>
    abduceMissingFacts(atom, ruleDefinitions, activeAtoms)

  // Producing-action hints: judged by the same matcher simulate/apply use.
  const actionDefinitions: ActionDefinition[] = collectActionDefinitions(nodes)
  const actionHintsFor = (atom: PredicateAtom): ActionHint[] =>
    abduceProducingActions(atom, actionDefinitions, activeAtoms)

  const backedAtoms = allFacts.filter((fact) => fact.derived || fact.effect).map((fact) => fact.atom)
  const goals = nodes.flatMap((node) =>
    toLogicGoal(node, activeAtomList, backedAtoms, hintsFor, actionHintsFor),
  )
  const hypotheses = allNodes.flatMap((node) =>
    toLogicHypothesis(node, activeAtomList, hintsFor),
  )
  const actions = nodes.flatMap(toLogicAction)
  const results = nodes.filter((node) => node.type === 'result').map(toLogicNote)
  const conflicts = nodes.filter((node) => node.type === 'conflict').map(toLogicNote)
  const predicateConflicts = detectPredicateConflicts(
    allFacts.map((fact) => ({ id: fact.nodeId, atom: fact.atom })),
  )
  // Functional-dependency conflicts: the domain declares which predicates are functional via a board
  // fact functional_dependency(predicate, key) (key may be comma-separated); the kernel adjudicates
  // (foundations: kernel decides, domain supplies semantics). Inert without any such declaration.
  // Folded into the same conflict pipeline below so disputed-taint + the conflicts section get it free.
  // No trusted-channel gate: declaring a functional dependency only TIGHTENS (it can never launder a
  // weaker conclusion), so a model self-declaring one is self-binding, not an attack surface.
  const functionalDependencies: FunctionalDependency[] = allFacts
    .filter((fact) => fact.atom.predicate === 'functional_dependency')
    .map((fact) => {
      const args = (fact.atom.args ?? {}) as Record<string, unknown>
      const predicate = typeof args.predicate === 'string' ? args.predicate : ''
      const key = typeof args.key === 'string' ? args.key.split(',').map((s) => s.trim()).filter(Boolean) : []
      return { predicate, key }
    })
    .filter((dep) => dep.predicate !== '' && dep.key.length > 0)
  const functionalConflicts = detectFunctionalConflicts(
    allFacts.map((fact) => ({ id: fact.nodeId, atom: fact.atom })),
    functionalDependencies,
  )
  const vocabulary = formatVocabulary(collectPredicateVocabulary(allNodes))

  // Paraconsistent taint: contradictions do not explode the closure, but
  // everything whose evidence chain touches a contradicted fact is
  // marked disputed so the model resolves the conflict before relying
  // on downstream conclusions.
  const disputedIds = collectDisputed(allNodes, predicateConflicts, functionalConflicts.flatMap((c) => c.factIds))
  for (const fact of [...facts, ...findings]) {
    if (disputedIds.has(fact.nodeId)) fact.disputed = true
  }
  for (const note of results) {
    if (disputedIds.has(note.nodeId)) note.disputed = true
  }

  const context: LogicContext = {
    spaceId: space.id,
    title: space.title,
    goals,
    facts,
    findings,
    hypotheses,
    axioms,
    actions,
    results,
    conflicts,
    predicateConflicts,
    functionalConflicts,
    vocabulary,
    critique: [],
    stats: {
      goals: goals.length,
      facts: facts.length,
      findings: findings.length,
      hypotheses: hypotheses.length,
      openHypotheses: hypotheses.filter((hypothesis) => hypothesis.status === 'open').length,
      axioms: axioms.length,
      actions: actions.length,
      results: results.length,
      conflicts: conflicts.length,
      predicateConflicts: predicateConflicts.length,
      functionalConflicts: functionalConflicts.length,
    },
  }
  context.critique = critiqueBoard(context)
  return context
}

/** Inline evidence-chain provenance for board text (absorbed from Codex
 *  e619d4d): show what each conclusion rests on, every turn. */
function formatEvidenceRefs(refs: string[]): string {
  return refs.length > 0 ? ` <- ${refs.join(', ')}` : ''
}

function formatKeyArgs(key: Record<string, unknown>): string {
  return Object.entries(key).map(([k, v]) => `${k}=${String(v)}`).join(', ')
}

/** Config/policy facts the MODEL's board view hides: domain/host declarations the model does NOT
 *  drive (e.g. functional_dependency). An unfamiliar predicate sitting in every board view can
 *  distract a weak driver (the same "extra stuff for the weak model" trap as an over-long prompt).
 *  The kernel still reads them from the board (functional-conflict detection / derive_aggregate
 *  refusal), and the human console (consoleView) renders them separately. */
const MODEL_HIDDEN_PREDICATES = new Set<string>(['functional_dependency'])

export function formatLogicContextAsText(context: LogicContext): string {
  // Is a model-recorded result already on the board? Matches validateDone's `recordedResults`
  // (not a system event, not an "applied action" log). Once one exists, a satisfied goal must point
  // the driver at `done`, NOT at record_result again — otherwise the model re-issues the same
  // record_result id turn after turn (duplicate-id rejected, board unchanged) to the turn limit
  // (the observed arith re-record loop).
  const hasRecordedResult = context.results.some(
    (r) => r.createdBy !== 'system' && !/^applied action:/i.test(r.label),
  )
  return [
    `logic_context ${context.spaceId}: ${context.title}`,
    formatSection(
      'goals',
      context.goals.flatMap((goal) => [
        `${goal.nodeId}: ${goal.desired.map(formatAtom).join(' AND ')} [${goal.satisfied ? 'satisfied' : 'open'}] (${goal.label})`,
        ...(goal.selfSealed
          ? [
              '  ! self-sealed: satisfied only by a bare assertion you wrote, not derived by any rule. ' +
                'A goal/finding earns "satisfied" from the closure - add a rule that derives it from more ' +
                'primitive facts, or this is just a note to yourself.',
            ]
          : []),
        // A satisfied (not self-sealed) goal tells the driver to STOP re-driving and close out:
        // cite the derived facts in record_result, then done. Without this the model keeps re-
        // asserting the same batch turn after turn (arith converged at turn 1 but ran to 9-10).
        ...(goal.satisfied
          ? goal.selfSealed
            ? []
            : [
                hasRecordedResult
                  ? '  ✓ satisfied by the closure and a result is recorded — call done to finish. ' +
                    'Re-recording the same result changes nothing (a duplicate id is rejected).'
                  : '  ✓ satisfied by the closure. Re-asserting facts already on the board changes nothing — ' +
                    'if this goal was the task, call record_result (citing the derived facts), then done.',
              ]
          : formatGoalGuidance(goal, context.facts)),
      ]),
    ),
    formatSection(
      'facts',
      context.facts
        .filter((fact) => !MODEL_HIDDEN_PREDICATES.has(fact.atom.predicate))
        .map((fact) => {
          const source = fact.derived ? 'derived' : fact.effect ? 'effect' : 'asserted'
          const disputed = fact.disputed ? ' [disputed]' : ''
          return `${fact.nodeId}: ${formatAtom(fact.atom)} [${source}]${disputed}${formatEvidenceRefs(fact.evidenceRefs)}`
        }),
    ),
    formatSection(
      'hypotheses',
      context.hypotheses.flatMap((hypothesis) => [
        `${hypothesis.nodeId}: ${formatAtom(hypothesis.atom)} [${hypothesis.status}]`,
        ...(hypothesis.instances ?? []).map(
          (instance) => `  instance: ${formatAtom(instance)}`,
        ),
        ...(hypothesis.status === 'open' ? formatHints(hypothesis.hints) : []),
      ]),
    ),
    formatSection(
      'findings',
      context.findings.map((finding) => {
        // An asserted finding is an unproven claim; a derived one is a
        // conclusion the rule engine stands behind. Keep them distinguishable.
        const source = finding.derived ? 'derived' : 'asserted, not derived'
        return `${finding.nodeId}: ${formatAtom(finding.atom)} [${source}]${finding.disputed ? ' [disputed]' : ''}${formatEvidenceRefs(finding.evidenceRefs)}`
      }),
    ),
    formatSection(
      'axioms',
      context.axioms.map(
        (axiom) =>
          `${axiom.nodeId}: IF ${axiom.when.map(formatAtom).join(' AND ')} THEN ${axiom.then
            .map(formatAtom)
            .join(' AND ')} (${axiom.label})`,
      ),
    ),
    formatSection(
      'actions',
      context.actions.map(
        (action) =>
          `${action.nodeId}: ${action.action}; PRE ${formatAtoms(action.preconditions)}; EFFECT ${formatAtoms(
            action.effects,
          )}${action.label && action.label !== action.action ? ` (${action.label})` : ''}`,
      ),
    ),
    formatSection(
      'results',
      context.results.map(
        (result) =>
          `${result.nodeId}: ${result.label} - ${result.summary}${result.disputed ? ' [disputed]' : ''}${formatEvidenceRefs(result.evidenceRefs)}`,
      ),
    ),
    formatSection(
      'conflicts',
      [
        ...context.conflicts.map(
          (conflict) => `${conflict.nodeId}: ${conflict.label} - ${conflict.summary}`,
        ),
        ...context.predicateConflicts.map(
          (conflict) =>
            `predicate contradiction: ${conflict.positiveFactId} contradicts ${conflict.negativeFactId} on ${formatAtom(conflict.atom)}`,
        ),
        ...context.functionalConflicts.map(
          (conflict) =>
            `functional conflict: ${conflict.predicate}(${formatKeyArgs(conflict.key)}) has ${conflict.factIds.length} disagreeing values [${conflict.factIds.join(', ')}] - same key, different value; one must be wrong (a bare assertion contradicting a derivation, or a stale value). Resolve before summing/relying on them.`,
        ),
      ],
    ),
    formatSection(
      'vocabulary',
      context.vocabulary.filter((v) => ![...MODEL_HIDDEN_PREDICATES].some((p) => v.startsWith(`${p}(`))),
    ),
    ...formatCritique(context.critique),
    ...formatNextSteps(nextSteps(context)),
  ].join('\n')
}

function collectDisputed(
  nodes: SpaceNode[],
  conflicts: Array<{ positiveFactId: string; negativeFactId: string }>,
  extraSeeds: Iterable<string> = [],
): Set<string> {
  const disputed = new Set<string>()
  for (const conflict of conflicts) {
    disputed.add(conflict.positiveFactId)
    disputed.add(conflict.negativeFactId)
  }
  for (const id of extraSeeds) disputed.add(id)
  let changed = disputed.size > 0
  while (changed) {
    changed = false
    for (const node of nodes) {
      if (disputed.has(node.id)) continue
      if (node.evidenceRefs.some((ref) => disputed.has(ref))) {
        disputed.add(node.id)
        changed = true
      }
    }
  }
  return disputed
}

function toLogicGoal(
  node: SpaceNode,
  activeAtoms: PredicateAtom[],
  backedAtoms: PredicateAtom[],
  hintsFor: (atom: PredicateAtom) => AbductionHint[],
  actionHintsFor: (atom: PredicateAtom) => ActionHint[],
): LogicContextGoal[] {
  if (node.type !== 'goal' || node.semantic?.kind !== 'goal') return []
  const desired = node.semantic.desired ?? []
  const unsatisfied = desired.filter((atom) => !atomHolds(atom, activeAtoms))
  const satisfied = desired.length > 0 && unsatisfied.length === 0
  // Self-sealed: satisfied, but NO desired atom is backed by a derived or
  // action-effect fact - every match is a bare assertion the model wrote.
  const selfSealed = satisfied && !desired.some((atom) => atomHolds(atom, backedAtoms))
  return [
    {
      nodeId: node.id,
      label: node.label,
      desired,
      satisfied,
      ...(selfSealed ? { selfSealed: true } : {}),
      hints: unsatisfied.flatMap(hintsFor),
      actionHints: unsatisfied.flatMap(actionHintsFor),
      status: node.status,
      confidence: node.confidence,
    },
  ]
}

/**
 * A fact node the rule CLOSURE stands behind, vs. a bare agent assertion.
 * This must be closure-only: the old `createdBy === 'system'` clause also
 * matched action-effect facts, which are the model's own construct (it
 * defined the action) — counting them as derived let a model launder
 * finding(...) past the record_result derivation gate by routing it
 * through an action effect, and rendered action products as [derived]
 * (conflating "the closure proved this" with "my action asserted this").
 */
export function isDerivedFactNode(node: SpaceNode): boolean {
  return (
    node.id.startsWith('derived:') ||
    node.summary.startsWith('Derived fact:') ||
    node.summary.startsWith('Rule-derived fact:')
  )
}

/** A fact asserted by applying an action (transformation product). */
export function isActionEffectFactNode(node: SpaceNode): boolean {
  return node.summary.startsWith('Action-effect fact:')
}

function toLogicFact(node: SpaceNode): LogicContextFact[] {
  if (node.type !== 'fact' || node.semantic?.kind !== 'predicate') return []
  return [
    {
      nodeId: node.id,
      label: node.label,
      summary: node.summary,
      atom: {
        predicate: node.semantic.predicate ?? node.label,
        args: node.semantic.args,
        negated: node.semantic.negated,
      },
      status: node.status,
      confidence: node.confidence,
      evidenceRefs: node.evidenceRefs,
      createdBy: node.createdBy,
      trustTier: node.trustTier as PremiseTier | undefined,
      derived: isDerivedFactNode(node),
      effect: isActionEffectFactNode(node) || undefined,
    },
  ]
}

function toLogicHypothesis(
  node: SpaceNode,
  activeAtoms: PredicateAtom[],
  hintsFor: (atom: PredicateAtom) => AbductionHint[],
): LogicContextHypothesis[] {
  if (node.type !== 'hypothesis' || node.semantic?.kind !== 'predicate') return []
  if (node.status === 'archived') return []
  const status =
    node.status === 'supported' ? 'supported' : node.status === 'rejected' ? 'refuted' : 'open'
  const atom: PredicateAtom = {
    predicate: node.semantic.predicate ?? node.label,
    args: node.semantic.args,
    negated: node.semantic.negated,
  }
  const instances =
    status === 'supported' && atomHasVariables(atom)
      ? activeAtoms.filter((fact) => atomHolds(atom, [fact]))
      : undefined
  return [
    {
      nodeId: node.id,
      atom,
      status,
      hints: status === 'open' ? hintsFor(atom) : [],
      instances,
      label: node.label,
      confidence: node.confidence,
    },
  ]
}

function toLogicAxiom(node: SpaceNode): LogicContextAxiom[] {
  if (node.type !== 'axiom' || node.semantic?.kind !== 'axiom') return []
  return [
    {
      nodeId: node.id,
      label: node.label,
      when: node.semantic.when ?? [],
      then: node.semantic.then ?? [],
      status: node.status,
      confidence: node.confidence,
    },
  ]
}

function toLogicAction(node: SpaceNode): LogicContextAction[] {
  if (node.type !== 'action' || node.semantic?.kind !== 'action') return []
  return [
    {
      nodeId: node.id,
      label: node.label,
      action: node.semantic.action ?? node.label,
      preconditions: node.semantic.preconditions ?? [],
      effects: node.semantic.effects ?? [],
      status: node.status,
      confidence: node.confidence,
    },
  ]
}

function toLogicNote(node: SpaceNode): LogicContextNote {
  return {
    nodeId: node.id,
    label: node.label,
    summary: node.summary,
    status: node.status,
    confidence: node.confidence,
    evidenceRefs: node.evidenceRefs,
    createdBy: node.createdBy,
  }
}

function formatHints(hints: AbductionHint[]): string[] {
  if (hints.length === 0) {
    // The kernel knows no rule can derive this atom - say so instead of
    // staying silent, so the model learns to add a rule or observe directly.
    return ['  no rule derives this yet: add_axiom whose "then" matches it, or assert the fact directly']
  }
  return hints
    .slice(0, 3)
    .map(
      (hint) =>
        `  needs via ${hint.ruleId}: ${hint.missing.map(formatAtom).join(' AND ')}`,
    )
}

/**
 * Guidance lines for an open goal: rule paths (abduction) AND producing
 * actions. The bare-assertion fallback only appears when NEITHER exists —
 * with a producing action on the board, suggesting "assert the fact
 * directly" would teach exactly the laundering move the derivation gate
 * blocks (apply_action is the honest way to make the atom true).
 */
function formatGoalGuidance(goal: LogicContextGoal, facts: LogicContextFact[] = []): string[] {
  const ruleLines =
    goal.hints.length > 0
      ? goal.hints
          .slice(0, 3)
          .map(
            (hint) =>
              `  needs via ${hint.ruleId}: ${hint.missing.map(formatAtom).join(' AND ')}`,
          )
      : []
  const actionLines = goal.actionHints.slice(0, 3).map(formatActionHint)
  if (ruleLines.length === 0 && actionLines.length === 0) {
    return [
      '  no rule derives this yet: add_axiom whose "then" matches it, or assert the fact directly',
      ...aggregateGapHint(goal, facts),
    ]
  }
  return [...ruleLines, ...actionLines]
}

/**
 * Data-aware gap hint (#32 problem 8, cross-review item 5): when an open
 * goal wants a predicate nothing derives, and same-predicate numeric facts
 * are stacking up on the board, the missing piece is almost always the
 * total - and the wrong move models reach for is a recursive accumulator.
 * Point at derive_aggregate with the concrete predicate/arg names filled in.
 */
function aggregateGapHint(goal: LogicContextGoal, facts: LogicContextFact[]): string[] {
  const goalPredicates = new Set(goal.desired.map((atom) => atom.predicate))
  const counts = new Map<string, { count: number; numericArg?: string }>()
  for (const fact of facts) {
    if (goalPredicates.has(fact.atom.predicate)) continue
    const entry = counts.get(fact.atom.predicate) ?? { count: 0 }
    entry.count += 1
    if (!entry.numericArg) {
      const numericKey = Object.entries(fact.atom.args ?? {}).find(
        ([, value]) => typeof value === 'number',
      )?.[0]
      if (numericKey) entry.numericArg = numericKey
    }
    counts.set(fact.atom.predicate, entry)
  }
  const candidate = [...counts.entries()]
    .filter(([, entry]) => entry.count >= 3 && entry.numericArg !== undefined)
    .sort((a, b) => b[1].count - a[1].count)[0]
  if (!candidate) return []
  const [predicate, entry] = candidate
  const goalAtom = goal.desired[0]
  const intoArg = Object.keys(goalAtom?.args ?? {})[0] ?? 'value'
  return [
    `  ${entry.count} ${predicate}(...) facts carry numeric "${entry.numericArg}" - to total them in one op: ` +
      `{"op":"derive_aggregate","id":"agg_1","source":{"predicate":"${predicate}","valueArg":"${entry.numericArg}"},` +
      `"into":{"predicate":"${goalAtom?.predicate ?? 'grand_total'}","valueArg":"${intoArg}"}} ` +
      `(do NOT write a rule whose head feeds its own body - closures cannot loop)`,
  ]
}

function formatActionHint(hint: ActionHint): string {
  const status = hint.applicable
    ? '[preconditions hold - apply_action]'
    : `[blocked on ${hint.blockedOn ? formatAtom(hint.blockedOn) : 'unsatisfied preconditions'}]`
  return `  producible via action ${hint.actionNodeId}: ${formatAtom(hint.produces)} ${status}`
}

/** Render next-steps lines (empty when the board is healthy). */
function formatNextSteps(steps: string[]): string[] {
  if (steps.length === 0) return []
  return ['next steps:', ...steps.map((step) => `- ${step}`)]
}

/**
 * Pure rendering helper: aggregate already-derived guidance into a flat list
 * of next-step strings. No new inference — only re-presents data already on
 * the context:
 *   - open goals' abduction hints (goal.hints) and action hints (goal.actionHints)
 *   - standing board-health problems (context.critique[].message)
 * Returns an empty array when the board is healthy and all goals are satisfied.
 */
export function nextSteps(ctx: LogicContext): string[] {
  const steps: string[] = []

  for (const goal of ctx.goals) {
    if (goal.satisfied) continue
    for (const hint of goal.hints.slice(0, 3)) {
      steps.push(
        `[goal ${goal.nodeId}] needs via ${hint.ruleId}: ${hint.missing.map(formatAtom).join(' AND ')}`,
      )
    }
    for (const hint of goal.actionHints.slice(0, 3)) {
      const status = hint.applicable
        ? '[preconditions hold - apply_action]'
        : `[blocked on ${hint.blockedOn ? formatAtom(hint.blockedOn) : 'unsatisfied preconditions'}]`
      steps.push(
        `[goal ${goal.nodeId}] producible via action ${hint.actionNodeId}: ${formatAtom(hint.produces)} ${status}`,
      )
    }
    if (goal.hints.length === 0 && goal.actionHints.length === 0) {
      steps.push(
        `[goal ${goal.nodeId}] no rule or action can close "${goal.label}" - add_axiom or define_action`,
      )
    }
  }

  for (const item of ctx.critique) {
    steps.push(`[critique ${item.kind}] ${item.message}`)
  }

  return steps
}

function formatSection(title: string, lines: string[]): string {
  if (lines.length === 0) return `${title}: none`
  return [`${title}:`, ...lines.map((line) => `- ${line}`)].join('\n')
}

function formatAtoms(atoms: PredicateAtom[]): string {
  return formatAtomList(atoms, ' AND ')
}
