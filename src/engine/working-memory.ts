import type { CreateNodeInput, Creator, PredicateAtom, SemanticArgs, SpaceNode } from '../model/types.js'
import {
  ARITHMETIC_BUILTINS,
  COMPARISON_BUILTINS,
  isArithmeticBuiltin,
  isComparisonBuiltin,
  isBuiltinPredicate,
} from '../kernel/builtins.js'
import { assertActionSafety, assertRuleSafety } from '../kernel/safety.js'
import { detectVacuousRule, detectUnfirableRule } from '../kernel/rule-quality.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import type { NodePatch, SpaceStore } from '../storage/space-store.js'
import { applySemanticRules, type SemanticRuleApplicationResult } from './semantic-rules.js'
import {
  FINDING_PREDICATE,
  getLogicContext,
  formatLogicContextAsText,
  isDerivedFactNode,
  type LogicContext,
} from './logic-context.js'
import { logicallyUsableNodes } from './semantic-active.js'
import { GoalpostMovingError, PROTECTED_GOALPOSTS, goalpostKey, amendmentMatches, formatGoalpostKey, type ProtectedGoalpost } from './goalpost.js'
import { isVariable } from '../kernel/predicate.js'
import { formatAtom } from './semantic-derivation.js'
import { retractNode } from './semantic-tools.js'
import {
  checkAtomSignature,
  collectPredicateVocabulary,
  registerAtom,
} from './vocabulary.js'

export type WorkingMemoryOperation =
  | ({ op: 'declare_goal' } & {
      id?: string
      label: string
      summary?: string
      desired: PredicateAtom[]
      confidence?: number
      activation?: number
    })
  | ({ op: 'assert_fact' } & {
      id?: string
      label?: string
      summary?: string
      predicate: string
      args?: SemanticArgs
      negated?: boolean
      confidence?: number
      activation?: number
      evidenceRefs?: string[]
    })
  | ({ op: 'add_axiom' } & {
      id?: string
      label: string
      summary?: string
      when: PredicateAtom[]
      then: PredicateAtom[]
      confidence?: number
      activation?: number
    })
  | ({ op: 'derive_aggregate' } & {
      id?: string
      label?: string
      summary?: string
      /** kinds: "sum"/"min"/"max" (need source.valueArg) and "count". */
      kind?: 'sum' | 'count' | 'min' | 'max' | 'avg'
      source: { predicate: string; valueArg?: string }
      into: { predicate: string; valueArg: string }
      /** v2 optional equality filter: only source facts whose args[where.arg]
       *  equals where.equals are aggregated. Absent = aggregate all (v1). */
      where?: { arg: string; equals: string | number | boolean }
      /** v3 optional grouping: bucket source facts by the distinct values of
       *  args[group_by], emit one into-fact per bucket carrying the group
       *  value. Composes with where (filter first, then group). */
      group_by?: string
    })
  | ({ op: 'define_action' } & {
      id?: string
      label: string
      summary?: string
      action: string
      preconditions?: PredicateAtom[]
      effects?: PredicateAtom[]
      confidence?: number
      activation?: number
    })
  | ({ op: 'declare_hypothesis' } & {
      id?: string
      label?: string
      summary?: string
      predicate: string
      args?: SemanticArgs
      negated?: boolean
      confidence?: number
      activation?: number
    })
  | ({ op: 'record_result' } & {
      id?: string
      label: string
      summary?: string
      evidenceRefs?: string[]
      confidence?: number
    })
  | ({ op: 'record_conflict' } & {
      id?: string
      label: string
      summary?: string
      evidenceRefs?: string[]
      confidence?: number
    })
  | {
      op: 'retract_node'
      nodeId: string
      reason?: string
    }
  | ({ op: 'revise_fact' } & {
      nodeId: string
      reason?: string
      id?: string
      label?: string
      summary?: string
      predicate: string
      args?: SemanticArgs
      negated?: boolean
      confidence?: number
      activation?: number
      evidenceRefs?: string[]
    })

export type WorkingMemoryOperationResult = {
  index: number
  op: WorkingMemoryOperation['op']
  nodeIds: string[]
  retractedNodeIds: string[]
}

export type WorkingMemoryApplyResult = {
  operationResults: WorkingMemoryOperationResult[]
  semanticRuleApplication: SemanticRuleApplicationResult
  warnings: string[]
  workingMemory: LogicContext
  workingMemoryText?: string
}

const inFlightApplies = new WeakMap<SpaceStore, Set<string>>()

export function applyWorkingMemoryOperations(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
  options: ApplyOptions = {},
): WorkingMemoryApplyResult {
  // Machine-attested predicates: a model-sourced batch may not assert/revise
  // them - they stand for facts only a real runner/editor can vouch for
  // (test_result, edited, build_status). The harness writes them via a
  // system-sourced batch; the model declaring them by fiat is the not-lying
  // 命门's last block. Whole batch rejected, nothing applied.
  assertAttestedPredicates(operations, options)
  assertAttestedDerivations(operations, options)
  const inFlight = inFlightApplies.get(store) ?? new Set<string>()
  if (inFlight.has(spaceId)) {
    throw new Error(
      `another working-memory batch is already being applied on space "${spaceId}" - ` +
        `retry after it finishes so your operation starts from the latest board.`,
    )
  }
  inFlight.add(spaceId)
  inFlightApplies.set(store, inFlight)
  try {
    const transaction = createStagingTransaction(store, spaceId)
    const result = applyWorkingMemoryOperationsDirect(
      transaction.staging,
      spaceId,
      operations,
      options,
    )
    transaction.commit()
    return result
  } finally {
    inFlight.delete(spaceId)
  }
}

function applyWorkingMemoryOperationsDirect(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
  options: ApplyOptions = {},
): WorkingMemoryApplyResult {
  const operationResults: WorkingMemoryOperationResult[] = []
  const newNodeIds: string[] = []
  // derive_aggregate pins its source facts as constants at expansion time (the
  // forward engine cannot fold variable-arity), so it must expand AFTER the batch's
  // other rules have fired. Collected here and expanded post-closure (see below).
  const deferredAggregates: Array<{
    operation: Extract<WorkingMemoryOperation, { op: 'derive_aggregate' }>
    index: number
  }> = []
  const warnings: string[] = []
  const dedupedAxioms = new Set<number>()
  const vocabulary = collectPredicateVocabulary(store.listNodes(spaceId))
  const factIndex = collectFactIndex(store, spaceId)
  const ruleIndex = collectRuleIndex(store, spaceId)

  // Normalize, then validate the whole batch before applying anything,
  // so a rejected operation cannot leave the batch half-applied.
  operations = operations.map(normalizeOperationShape).map(normalizeOperationScalars)
  const existingIds = new Set(store.listNodes(spaceId).map((node) => node.id))
  const removedInBatch = new Set(
    operations.flatMap((operation) =>
      operation.op === 'retract_node' || operation.op === 'revise_fact'
        ? [operation.nodeId]
        : [],
    ),
  )
  const newIdsInBatch = new Set<string>()
  for (const operation of operations) {
    assertKnownOperation(operation)
    assertNoBuiltinAssertion(operation)
    assertValidAtoms(operation)
    assertNoForgedProvenance(operation)
    assertNoStrictPlaceholders(operation)
    assertValidNodeReference(operation)
    assertFreshNodeId(operation, existingIds, removedInBatch, newIdsInBatch)
    if (operation.op === 'add_axiom') {
      assertRuleSafety({
        id: operation.id ?? operation.label,
        when: operation.when,
        then: operation.then,
      })
    }
    if (operation.op === 'define_action') {
      assertActionSafety({
        id: operation.id ?? operation.label ?? operation.action,
        preconditions: operation.preconditions,
        effects: operation.effects,
      })
    }
  }
  assertNoReservedLabelAbuse(store, spaceId, operations, options)
  assertNoGoalpostMoving(store, spaceId, operations, options)
  assertCommittedDerivation(store, spaceId, operations, options)
  assertDerivationGate(store, spaceId, operations)

  operations.forEach((operation, index) => {
    for (const atom of operationAtoms(operation)) {
      const warning = checkAtomSignature(vocabulary, atom)
      if (warning) warnings.push(`op #${index} (${operation.op}): ${warning}`)
      registerAtom(vocabulary, atom)
    }

    if (operation.op === 'declare_goal' || operation.op === 'declare_hypothesis') {
      const placeholder = detectPlaceholderConstants(operation)
      if (placeholder) warnings.push(`op #${index} (${operation.op}): ${placeholder}`)
    }

    if (operation.op === 'assert_fact') {
      // Idempotent re-assert: an identical fact (same predicate, args,
      // sign) maps to the SAME state, so creating a second node only
      // clutters the board - real runs ignored the warning-only version
      // and accumulated duplicate copies. Reuse the existing node instead.
      const existing = factIndex.get(
        canonicalAtomKey({
          predicate: operation.predicate,
          args: operation.args,
          negated: operation.negated,
        }),
      )
      if (existing) {
        warnings.push(
          `op #${index} (assert_fact): this exact fact is already on the board as ${existing}; ` +
            `reused it instead of adding a copy (retract it first if you meant to replace it)`,
        )
        operationResults.push(operationResult(index, operation.op, [existing]))
        return
      }
    }

    if (operation.op === 'revise_fact') {
      const existing = factIndex.get(
        canonicalAtomKey({
          predicate: operation.predicate,
          args: operation.args,
          negated: operation.negated,
        }),
      )
      if (existing && existing !== operation.nodeId) {
        warnings.push(
          `op #${index} (revise_fact): the replacement is identical to ${existing}, which is already on the board`,
        )
      }
    }

    if (operation.op === 'add_axiom') {
      // Idempotent re-add: an identical rule derives nothing new; real
      // runs re-added the same rule up to 4 times after context loss.
      const existingRule = ruleIndex.get(canonicalRuleKey(operation.when, operation.then))
      if (existingRule) {
        warnings.push(
          `op #${index} (add_axiom): an identical rule is already on the board as ${existingRule}; ` +
            `reused it instead of adding a copy`,
        )
        operationResults.push(operationResult(index, operation.op, [existingRule]))
        // Its unfirability (if any) was already reported when first added.
        dedupedAxioms.add(index)
        return
      }
      const vacuous = detectVacuousRule({
        id: operation.id ?? operation.label,
        when: operation.when,
        then: operation.then,
      })
      if (vacuous) warnings.push(`op #${index} (add_axiom): ${vacuous}`)
      const unfirable = detectUnfirableRule({
        id: operation.id ?? operation.label,
        when: operation.when,
        then: operation.then,
      })
      if (unfirable) warnings.push(`op #${index} (add_axiom): ${unfirable}`)
    }

    if (operation.op === 'derive_aggregate') {
      // Defer to the post-loop closure pass: the source facts may be derived by a
      // rule in THIS SAME batch, which has not fired yet (closure runs after the
      // loop). Expanding now would see no source facts. See the deferred-expansion
      // block after the loop.
      deferredAggregates.push({ operation, index })
      return
    }

    const result = applyOperation(store, spaceId, operation, index, options.createdBy, options.trustTier)
    operationResults.push(result)
    newNodeIds.push(...result.nodeIds)

    if (operation.op === 'assert_fact' || operation.op === 'revise_fact') {
      const newId = result.nodeIds[result.nodeIds.length - 1]
      if (newId) {
        factIndex.set(
          canonicalAtomKey({
            predicate: operation.predicate,
            args: operation.args,
            negated: operation.negated,
          }),
          newId,
        )
      }
    }

    if (operation.op === 'add_axiom') {
      const newId = result.nodeIds[result.nodeIds.length - 1]
      if (newId) ruleIndex.set(canonicalRuleKey(operation.when, operation.then), newId)
    }
  })

  // Closure pass(es). With no deferred aggregate this is the single historical pass
  // (zero behaviour change). With deferred aggregates we iterate: a closure materializes
  // the ready sources, then every aggregate whose source predicate now has facts expands,
  // and we repeat. This resolves CHAINED aggregates in one batch (B.source = A.into): A
  // expands, the next closure materializes A's output, which makes B ready. A round that
  // can expand nothing (every remaining source predicate is still empty - a genuine
  // missing source or a circular dependency) expands the remainder anyway so
  // expandAggregate throws its standard "no active <pred> facts" teaching error. All
  // passes run inside the staging transaction (commit is the caller's last step), so any
  // throw still rolls the whole batch back (#32 atomicity).
  let semanticRuleApplication: SemanticRuleApplicationResult
  if (deferredAggregates.length === 0) {
    semanticRuleApplication = applySemanticRules(store, spaceId)
  } else {
    const expandAndApply = (
      operation: Extract<WorkingMemoryOperation, { op: 'derive_aggregate' }>,
      index: number,
    ): void => {
      const baseRuleId = operation.id ?? `agg_${operation.into.predicate}`
      const retractedNodeIds: string[] = []
      // Re-run sweeps the WHOLE family: the bare base id (a prior ungrouped run)
      // AND every "<base>__g_*" bucket rule from a prior grouped run. Sweeping the
      // family (not just one id) keeps group_by re-runs idempotent - otherwise a
      // vanished bucket's stale rule keeps firing.
      const familyIds = store
        .listNodes(spaceId)
        .filter((node) => node.id === baseRuleId || node.id.startsWith(`${baseRuleId}__g_`))
        .map((node) => node.id)
      for (const staleId of familyIds) {
        const removed = retractOperation(
          store,
          spaceId,
          staleId,
          'derive_aggregate re-run: previous aggregate rule(s) replaced',
          index,
          'retract_node',
        )
        retractedNodeIds.push(...removed.retractedNodeIds)
      }
      const expansions = expandAggregate(store, spaceId, operation)
      const aggNodeIds: string[] = []
      for (const expansion of expansions) {
        assertRuleSafety({ id: expansion.ruleId, when: expansion.when, then: expansion.then })
        const result = applyOperation(
          store,
          spaceId,
          { op: 'add_axiom', id: expansion.ruleId, label: expansion.label, when: expansion.when, then: expansion.then },
          index,
        )
        ruleIndex.set(
          canonicalRuleKey(expansion.when, expansion.then),
          result.nodeIds[result.nodeIds.length - 1] ?? expansion.ruleId,
        )
        aggNodeIds.push(...result.nodeIds)
      }
      operationResults.push({ index, op: 'derive_aggregate', nodeIds: aggNodeIds, retractedNodeIds })
      newNodeIds.push(...aggNodeIds)
    }

    const closures: SemanticRuleApplicationResult[] = []
    let remaining = deferredAggregates
    while (remaining.length > 0) {
      closures.push(applySemanticRules(store, spaceId))
      const facts = getLogicContext(store, spaceId).facts
      const ready = remaining.filter(({ operation }) =>
        facts.some((f) => f.atom.predicate === operation.source.predicate && f.atom.negated !== true),
      )
      // Nothing ready ⇒ the remainder is genuinely stuck (missing source or a cycle).
      // Expand it anyway: expandAggregate throws the standard teaching error, which
      // rolls the whole batch back. (Never loops: every round either shrinks
      // `remaining` or throws.)
      const toExpand = ready.length > 0 ? ready : remaining
      for (const { operation, index } of toExpand) expandAndApply(operation, index)
      remaining = ready.length > 0 ? remaining.filter((d) => !ready.includes(d)) : []
    }
    // Final closure materializes the last round's aggregate outputs and, being a
    // from-scratch recompute, drops any stale derived fact a retracted rule produced.
    closures.push(applySemanticRules(store, spaceId))
    operationResults.sort((a, b) => a.index - b.index)
    semanticRuleApplication = mergeClosures(closures)
  }
  assertResultEvidenceRefsResolvable(
    store,
    spaceId,
    operationResults
      .filter((result) => result.op === 'record_result')
      .flatMap((result) => result.nodeIds),
  )

  const workingMemory = getLogicContext(store, spaceId)
  appendUnfirableRuleWarnings(operations, workingMemory, warnings, dedupedAxioms)
  return {
    operationResults,
    semanticRuleApplication,
    warnings,
    workingMemory,
    workingMemoryText:
      options.format === 'text' ? formatLogicContextAsText(workingMemory) : undefined,
  }
}

/**
 * Net a sequence of closure passes into one result for the return contract (the
 * deferred-aggregate path runs the closure 2+ times; chained aggregates add a pass
 * per link). added/removed are netted IN ORDER: a derived fact created in one pass
 * then removed in a later one (or vice versa) ends up in whichever the LAST pass did,
 * so the `+N/-M facts` count a caller prints reflects the true net change. The
 * scalar/list fields describe the FINAL state, so they come from the last pass.
 */
function mergeClosures(passes: SemanticRuleApplicationResult[]): SemanticRuleApplicationResult {
  const last = passes[passes.length - 1]!
  const added = new Set<string>()
  const removed = new Set<string>()
  const applied = new Set<string>()
  for (const pass of passes) {
    for (const id of pass.addedFactNodeIds) {
      removed.delete(id)
      added.add(id)
    }
    for (const id of pass.removedFactNodeIds) {
      added.delete(id)
      removed.add(id)
    }
    for (const id of pass.appliedRuleNodeIds) applied.add(id)
  }
  return {
    strataCount: last.strataCount,
    appliedRuleNodeIds: [...applied],
    addedFactNodeIds: [...added],
    removedFactNodeIds: [...removed],
    satisfiedGoalNodeIds: last.satisfiedGoalNodeIds,
    hypothesisUpdates: last.hypothesisUpdates,
  }
}

type StoreMutation =
  | { op: 'add'; spaceId: string; input: CreateNodeInput & { id: string } }
  | { op: 'update'; spaceId: string; nodeId: string; patch: NodePatch }
  | { op: 'remove'; spaceId: string; nodeId: string }

function createStagingTransaction(
  target: SpaceStore,
  spaceId: string,
): { staging: SpaceStore; commit: () => void } {
  const sourceSpace = target.getSpace(spaceId)
  const memory = new MemorySpaceStore()
  memory.createSpace({
    id: sourceSpace.id,
    title: sourceSpace.title,
    summary: sourceSpace.summary,
    scopes: [...sourceSpace.scopes],
  })
  for (const node of target.listNodes(spaceId)) {
    memory.addNode(spaceId, createInputFromNode(node))
  }

  const staging = new RecordingSpaceStore(memory)
  return {
    staging,
    commit: () => {
      for (const mutation of staging.mutations) {
        switch (mutation.op) {
          case 'add':
            target.addNode(mutation.spaceId, mutation.input)
            break
          case 'update':
            target.updateNode(mutation.spaceId, mutation.nodeId, mutation.patch)
            break
          case 'remove':
            target.removeNode(mutation.spaceId, mutation.nodeId)
            break
        }
      }
    },
  }
}

class RecordingSpaceStore implements SpaceStore {
  readonly mutations: StoreMutation[] = []

  constructor(private readonly inner: MemorySpaceStore) {}

  createSpace(input: Parameters<SpaceStore['createSpace']>[0]): ReturnType<SpaceStore['createSpace']> {
    return this.inner.createSpace(input)
  }

  getSpace(spaceId: string): ReturnType<SpaceStore['getSpace']> {
    return this.inner.getSpace(spaceId)
  }

  listSpaces(): ReturnType<SpaceStore['listSpaces']> {
    return this.inner.listSpaces()
  }

  addNode(spaceId: string, input: CreateNodeInput): SpaceNode {
    const node = this.inner.addNode(spaceId, input)
    this.mutations.push({ op: 'add', spaceId, input: createInputFromNode(node) })
    return node
  }

  updateNode(spaceId: string, nodeId: string, patch: NodePatch): SpaceNode {
    const node = this.inner.updateNode(spaceId, nodeId, patch)
    this.mutations.push({ op: 'update', spaceId, nodeId, patch })
    return node
  }

  getNode(spaceId: string, nodeId: string): SpaceNode {
    return this.inner.getNode(spaceId, nodeId)
  }

  listNodes(spaceId: string): SpaceNode[] {
    return this.inner.listNodes(spaceId)
  }

  removeNode(spaceId: string, nodeId: string): void {
    this.inner.removeNode(spaceId, nodeId)
    this.mutations.push({ op: 'remove', spaceId, nodeId })
  }
}

// Staging commit replays nodes through this shape; every entry-settable
// SpaceNode field must be preserved here or it will vanish at commit time.
function createInputFromNode(node: SpaceNode): CreateNodeInput & { id: string } {
  return {
    id: node.id,
    type: node.type,
    label: node.label,
    summary: node.summary,
    status: node.status,
    confidence: node.confidence,
    activation: node.activation,
    evidenceRefs: [...node.evidenceRefs],
    semantic: node.semantic === undefined ? undefined : structuredClone(node.semantic),
    createdBy: node.createdBy,
    trustTier: node.trustTier,
  }
}

function applyOperation(
  store: SpaceStore,
  spaceId: string,
  operation: WorkingMemoryOperation,
  index: number,
  createdBy: Creator = 'agent',
  trustTier?: string,
): WorkingMemoryOperationResult {
  switch (operation.op) {
    case 'declare_goal': {
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'goal',
        label: operation.label,
        summary: operation.summary,
        confidence: operation.confidence,
        activation: operation.activation ?? 1,
        semantic: { kind: 'goal', desired: operation.desired },
        createdBy,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'assert_fact': {
      const atom = { predicate: operation.predicate, args: operation.args, negated: operation.negated }
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'fact',
        label: operation.label ?? formatAtom(atom),
        summary: operation.summary ?? `Fact: ${formatAtom(atom)}`,
        confidence: operation.confidence,
        activation: operation.activation,
        evidenceRefs: operation.evidenceRefs,
        semantic: {
          kind: 'predicate',
          predicate: operation.predicate,
          args: operation.args,
          negated: operation.negated,
        },
        createdBy,
        // tier is honored ONLY for a trusted createdBy — a model op can never forge it (fail-closed)
        trustTier: createdBy === 'tool' || createdBy === 'system' ? trustTier : undefined,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'add_axiom': {
      assertRuleSafety({
        id: operation.id ?? operation.label,
        when: operation.when,
        then: operation.then,
      })
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'axiom',
        label: operation.label,
        summary: operation.summary,
        confidence: operation.confidence,
        activation: operation.activation ?? 0.9,
        semantic: {
          kind: 'axiom',
          when: operation.when,
          then: operation.then,
        },
        createdBy,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'define_action': {
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'action',
        // Models routinely omit label; default to the action name instead of
        // storing undefined (which the board rendered literally).
        label: operation.label ?? operation.action,
        summary: operation.summary,
        confidence: operation.confidence,
        activation: operation.activation,
        semantic: {
          kind: 'action',
          action: operation.action,
          preconditions: operation.preconditions,
          effects: operation.effects,
        },
        createdBy,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'declare_hypothesis': {
      const atom = { predicate: operation.predicate, args: operation.args, negated: operation.negated }
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'hypothesis',
        label: operation.label ?? formatAtom(atom),
        summary: operation.summary ?? `Hypothesis: ${formatAtom(atom)}`,
        status: 'open',
        confidence: operation.confidence,
        activation: operation.activation,
        semantic: {
          kind: 'predicate',
          predicate: operation.predicate,
          args: operation.args,
          negated: operation.negated,
        },
        createdBy,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'record_result':
    case 'record_conflict': {
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: operation.op === 'record_result' ? 'result' : 'conflict',
        label: operation.label,
        summary: operation.summary,
        status: 'verified',
        confidence: operation.confidence,
        evidenceRefs: operation.evidenceRefs,
        createdBy,
      })
      return operationResult(index, operation.op, [node.id])
    }
    case 'derive_aggregate': {
      // Expanded into a chain rule by applyWorkingMemoryOperations before
      // execution ever reaches this generic dispatcher.
      throw new Error('internal: derive_aggregate must be expanded before applyOperation')
    }
    case 'retract_node': {
      return retractOperation(store, spaceId, operation.nodeId, operation.reason, index, operation.op)
    }
    case 'revise_fact': {
      const retracted = retractOperation(
        store,
        spaceId,
        operation.nodeId,
        operation.reason,
        index,
        operation.op,
      )
      const atom = { predicate: operation.predicate, args: operation.args, negated: operation.negated }
      const node = store.addNode(spaceId, {
        id: operation.id,
        type: 'fact',
        label: operation.label ?? formatAtom(atom),
        summary: operation.summary ?? `Fact: ${formatAtom(atom)}`,
        confidence: operation.confidence,
        activation: operation.activation,
        evidenceRefs: operation.evidenceRefs,
        semantic: {
          kind: 'predicate',
          predicate: operation.predicate,
          args: operation.args,
          negated: operation.negated,
        },
        createdBy,
      })
      retracted.nodeIds.push(node.id)
      return retracted
    }
  }
}

function retractOperation(
  store: SpaceStore,
  spaceId: string,
  nodeId: string,
  reason: string | undefined,
  index: number,
  op: WorkingMemoryOperation['op'],
): WorkingMemoryOperationResult {
  const retraction = retractNode(store, spaceId, { nodeId, reason })
  return {
    index,
    op,
    nodeIds: [],
    retractedNodeIds: retraction.removedNodeIds,
  }
}

/**
 * Values that are almost certainly type placeholders rather than real
 * constants. Seen in a real run: a goal of finding(file=string, line=number)
 * can never be satisfied because it only matches the literal value "string".
 *
 * The unambiguous type names are a HARD error (the warning-only version
 * was ignored in three consecutive real runs, leaving the goal forever
 * unsatisfiable); fuzzier words stay warnings because they can be
 * legitimate domain constants.
 */
const STRICT_PLACEHOLDER_VALUES = new Set([
  'string',
  'number',
  'boolean',
  'integer',
  'int',
  'float',
  'double',
  'object',
  'array',
])

const SOFT_PLACEHOLDER_VALUES = new Set([
  'value',
  'values',
  'any',
  'anything',
  'placeholder',
  'unknown',
  'tbd',
  'todo',
  'n/a',
  'na',
  'xxx',
])

function placeholderArgs(
  operation: WorkingMemoryOperation & { op: 'declare_goal' | 'declare_hypothesis' },
  values: Set<string>,
): string[] {
  const hits: string[] = []
  for (const atom of operationAtoms(operation)) {
    for (const [key, value] of Object.entries(atom.args ?? {})) {
      if (typeof value !== 'string') continue
      if (value.startsWith('?')) continue
      if (values.has(value.toLowerCase())) hits.push(`${key}=${value}`)
    }
  }
  return hits
}

function assertNoStrictPlaceholders(operation: WorkingMemoryOperation): void {
  if (operation.op !== 'declare_goal' && operation.op !== 'declare_hypothesis') return
  const hits = placeholderArgs(operation, STRICT_PLACEHOLDER_VALUES)
  // Alternation literals (judgment=confirmed|refuted) are placeholders too:
  // the atom only matches that exact pipe-string, never a real value.
  for (const atom of operationAtoms(operation)) {
    for (const [key, value] of Object.entries(atom.args ?? {})) {
      if (typeof value === 'string' && !value.startsWith('?') && value.includes('|')) {
        hits.push(`${key}=${value}`)
      }
    }
  }
  if (hits.length === 0) return
  throw new Error(
    `${operation.op}: argument(s) ${hits.join(', ')} are type-name placeholders, not real values - ` +
      `this atom only matches those literal strings and can never be satisfied by real facts. ` +
      `Use a "?variable" (e.g. file=?f, line=?l) to mean "any value"; pattern goals/hypotheses ` +
      `are satisfied by any matching instance. The whole batch was rejected; nothing was applied.`,
  )
}

function detectPlaceholderConstants(
  operation: WorkingMemoryOperation & { op: 'declare_goal' | 'declare_hypothesis' },
): string | undefined {
  const suspicious = placeholderArgs(operation, SOFT_PLACEHOLDER_VALUES)
  if (suspicious.length === 0) return undefined
  return (
    `argument(s) ${suspicious.join(', ')} look like type placeholders, not real values - ` +
    `this atom only matches those literal strings, so it can never be satisfied by real facts. ` +
    `Use a "?variable" (e.g. file=?f) to mean "any value": pattern goals/hypotheses are satisfied by any matching instance.`
  )
}

/** Canonical identity of a ground atom: predicate + sorted args + sign. */
function canonicalAtomKey(atom: PredicateAtom): string {
  const args = atom.args ?? {}
  const keys = Object.keys(args).sort()
  const body = keys.map((key) => `${key}=${JSON.stringify(args[key])}`).join(',')
  return `${atom.negated ? '!' : ''}${atom.predicate}(${body})`
}

/** Canonical identity of a rule: literal order is semantically irrelevant. */
function canonicalRuleKey(when: PredicateAtom[], then: PredicateAtom[]): string {
  const literalKey = (atom: PredicateAtom): string =>
    `${atom.naf ? '~' : ''}${canonicalAtomKey(atom)}`
  return `${(when ?? []).map(literalKey).sort().join(' & ')} => ${(then ?? [])
    .map(literalKey)
    .sort()
    .join(' & ')}`
}

function collectRuleIndex(store: SpaceStore, spaceId: string): Map<string, string> {
  const index = new Map<string, string>()
  for (const node of logicallyUsableNodes(store.listNodes(spaceId))) {
    if (node.type !== 'axiom' || node.semantic?.kind !== 'axiom') continue
    index.set(canonicalRuleKey(node.semantic.when ?? [], node.semantic.then ?? []), node.id)
  }
  return index
}

function collectFactIndex(store: SpaceStore, spaceId: string): Map<string, string> {
  const index = new Map<string, string>()
  for (const node of logicallyUsableNodes(store.listNodes(spaceId))) {
    if (node.type !== 'fact' || node.semantic?.kind !== 'predicate') continue
    index.set(
      canonicalAtomKey({
        predicate: node.semantic.predicate ?? node.label,
        args: node.semantic.args,
        negated: node.semantic.negated,
      }),
      node.id,
    )
  }
  return index
}

/**
 * Hard derivation gate: record_result is the agent's "I am done" move,
 * so it must not rest on findings the rule engine does not stand behind.
 * An asserted (not derived) positive finding(...) is an unproven claim;
 * the nudge layer already warns about it, but prompts can be ignored —
 * this invariant cannot. Negated findings are exempt: they refute
 * hypotheses rather than claim results. record_conflict is never gated.
 */
export class DerivationGateError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DerivationGateError'
  }
}

export class ReservedLabelError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ReservedLabelError'
  }
}

/** result nodes with this label prefix are the memory-ledger audit channel (see memory-ledger.ts). */
export const RESERVED_RESULT_LABEL = 'memory_event:'

/**
 * Reserved-label gate — ledger integrity (P0). The memory-ledger audit channel (result nodes labelled
 * `memory_event:*`) is append-only and host-owned. A non-trusted (model) batch may neither WRITE such
 * a label (forge an event) nor RETRACT/REVISE an existing one (delete the audit). This is the load-
 * bearing point Codex flagged: the non-logical result channel only guarantees "never a proof premise";
 * UNFORGEABLE + UNDELETABLE is a separate property, enforced HERE (WM gate) and by the reader's
 * createdBy==='system' filter. Trusted host writes go through store.addNode (recordMemoryEvent), which
 * does not pass through this gate; this gate exists purely to stop the model op stream.
 */
function assertNoReservedLabelAbuse(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
  options: ApplyOptions,
): void {
  const trustedBatch = options.source === 'system' || options.createdBy === 'tool' || options.createdBy === 'system'
  if (trustedBatch) return
  let byId: Map<string, SpaceNode> | undefined
  for (const operation of operations) {
    if (
      (operation.op === 'record_result' || operation.op === 'record_conflict') &&
      operation.label?.startsWith(RESERVED_RESULT_LABEL)
    ) {
      throw new ReservedLabelError(
        `${operation.op} blocked: label "${operation.label}" uses the reserved memory-ledger prefix ` +
          `"${RESERVED_RESULT_LABEL}". Only a trusted (system/tool) batch may write a ledger event; the model op ` +
          `stream cannot forge one. Nothing was applied.`,
      )
    }
    if (operation.op === 'retract_node' || operation.op === 'revise_fact') {
      if (!byId) byId = new Map(store.listNodes(spaceId).map((node) => [node.id, node]))
      const target = byId.get(operation.nodeId)
      if (target?.type === 'result' && target.label?.startsWith(RESERVED_RESULT_LABEL)) {
        throw new ReservedLabelError(
          `${operation.op} blocked: "${operation.nodeId}" is a memory-ledger event (label "${target.label}"). ` +
            `The ledger is append-only; the model op stream cannot delete or alter audit events. Nothing was applied.`,
        )
      }
    }
  }
}

export class EvidenceReferenceError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'EvidenceReferenceError'
  }
}

/** Options for applying a working-memory batch. `source` distinguishes a
 *  model-driven batch from a trusted system/harness batch; `attestedPredicates`
 *  names predicates only the harness may write (machine-attested facts);
 *  `attestedDerivations` constrains which rules may PRODUCE a guarded conclusion. */
export type ApplyOptions = {
  format?: 'json' | 'text'
  source?: 'model' | 'system'
  /**
   * Provenance channel stamped on every node this batch creates (default
   * 'agent' = the model's own assertion). A trusted ingest channel passes
   * 'tool'/'system' so the fact carries — on the board itself — that it entered
   * through a real source, not the model's keyboard. premise-provenance reads
   * this to mark such ground facts attested regardless of any predicate
   * declaration: provenance is carried by HOW the fact entered, not by policy.
   */
  createdBy?: Creator
  /** Typed trust tier stamped on facts in this trusted batch (numeric→approximate,
   *  ml→uncertain, perceived, …). Honored only for a trusted createdBy; absent ⇒ the
   *  createdBy default. Set by trusted ingest, never reachable from a model op. */
  trustTier?: string
  attestedPredicates?: Iterable<string>
  /**
   * For a model-sourced add_axiom whose head matches `head` (predicate + every
   * specified arg), its body MUST include each atom in `requires` (predicate +
   * every specified arg value). Generalizes Codex's fixed-must-be-machine-backed
   * guard WITHOUT hardcoding domain names: e.g. a rule producing
   * finding(kind=fixed) must read edited(...) and test_result(status=pass), so a
   * model cannot define a weak fixed-rule that skips the machine evidence.
   */
  attestedDerivations?: { head: PredicateAtom; requires: PredicateAtom[] }[]
}

export class AttestedPredicateError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AttestedPredicateError'
  }
}

// GoalpostMovingError / ProtectedGoalpost / PROTECTED_GOALPOSTS + goalpostKey / amendmentMatches /
// formatGoalpostKey moved to goalpost.ts so the ACTION layer (deriveActionEffects / simulate) shares the
// same committed-baseline protection without an import cycle (closes action-laundering). Re-exported here
// so existing importers (goalpost-gate.test, etc.) keep resolving them from working-memory.
export { GoalpostMovingError }
export type { ProtectedGoalpost }

/**
 * Reject a model-sourced batch that tries to PRODUCE a machine-attested
 * predicate by any route: a direct assert_fact/revise_fact, a rule (add_axiom)
 * whose head is the attested predicate, or an action (define_action) whose
 * effect produces it. A model may only READ an attested predicate (in a rule
 * body / precondition), never put it on the board - that is the harness's job.
 * Closing the rule/action routes matters: deriving test_result(pass) from a
 * vacuous rule is the same fake-green claim as asserting it directly.
 * System/harness batches (and any batch with no attested set) pass unchanged.
 */
function assertAttestedPredicates(operations: WorkingMemoryOperation[], options: ApplyOptions): void {
  if (options.source !== 'model') return
  const attested = new Set(options.attestedPredicates ?? [])
  if (attested.size === 0) return
  const hits = (predicate: unknown): boolean => typeof predicate === 'string' && attested.has(predicate)
  const offenders = operations.flatMap((operation, index) => {
    let how: string | undefined
    if ((operation.op === 'assert_fact' || operation.op === 'revise_fact') && hits(operation.predicate)) {
      how = `${operation.op} ${operation.predicate}`
    } else if (operation.op === 'add_axiom' && (operation.then ?? []).some((atom) => hits(atom.predicate))) {
      how = `add_axiom head produces ${(operation.then ?? []).find((atom) => hits(atom.predicate))!.predicate}`
    } else if (operation.op === 'define_action' && (operation.effects ?? []).some((atom) => hits(atom.predicate))) {
      how = `define_action effect produces ${(operation.effects ?? []).find((atom) => hits(atom.predicate))!.predicate}`
    } else if (operation.op === 'derive_aggregate' && hits(operation.into?.predicate)) {
      // The aggregate produce-route (caught by Codex's parallel version): a
      // model could derive_aggregate INTO an attested predicate to manufacture it.
      how = `derive_aggregate into ${operation.into.predicate}`
    }
    return how ? [`op #${index} (${how})`] : []
  })
  if (offenders.length === 0) return
  throw new AttestedPredicateError(
    `blocked: ${offenders.length} operation(s) PRODUCE a MACHINE-ATTESTED predicate you may not write yourself — ${offenders.join(', ')}. ` +
      `Predicates {${[...attested].join(', ')}} stand for facts only the real runner/editor can vouch for; ` +
      `putting one on the board — by asserting it, deriving it from a rule, or producing it from an action — ` +
      `is the "I think it passed" claim the board refuses. You may READ them in a rule body, never produce them. ` +
      `Run the action/edit so the harness records it (a system-attested fact), then derive your conclusion from it. ` +
      `The whole batch was rejected; nothing was applied.`,
  )
}

/**
 * I12 no-goalpost-moving gate: once a baseline has been committed,
 * a model-sourced batch may not silently retract or revise it to make closure
 * easier. Trusted host/human batches can amend directly; a trusted
 * amendment_result lets a model batch perform the recorded revision only when
 * that same batch consumes the amendment. Without this one-shot requirement a
 * single human approval would become a standing pass to move the goalpost again.
 */
function assertNoGoalpostMoving(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
  options: ApplyOptions,
): void {
  const trustedBatch = options.source === 'system' || options.createdBy === 'tool' || options.createdBy === 'system'
  if (trustedBatch) return
  if (!operations.some((operation) => operation.op === 'retract_node' || operation.op === 'revise_fact')) return

  const nodes = logicallyUsableNodes(store.listNodes(spaceId))
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const trusted = (node: SpaceNode): boolean => node.createdBy === 'tool' || node.createdBy === 'system'
  const committed = new Map(PROTECTED_GOALPOSTS.map((spec) => [spec, new Set<string>()]))
  const amended = new Map(PROTECTED_GOALPOSTS.map((spec) => [spec, new Map<string, Set<string>>()]))
  for (const node of nodes) {
    if (node.type !== 'fact' || node.semantic?.kind !== 'predicate') continue
    if (!trusted(node)) continue
    for (const spec of PROTECTED_GOALPOSTS) {
      if (node.semantic.predicate === spec.marker) {
        const key = goalpostKey(node.semantic.args, spec.markerKey ?? spec.key)
        if (key) committed.get(spec)!.add(key)
      } else if (node.semantic.predicate === 'amendment_result' && amendmentMatches(node.semantic.args, spec)) {
        const key = goalpostKey(node.semantic.args, spec.amendmentKey ?? spec.key)
        if (key) {
          const byKey = amended.get(spec)!
          const ids = byKey.get(key) ?? new Set<string>()
          ids.add(node.id)
          byKey.set(key, ids)
        }
      }
    }
  }
  if (![...committed.values()].some((keys) => keys.size > 0)) return
  const consumedAmendmentNodeIds = new Set(
    operations.flatMap((operation) => {
      if (operation.op !== 'retract_node') return []
      const target = byId.get(operation.nodeId)
      if (!target || target.type !== 'fact' || target.semantic?.kind !== 'predicate') return []
      return trusted(target) && target.semantic.predicate === 'amendment_result' ? [target.id] : []
    }),
  )
  const consumesAmendment = (spec: ProtectedGoalpost, key: string): boolean => {
    const ids = amended.get(spec)!.get(key)
    if (!ids) return false
    return [...ids].some((id) => consumedAmendmentNodeIds.has(id))
  }

  const offenders: string[] = []
  operations.forEach((operation, index) => {
    if (operation.op !== 'retract_node' && operation.op !== 'revise_fact') return
    const target = byId.get(operation.nodeId)
    if (!target || target.type !== 'fact' || target.semantic?.kind !== 'predicate') return
    const predicate = target.semantic.predicate
    const marker = PROTECTED_GOALPOSTS.find((spec) => spec.marker === predicate)
    if (marker && trusted(target)) {
      offenders.push(`op #${index} ${operation.op} ${target.id} removes the ${marker.marker} marker`)
    }
    for (const spec of PROTECTED_GOALPOSTS) {
      if (predicate !== spec.target) continue
      const key = goalpostKey(target.semantic.args, spec.key)
      if (!key || !committed.get(spec)!.has(key) || consumesAmendment(spec, key)) continue
      const amendmentIds = amended.get(spec)!.get(key)
      const consumeHint =
        amendmentIds && amendmentIds.size > 0
          ? `; matching trusted amendment_result must be consumed in this batch with retract_node (${[...amendmentIds].join(', ')})`
          : ''
      offenders.push(`op #${index} ${operation.op} ${target.id} rewrites committed ${spec.target} baseline (${formatGoalpostKey(spec, key)})${consumeHint}`)
    }
  })

  if (offenders.length === 0) return
  throw new GoalpostMovingError(
    `goalpost-moving blocked: ${offenders.join('; ')}. ` +
      `A committed baseline is part of what the agent promised to satisfy; ` +
      `a model-sourced batch may not retract or revise it to make the task easier. ` +
      `Ask the human/host to amend it through a trusted system/tool batch or first record a trusted amendment_result with the matching key and kind, ` +
      `then consume that amendment_result in the same model batch with retract_node so the approval is one-shot. ` +
      `The whole batch was rejected; nothing was applied.`,
  )
}

// goalpostKey / amendmentMatches / formatGoalpostKey moved to goalpost.ts (imported above).

/** True iff `atom` matches `pattern`: same predicate and every arg SPECIFIED in
 *  the pattern is present on the atom with an equal (literal) value. */
function atomMatchesPattern(atom: PredicateAtom, pattern: PredicateAtom): boolean {
  if (atom.predicate !== pattern.predicate) return false
  for (const [key, value] of Object.entries(pattern.args ?? {})) {
    if ((atom.args ?? {})[key] !== value) return false
  }
  return true
}

/**
 * Reject a model-sourced rule that produces a guarded conclusion WITHOUT reading
 * the machine evidence it must rest on. Generalizes "fixed must be derived from
 * edited + test_result(pass)" to any {head, requires} contract, with no domain
 * names baked into the kernel (the contract is a parameter). A model may not
 * define a weak rule whose head laundes a conclusion past the evidence.
 *
 * KNOWN LIMITATION (Codex review P1-2, shared by both versions): this checks
 * that each required atom is PRESENT in the body (predicate + literal args), not
 * that it is variable-CO-BOUND with the head - so "read SOME edited + SOME
 * test_result(pass)" could in principle back a finding(kind=fixed) for a
 * different issue. Exploiting it still needs genuine harness-attested facts on
 * the board (they cannot be forged), so the not-lying命门 holds; tightening to
 * binding-aware contracts (require the body atom to share the head's key var) is
 * a follow-up.
 */
function assertAttestedDerivations(operations: WorkingMemoryOperation[], options: ApplyOptions): void {
  if (options.source !== 'model') return
  const contracts = options.attestedDerivations ?? []
  if (contracts.length === 0) return
  const offenders: string[] = []
  operations.forEach((operation, index) => {
    if (operation.op !== 'add_axiom') return
    for (const head of operation.then ?? []) {
      for (const contract of contracts) {
        if (!atomMatchesPattern(head, contract.head)) continue
        // Each required evidence atom must be PRESENT in the body...
        const evidence = contract.requires.map((req) => ({
          req,
          atom: (operation.when ?? []).find((w) => atomMatchesPattern(w, req)),
        }))
        const missing = evidence.filter((e) => !e.atom).map((e) => e.req)
        if (missing.length > 0) {
          offenders.push(
            `op #${index} produces ${formatAtom(head)} but its body omits required machine evidence: ${missing.map(formatAtom).join(', ')}`,
          )
          continue
        }
        // ...AND the head's variables must be CO-BOUND by that evidence, so the
        // proof is about the same entities the conclusion names (closes the
        // "read SOME edited + SOME pass to back a different fixed" gap).
        const headVars = Object.values(head.args ?? {}).filter(isVariable)
        const evidenceVars = new Set(
          evidence.flatMap((e) => Object.values(e.atom!.args ?? {}).filter(isVariable)),
        )
        const unbound = headVars.filter((v) => !evidenceVars.has(v))
        if (unbound.length > 0) {
          offenders.push(
            `op #${index} produces ${formatAtom(head)} but its head variable(s) ${unbound.join(', ')} are not bound by the required evidence — the evidence must be about the same entities the conclusion names`,
          )
        }
      }
    }
  })
  if (offenders.length === 0) return
  throw new AttestedPredicateError(
    `blocked: ${offenders.length} rule(s) produce a guarded conclusion without reading the machine evidence — ${offenders.join('; ')}. ` +
      `A conclusion like finding(kind=fixed) must be DERIVED from the harness-attested facts (e.g. edited + test_result(status=pass)); ` +
      `a rule that skips them launders a claim past the evidence. Add the required body atoms. The whole batch was rejected.`,
  )
}

/**
 * ② domain-pack hard enforcement — committed baseline UP at the RULE layer. A trusted
 * `committed_derivation(predicate=P)` marks a head whose DEFINITION the domain owns. A model-sourced
 * batch may not, UNLESS it consumes a matching one-shot amendment (2d, below):
 *   (2a) add_axiom a competing rule that derives P — it cannot redefine P weaker (e.g. a weak `safe`);
 *   (2b) retract/revise a TRUSTED rule that DEFINES P — it cannot delete the domain's definition to
 *        starve P (the dual of 2a: 2a blocks adding a weak rule, 2b blocks deleting the strong one);
 *   (—)  retract/revise the committed_derivation marker — lifting the lock is NEVER model-authorized
 *        (uncommitting goes through a trusted system/host batch), so it has no amendment exception.
 * 2d (definition amendment): a trusted `amendment_result(kind='derivation', predicate=P)` RETRACTED
 * (consumed) in the SAME model batch authorizes a one-shot change to P's definition — it unlocks both
 * the 2a add and the 2b retract/revise for that P (mirrors the committed-fact amendment in
 * assertNoGoalpostMoving). The domain/host can always extend the definition through a trusted batch.
 * Opt-in (inert with no marker); trusted-channel only (a model-forged marker or amendment does not
 * count). A model may still READ P in a rule body — only PRODUCING/deleting its definition is gated
 * (mirrors attestedDerivations' read-not-produce). The rule-layer dual of committed_phase; turns
 * "model redefines a concept too weak" / "model deletes the domain rule" from a semantic residual
 * into a hard block, with a controlled (amendment) path for authorized definition revision.
 */
function assertCommittedDerivation(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
  options: ApplyOptions,
): void {
  if (options.source !== 'model') return // a trusted host/domain batch owns the definition; it may write freely
  const nodes = logicallyUsableNodes(store.listNodes(spaceId))
  const trusted = (node: SpaceNode): boolean => node.createdBy === 'tool' || node.createdBy === 'system'
  const committedHeads = new Set<string>()
  const amendmentsByPred = new Map<string, Set<string>>() // predicate P -> trusted amendment_result(kind=derivation) node ids
  for (const node of nodes) {
    if (node.type !== 'fact' || node.semantic?.kind !== 'predicate' || !trusted(node)) continue
    if (node.semantic.predicate === 'committed_derivation') {
      const head = node.semantic.args?.predicate
      if (typeof head === 'string' && head.length > 0) committedHeads.add(head)
    } else if (node.semantic.predicate === 'amendment_result' && node.semantic.args?.kind === 'derivation') {
      const pred = node.semantic.args?.predicate
      if (typeof pred === 'string' && pred.length > 0) {
        const ids = amendmentsByPred.get(pred) ?? new Set<string>()
        ids.add(node.id)
        amendmentsByPred.set(pred, ids)
      }
    }
  }
  if (committedHeads.size === 0) return // opt-in: no committed definition → inert (zero regression)

  const byId = new Map(nodes.map((node) => [node.id, node]))
  // 2d one-shot: a trusted amendment_result(kind=derivation, predicate=P) RETRACTED in this batch
  // authorizes modifying P's definition (add a new rule deriving P, or retract/revise a P-defining rule).
  const consumedAmendmentIds = new Set(
    operations.flatMap((operation) => {
      if (operation.op !== 'retract_node') return []
      const target = byId.get(operation.nodeId)
      if (!target || target.type !== 'fact' || target.semantic?.kind !== 'predicate') return []
      return trusted(target) && target.semantic.predicate === 'amendment_result' && target.semantic.args?.kind === 'derivation'
        ? [target.id]
        : []
    }),
  )
  const amended = (predicate: string): boolean => {
    const ids = amendmentsByPred.get(predicate)
    if (!ids) return false
    return [...ids].some((id) => consumedAmendmentIds.has(id))
  }

  const offenders: string[] = []
  operations.forEach((operation, index) => {
    if (operation.op === 'add_axiom') {
      for (const head of operation.then ?? []) {
        if (committedHeads.has(head.predicate) && !amended(head.predicate)) {
          offenders.push(`op #${index} add_axiom derives committed head "${head.predicate}" — the domain owns its definition; a model may not add a competing rule (consume a trusted amendment_result(kind='derivation', predicate='${head.predicate}') in this batch to amend it)`)
        }
      }
    } else if (operation.op === 'retract_node' || operation.op === 'revise_fact') {
      const target = byId.get(operation.nodeId)
      if (!target) return
      // (—) lifting the lock itself is never model-authorized (mirrors the fact gate's unconditional marker block)
      if (target.type === 'fact' && target.semantic?.kind === 'predicate' && trusted(target) && target.semantic.predicate === 'committed_derivation') {
        offenders.push(`op #${index} ${operation.op} ${target.id} removes a committed_derivation marker — a model may not lift the domain's definition lock`)
        return
      }
      // (2b) a TRUSTED rule that DEFINES a committed head is the domain's definition — protected from
      // model retract/revise unless a matching derivation amendment is consumed in this batch.
      if (target.type === 'axiom' && target.semantic?.kind === 'axiom' && trusted(target)) {
        const protectedHeads = (target.semantic.then ?? [])
          .map((head) => head.predicate)
          .filter((predicate) => committedHeads.has(predicate) && !amended(predicate))
        if (protectedHeads.length > 0) {
          offenders.push(`op #${index} ${operation.op} ${target.id} deletes the trusted rule defining committed head "${protectedHeads.join(', ')}" — a model may not remove the domain's definition (consume a matching amendment_result(kind='derivation') in this batch to revise it)`)
        }
      }
    }
  })

  if (offenders.length === 0) return
  throw new GoalpostMovingError(
    `committed-derivation blocked: ${offenders.join('; ')}. ` +
      `A committed_derivation marks a head whose definition the domain owns — a model-sourced batch may not add a competing rule that derives it, delete the trusted rule that defines it, nor lift the lock. ` +
      `A model may READ the head in a rule body; to EXTEND or REVISE its definition, either amend it through a trusted system/host batch, or consume a trusted amendment_result(kind='derivation', predicate=P) in the same model batch (one-shot). The whole batch was rejected; nothing was applied.`,
  )
}

const GATE_LIST_LIMIT = 10

function assertDerivationGate(
  store: SpaceStore,
  spaceId: string,
  operations: WorkingMemoryOperation[],
): void {
  if (!operations.some((operation) => operation.op === 'record_result')) return

  // Findings directly retracted (or revised away) in this same batch are excused.
  const removedIds = new Set(
    operations.flatMap((operation) =>
      operation.op === 'retract_node' || operation.op === 'revise_fact'
        ? [operation.nodeId]
        : [],
    ),
  )

  const onBoard = logicallyUsableNodes(store.listNodes(spaceId))
    .filter(
      (node) =>
        node.type === 'fact' &&
        node.semantic?.kind === 'predicate' &&
        (node.semantic.predicate ?? node.label) === FINDING_PREDICATE &&
        node.semantic.negated !== true &&
        !isDerivedFactNode(node) &&
        !removedIds.has(node.id),
    )
    .map((node) => `${node.id}: ${node.label}`)

  const inBatch = operations.flatMap((operation, index) =>
    (operation.op === 'assert_fact' || operation.op === 'revise_fact') &&
    operation.predicate === FINDING_PREDICATE &&
    operation.negated !== true
      ? [`op #${index} (${operation.op}${operation.id ? ` ${operation.id}` : ''})`]
      : [],
  )

  if (onBoard.length === 0 && inBatch.length === 0) return

  const offenders = [
    ...(onBoard.length > 0
      ? [`on the board: ${truncateList(onBoard)}`]
      : []),
    ...(inBatch.length > 0
      ? [`asserted in this same batch: ${truncateList(inBatch)}`]
      : []),
  ].join('; ')

  throw new DerivationGateError(
    `record_result blocked: ${onBoard.length + inBatch.length} positive finding fact(s) are asserted, not derived — ${offenders}. ` +
      `A recorded result must rest on findings the rule closure stands behind. For each finding either ` +
      `(1) retract it, assert the primitive observation you actually verified (e.g. empty_catch(file=..., line=...)), ` +
      `and add_axiom a rule deriving finding(...) from that observation — the closure re-derives the finding; or ` +
      `(2) retract_node it if the observation does not hold. Then submit record_result again. ` +
      `The whole batch was rejected; nothing was applied.`,
  )
}

function assertResultEvidenceRefsResolvable(
  store: SpaceStore,
  spaceId: string,
  resultNodeIds: string[],
): void {
  if (resultNodeIds.length === 0) return
  const nodes = logicallyUsableNodes(store.listNodes(spaceId))
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const evidenceNodeIds = new Set(
    nodes
      .filter((node) => node.type !== 'result')
      .map((node) => node.id),
  )
  const offenders: string[] = []

  for (const resultId of resultNodeIds) {
    const result = byId.get(resultId)
    if (!result || result.type !== 'result') continue
    const evidenceRefs = Array.isArray(result.evidenceRefs) ? result.evidenceRefs : []
    const missing = evidenceRefs.filter((ref) => typeof ref !== 'string' || !evidenceNodeIds.has(ref))
    if (missing.length > 0) {
      offenders.push(`${result.id}: ${truncateList([...new Set(missing.map(String))])}`)
    }
  }

  if (offenders.length === 0) return
  throw new EvidenceReferenceError(
    `record_result blocked: evidenceRefs must resolve to current board node ids; unresolved reference(s): ${truncateList(offenders)}. ` +
      `Call get_logic_context and cite actual node ids from facts, findings, axioms, goals, actions, hypotheses, or conflicts ` +
      `(not prose labels, external file references, or invented ids). If the evidence is missing, assert or derive it first, ` +
      `then record_result again. The whole batch was rejected; nothing was applied.`,
  )
}

function truncateList(items: string[]): string {
  if (items.length <= GATE_LIST_LIMIT) return items.join(', ')
  return `${items.slice(0, GATE_LIST_LIMIT).join(', ')}, … and ${items.length - GATE_LIST_LIMIT} more`
}

/**
 * Readable errors for the two node-reference mistakes seen in real runs
 * (verification #8b): retract_node without a nodeId previously surfaced
 * as "Node undefined does not belong to space ..."; re-adding an existing
 * id crashed MID-BATCH (after earlier ops had applied), breaking the
 * validate-whole-batch-first promise.
 */
function assertValidNodeReference(operation: WorkingMemoryOperation): void {
  if (operation.op !== 'retract_node' && operation.op !== 'revise_fact') return
  if (typeof operation.nodeId === 'string' && operation.nodeId.length > 0) return
  throw new Error(
    `${operation.op} requires "nodeId": the id of the board entry to ` +
      `${operation.op === 'retract_node' ? 'remove' : 'replace'}, ` +
      `e.g. {"op":"${operation.op}","nodeId":"O3",...}. Find the id in the working memory listing.`,
  )
}

function assertFreshNodeId(
  operation: WorkingMemoryOperation,
  existingIds: Set<string>,
  removedInBatch: Set<string>,
  newIdsInBatch: Set<string>,
): void {
  const id = 'id' in operation ? operation.id : undefined
  if (typeof id !== 'string' || id.length === 0) return
  // derive_aggregate re-runs REPLACE the previous aggregate rule by design
  // (retract-then-expand), so an existing id is the normal idempotent path.
  if (operation.op === 'derive_aggregate') return
  const collidesOnBoard = existingIds.has(id) && !removedInBatch.has(id)
  const collidesInBatch = newIdsInBatch.has(id)
  if (collidesOnBoard || collidesInBatch) {
    throw new Error(
      `node id "${id}" already exists on the board (${operation.op}). ` +
        `If you meant to re-add the same entry: it is already there, no action needed. ` +
        `To change it, use retract_node or revise_fact; otherwise pick a fresh id. ` +
        `The whole batch was rejected; nothing was applied.`,
    )
  }
  newIdsInBatch.add(id)
}

/**
 * Teaching warning for rules that can NEVER fire as written: a positive,
 * non-builtin body literal whose predicate has no facts on the board
 * (post-closure, so batch-supplied and derived facts count) and no rule
 * head that could derive it.
 *
 * Why this exists (repair round 2026-06-13, sumTo): the model invented
 * compare(op=eq,...) where the builtin is eq(left,right). The rule was
 * VALID, so add_axiom succeeded silently; it just never fired. With no
 * error, no derivation and no explanation, the model built a wrong theory
 * ("rules must be re-added to trigger evaluation") and burned 6 of its 8
 * turns on retract/re-add loops. Fail visibly: name the predicate, name
 * the escape routes, make the builtin fix copy-pasteable.
 *
 * Precision-first: the flagship audit flow legitimately adds guard rules
 * BEFORE asserting the observations that feed them (#29 distillable
 * template) - lecturing that correct pattern would be recurring noise. So
 * the warning only fires when the rule signals IN-RULE COMPUTATION intent:
 * it already mixes in builtin literals, or the unsupplied predicate is
 * shaped like a builtin call (left/right/result args). Both held in the
 * sumTo wreck; neither holds for facts-arrive-later audit rules.
 */
function appendUnfirableRuleWarnings(
  operations: readonly WorkingMemoryOperation[],
  context: LogicContext,
  warnings: string[],
  skipIndices: ReadonlySet<number>,
): void {
  const supplied = new Set(context.facts.map((fact) => fact.atom.predicate))
  for (const axiom of context.axioms) {
    for (const head of axiom.then) supplied.add(head.predicate)
  }
  const looksBuiltinShaped = (literal: PredicateAtom): boolean => {
    const keys = new Set(Object.keys(literal.args ?? {}))
    return (keys.has('left') && keys.has('right')) || keys.has('result')
  }
  operations.forEach((operation, index) => {
    if (operation.op !== 'add_axiom') return
    if (skipIndices.has(index)) return
    // Self-recursive accumulator (#32 problem 8): head predicate re-appears
    // in the body NEXT TO arithmetic, with no comparison guard bounding it.
    // A monotonic closure cannot run an accumulator loop - every derived
    // value re-fires the rule, partial results combine combinatorially,
    // and once the closure explodes EVERY later apply on the space fails.
    // Warn at add time, before any seed fact arms the bomb. An lt/lte
    // guard is the sanctioned bounded form (what the non-convergence
    // teaching itself recommends), and value-free recursion (transitive
    // closure) is legitimate datalog - both stay silent.
    // Stepped recursion is EXEMPT (cross-review catch): when body and head
    // share the predicate but differ on some CONSTANT key (partial_sum
    // step=1 -> step=2), each rule maps one fixed stratum to the next, so a
    // finite rule set takes finitely many steps - that is the very pattern
    // this warning recommends. Dangerous recursion has no such grounding:
    // the recursive args are fresh variables fed by arithmetic.
    const isConstant = (v: unknown): boolean =>
      v !== undefined && !(typeof v === 'string' && v.startsWith('?'))
    const groundedStep = (body: PredicateAtom, head: PredicateAtom): boolean =>
      Object.keys(body.args ?? {}).some((key) => {
        const bv = (body.args ?? {})[key]
        const hv = (head.args ?? {})[key]
        return isConstant(bv) && isConstant(hv) && bv !== hv
      })
    const positiveBody = (operation.when ?? []).filter((l) => l.naf !== true)
    const recursiveLiteral = positiveBody.find((l) =>
      (operation.then ?? []).some((h) => h.predicate === l.predicate && !groundedStep(l, h)),
    )
    if (
      recursiveLiteral &&
      positiveBody.some((l) => isArithmeticBuiltin(l.predicate)) &&
      !positiveBody.some((l) => isComparisonBuiltin(l.predicate))
    ) {
      warnings.push(
        `op #${index} (add_axiom): rule "${operation.id ?? operation.label}" is SELF-RECURSIVE with ` +
          `arithmetic - its head "${recursiveLiteral.predicate}" also appears in its body next to an ` +
          `arithmetic built-in. A monotonic closure cannot run an accumulator loop: every derived value ` +
          `re-fires the rule, partial results multiply, and the closure will explode. Aggregate with ` +
          `ONE chain rule instead (add(?c1,?c2,?s1) then add(?s1,?c3,?s2) ... in a single "when"), or ` +
          `stepped predicates (partial_sum(step=1) -> partial_sum(step=2) -> ...), or bound the ` +
          `recursion with a comparison guard (e.g. {"predicate":"lt","args":{"left":"?v","right":100}}).`,
      )
    }
    const mixesBuiltins = (operation.when ?? []).some((l) => isBuiltinPredicate(l.predicate))
    for (const literal of operation.when ?? []) {
      if (literal.naf === true) continue
      if (isBuiltinPredicate(literal.predicate)) continue
      if (supplied.has(literal.predicate)) continue
      if (!mixesBuiltins && !looksBuiltinShaped(literal)) continue
      warnings.push(
        `op #${index} (add_axiom): rule "${operation.id ?? operation.label}" can never fire as written - ` +
          `nothing supplies "${literal.predicate}(...)": no facts assert it and no rule derives it. ` +
          `Either assert ${literal.predicate}(...) facts, add a rule whose "then" produces it, or - if ` +
          `you meant a comparison/arithmetic - use the built-ins directly in the rule body: ` +
          `${[...COMPARISON_BUILTINS].join('/')} or ${[...ARITHMETIC_BUILTINS].join('/')} ` +
          `(e.g. {"predicate":"eq","args":{"left":"?a","right":"?b"}}).`,
      )
    }
  })
}

/**
 * derive_aggregate v1 (#32 root fix): the engine writes the chain rule the
 * model would otherwise hand-roll. Models naturally reach for a recursive
 * accumulator (the one shape a monotonic closure cannot run); the recipe
 * expands to the sanctioned SINGLE chain rule instead - deterministic fact
 * order, every source fact pinned by its constant args, exact-or-fail
 * arithmetic inherited from the add builtin, evidence flowing through the
 * closure like any hand-written rule.
 *
 * Known v1 limit, taught in the rule label: the expansion pins the facts
 * present NOW. After adding more source facts, re-run derive_aggregate
 * with the same id - the old rule (and its stale total) is retracted and
 * replaced.
 */
function expandAggregate(
  store: SpaceStore,
  spaceId: string,
  operation: Extract<WorkingMemoryOperation, { op: 'derive_aggregate' }>,
): Array<{ ruleId: string; when: PredicateAtom[]; then: PredicateAtom[]; label: string }> {
  const kind = operation.kind ?? 'sum'
  if (kind !== 'sum' && kind !== 'count' && kind !== 'min' && kind !== 'max' && kind !== 'avg') {
    throw new Error(
      `derive_aggregate: unknown kind "${String(kind)}" - supports "sum"/"min"/"max"/"avg" (need source.valueArg) and "count".`,
    )
  }
  const sourcePredicate = operation.source?.predicate
  const intoPredicate = operation.into?.predicate
  const intoArg = operation.into?.valueArg
  if (typeof sourcePredicate !== 'string' || sourcePredicate.length === 0 ||
      typeof intoPredicate !== 'string' || intoPredicate.length === 0 ||
      typeof intoArg !== 'string' || intoArg.length === 0) {
    throw new Error(
      'derive_aggregate needs source.predicate and into.{predicate,valueArg}, e.g. ' +
        '{"op":"derive_aggregate","id":"agg_total","source":{"predicate":"cost","valueArg":"total"},' +
        '"into":{"predicate":"grand_total","valueArg":"value"}}',
    )
  }
  if (intoPredicate === FINDING_PREDICATE) {
    throw new Error(
      `derive_aggregate: aggregating straight into ${FINDING_PREDICATE}(...) is not allowed - ` +
        `aggregate into a neutral predicate (e.g. grand_total) and derive the ${FINDING_PREDICATE} ` +
        `from it with a rule, so findings keep real filtering power.`,
    )
  }
  const valueArg = operation.source.valueArg
  const needsValueArg = kind === 'sum' || kind === 'min' || kind === 'max' || kind === 'avg'
  if (needsValueArg && (typeof valueArg !== 'string' || valueArg.length === 0)) {
    throw new Error(
      `derive_aggregate: kind "${kind}" needs source.valueArg - the numeric arg to ` +
        `${kind === 'sum' ? 'add up' : `take the ${kind} of`}.`,
    )
  }
  const context = getLogicContext(store, spaceId)
  // ② Refuse to aggregate a source that has a functional conflict (same declared key, different
  // value - e.g. a bare-asserted cost contradicting the board-derived one). Summing it would
  // double-count the key or fold in a wrong value (arith p10). exact-or-fail: a polluted source
  // must be resolved (retract the wrong fact), not silently summed. Reuses the functionalConflicts
  // logic-context already derived; inert without a declared functional_dependency (zero regression).
  const sourceConflicts = context.functionalConflicts.filter((c) => c.predicate === sourcePredicate)
  if (sourceConflicts.length > 0) {
    const detail = sourceConflicts
      .map((c) => `${c.predicate}(${Object.entries(c.key).map(([k, v]) => `${k}=${v}`).join(', ')}): [${c.factIds.join(', ')}]`)
      .join('; ')
    throw new Error(
      `derive_aggregate: the source "${sourcePredicate}" has a functional conflict - ${detail}. ` +
        `Summing it would double-count the key or fold in a wrong value. Retract the wrong fact ` +
        `(usually a bare assertion that should have been left to a rule), then re-run derive_aggregate.`,
    )
  }
  const candidates = context.facts.filter(
    (fact) => fact.atom.predicate === sourcePredicate && fact.atom.negated !== true,
  )
  if (candidates.length === 0) {
    const present = [...new Set(context.facts.map((fact) => fact.atom.predicate))].slice(0, 8)
    throw new Error(
      `derive_aggregate: no active "${sourcePredicate}" facts to aggregate. The aggregate expands ` +
        `AFTER the batch's first closure, so a "${sourcePredicate}" derived by an add_axiom in THIS ` +
        `SAME batch IS visible here — if it is still missing, nothing actually produces ` +
        `"${sourcePredicate}": either no rule derives it (and no fact asserts it), or the deriving ` +
        `rule's body is not satisfied (check its when-literals and that the base facts it needs are on ` +
        `the board). Fact predicates currently on the board: ${present.join(', ') || '(none)'}.`,
    )
  }
  const where = operation.where
  // Normalize the filter value through the same canonical round-trip the
  // engine applies to fact args ("5"->5), so equals:"5" matches a stored 5.
  const wherePin = where == null ? undefined : normalizeScalar(where.equals)
  if (where != null) {
    if (typeof where.arg !== 'string' || where.arg.length === 0) {
      throw new Error(
        'derive_aggregate: where.arg must be a non-empty string naming a source-predicate argument, ' +
          'e.g. {"where":{"arg":"region","equals":"east"}}.',
      )
    }
    if (!candidates.some((fact) => Object.prototype.hasOwnProperty.call(fact.atom.args ?? {}, where.arg))) {
      throw new Error(
        `derive_aggregate: where.arg "${where.arg}" is not present on any active "${sourcePredicate}" ` +
          `fact - check the argument name. Args seen: ${[
            ...new Set(candidates.flatMap((fact) => Object.keys(fact.atom.args ?? {}))),
          ].join(', ') || '(none)'}.`,
      )
    }
  }
  const facts =
    where == null
      ? candidates
      : candidates.filter((fact) => (fact.atom.args ?? {})[where.arg] === wherePin)
  if (facts.length === 0) {
    const present = [...new Set(context.facts.map((fact) => fact.atom.predicate))].slice(0, 8)
    throw new Error(
      `derive_aggregate: no active "${sourcePredicate}" facts where ${where!.arg}=` +
        `${JSON.stringify(where!.equals)} to aggregate - widen or drop the where filter. ` +
        `Fact predicates currently on the board: ${present.join(', ') || '(none)'}.`,
    )
  }
  // group_by: bucket the (post-where) facts by the distinct values of an arg
  // and emit one rule per bucket. Absent = one global aggregate (v1/v2).
  const groupBy = operation.group_by
  if (groupBy != null) {
    if (typeof groupBy !== 'string' || groupBy.length === 0) {
      throw new Error(
        'derive_aggregate: group_by must be a non-empty string naming a source-predicate argument, ' +
          'e.g. {"group_by":"region"}.',
      )
    }
    const missing = facts.filter((fact) => !Object.prototype.hasOwnProperty.call(fact.atom.args ?? {}, groupBy))
    if (missing.length === facts.length) {
      throw new Error(
        `derive_aggregate: group_by "${groupBy}" is not present on any active "${sourcePredicate}" ` +
          `fact${where != null ? ' (after the where filter)' : ''} - check the argument name. Args seen: ${[
            ...new Set(facts.flatMap((fact) => Object.keys(fact.atom.args ?? {}))),
          ].join(', ') || '(none)'}.`,
      )
    }
    // Partial coverage = a silent undercount: some facts WOULD be grouped, but
    // the ones lacking the key would vanish from every bucket (fail visibly,
    // Codex review). Either every fact carries the group key, or none.
    if (missing.length > 0) {
      throw new Error(
        `derive_aggregate: group_by "${groupBy}" is missing on ${missing.length} of ${facts.length} ` +
          `active "${sourcePredicate}" fact(s) (e.g. "${missing[0]!.nodeId}") - those would be dropped ` +
          `from every bucket, silently undercounting. Give every ${sourcePredicate}(...) fact a ` +
          `"${groupBy}" arg, or add a where filter so only keyed facts are aggregated.`,
      )
    }
  }
  const baseRuleId = operation.id ?? `agg_${intoPredicate}`
  // min/max fold through the matching binary built-in; sum/count fold through
  // add (count totals 1s). All value-bearing kinds GROUND the valueArg as a
  // constant to avoid the cross-product.
  const foldBuiltin = kind === 'min' ? 'min' : kind === 'max' ? 'max' : 'add' // avg sums, then divides by count below

  const buildRule = (
    bucketFacts: typeof facts,
    groupValue: SemanticArgs[string] | undefined,
    ruleId: string,
  ): { ruleId: string; when: PredicateAtom[]; then: PredicateAtom[]; label: string } => {
    const sorted = [...bucketFacts].sort((a, b) =>
      canonicalAtomKey(a.atom) < canonicalAtomKey(b.atom) ? -1 : 1,
    )
    const when: PredicateAtom[] = []
    const terms: Array<string | number> = []
    sorted.forEach((fact) => {
      const args: SemanticArgs = {}
      for (const [key, value] of Object.entries(fact.atom.args ?? {})) {
        if (needsValueArg && key === valueArg) continue
        args[key] = value
      }
      if (where != null) args[where.arg] = wherePin as SemanticArgs[string]
      // Pin the group constant on each source literal so this rule binds ONLY
      // this bucket's facts (same grounding logic as the value below).
      if (groupBy != null) args[groupBy] = groupValue as SemanticArgs[string]
      if (needsValueArg) {
        const value = (fact.atom.args ?? {})[valueArg!]
        if (typeof value !== 'number') {
          throw new Error(
            `derive_aggregate: fact "${fact.nodeId}" carries no numeric "${valueArg}" ` +
              `(got ${JSON.stringify(value)}). Every ${sourcePredicate}(...) fact must have it - ` +
              `fix or retract that fact, or aggregate a different valueArg.`,
          )
        }
        // GROUND the value as a constant so this literal binds EXACTLY this
        // fact (var-izing it cross-products when two facts share every other
        // arg). The constant flows into the fold; the ground literal still
        // requires the fact, so the evidence chain holds.
        args[valueArg!] = value
        terms.push(value)
      } else {
        terms.push(1)
      }
      when.push({ predicate: sourcePredicate, args })
    })
    let accumulator: string | number = terms[0]!
    terms.slice(1).forEach((term, j) => {
      const out = `?s${j + 1}`
      when.push({ predicate: foldBuiltin, args: { left: accumulator, right: term, result: out } })
      accumulator = out
    })
    // avg = sum / count: append one div by the (constant) bucket size. div is
    // IEEE (declared): an integer mean that does not divide evenly is a float,
    // not an exact-or-fail failure - that is the documented avg contract.
    if (kind === 'avg') {
      const out = '?avg'
      when.push({ predicate: 'div', args: { left: accumulator, right: sorted.length, result: out } })
      accumulator = out
    }
    // Carry the group value into the head so each bucket's result is distinct.
    const headArgs: SemanticArgs = { [intoArg]: accumulator }
    if (groupBy != null) headArgs[groupBy] = groupValue as SemanticArgs[string]
    const then: PredicateAtom[] = [{ predicate: intoPredicate, args: headArgs }]
    const whereLabel = where != null ? ` where ${where.arg}=${JSON.stringify(where.equals)}` : ''
    const groupLabel = groupBy != null ? ` grouped by ${groupBy}=${JSON.stringify(groupValue)}` : ''
    const label =
      `${kind} of ${sourcePredicate}${needsValueArg ? `.${valueArg}` : ''}${whereLabel}${groupLabel} over ${sorted.length} facts ` +
      `-> ${intoPredicate}.${intoArg} (engine-expanded chain; re-run derive_aggregate after ` +
      `${sourcePredicate} facts change - this rule pins today's ${sorted.length})`
    return { ruleId, when, then, label }
  }

  if (groupBy == null) {
    return [buildRule(facts, undefined, baseRuleId)]
  }

  // Bucket by canonical group value; groups sorted by canonical key,
  // facts within a bucket sorted in buildRule (deterministic).
  const buckets = new Map<string, { value: SemanticArgs[string]; facts: typeof facts }>()
  for (const fact of facts) {
    if (!Object.prototype.hasOwnProperty.call(fact.atom.args ?? {}, groupBy)) continue
    const value = (fact.atom.args ?? {})[groupBy] as SemanticArgs[string]
    const key = JSON.stringify(value)
    const bucket = buckets.get(key)
    if (bucket) bucket.facts.push(fact)
    else buckets.set(key, { value, facts: [fact] })
  }
  return [...buckets.keys()].sort().map((key) => {
    const { value, facts: bucketFacts } = buckets.get(key)!
    const suffix = key.replace(/[^A-Za-z0-9_.-]/g, '_')
    return buildRule(bucketFacts, value, `${baseRuleId}__g_${suffix}`)
  })
}

function assertNoBuiltinAssertion(operation: WorkingMemoryOperation): void {
  if (
    (operation.op === 'assert_fact' ||
      operation.op === 'revise_fact' ||
      operation.op === 'declare_hypothesis') &&
    isBuiltinPredicate(operation.predicate)
  ) {
    throw new Error(
      `"${operation.predicate}" is a reserved built-in comparison predicate; it can only appear in rule bodies`,
    )
  }
}

function operationAtoms(operation: WorkingMemoryOperation): PredicateAtom[] {
  switch (operation.op) {
    case 'derive_aggregate':
      // The synthesized chain rule is validated when it is applied; the op
      // itself carries no literal atoms to check.
      return []
    case 'declare_goal':
      return operation.desired
    case 'assert_fact':
    case 'declare_hypothesis':
    case 'revise_fact':
      return [{ predicate: operation.predicate, args: operation.args, negated: operation.negated }]
    case 'add_axiom':
      return [...operation.when, ...operation.then]
    case 'define_action':
      return [...(operation.preconditions ?? []), ...(operation.effects ?? [])]
    case 'record_result':
    case 'record_conflict':
    case 'retract_node':
      return []
  }
}

/**
 * Tolerate the most common shape mistake models make:
 * {"declare_goal": {...}} instead of {"op": "declare_goal", ...}.
 */
function normalizeOperationShape(operation: WorkingMemoryOperation): WorkingMemoryOperation {
  const raw = operation as unknown as Record<string, unknown>
  if (typeof raw.op === 'string') return operation
  const keys = Object.keys(raw).filter((key) => key !== 'note')
  const [key] = keys
  if (keys.length === 1 && key && KNOWN_OPERATIONS.has(key) && typeof raw[key] === 'object' && raw[key] !== null) {
    return { op: key, ...(raw[key] as Record<string, unknown>) } as WorkingMemoryOperation
  }
  return operation
}

/**
 * Normalize CANONICAL numeric strings ("5", "-3", "2.5") to numbers at the
 * boundary, so every layer agrees on scalar identity. Before this, the
 * layers disagreed: atomKey treated amount(mol=5) and amount(mol="5") as
 * the same fact, the matcher treated the bindings as different, arithmetic
 * coerced strings while comparisons required numbers — so a model writing
 * {"mol":"5"} (a high-frequency JSON accident) got silently-failing gte
 * guards and "action not applicable" with no clue.
 *
 * Only round-trip-canonical strings convert (String(Number(v)) === v):
 * "007", "5.0", "1e3", " 5" keep their identity as strings. Variables
 * ("?x") and non-finite values are never touched.
 */
function normalizeScalar<T>(value: T): T | number {
  if (typeof value !== 'string' || value.length === 0 || value.startsWith('?')) return value
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return value
  return String(numeric) === value ? numeric : value
}

function normalizeAtomScalars(atom: PredicateAtom): PredicateAtom {
  if (!atom || typeof atom !== 'object' || !atom.args || typeof atom.args !== 'object') return atom
  const args: SemanticArgs = {}
  for (const [key, value] of Object.entries(atom.args)) {
    args[key] = normalizeScalar(value)
  }
  return { ...atom, args }
}

function normalizeAtomList(atoms: PredicateAtom[] | undefined): PredicateAtom[] | undefined {
  // Malformed shapes pass through untouched so assertValidAtoms can still
  // produce its teaching error instead of a crash here.
  if (!Array.isArray(atoms)) return atoms
  return atoms.map(normalizeAtomScalars)
}

function normalizeOperationScalars(operation: WorkingMemoryOperation): WorkingMemoryOperation {
  switch (operation.op) {
    case 'assert_fact':
    case 'revise_fact':
    case 'declare_hypothesis': {
      if (!operation.args || typeof operation.args !== 'object') return operation
      const args: SemanticArgs = {}
      for (const [key, value] of Object.entries(operation.args)) {
        args[key] = normalizeScalar(value)
      }
      return { ...operation, args }
    }
    case 'declare_goal':
      return { ...operation, desired: normalizeAtomList(operation.desired) as PredicateAtom[] }
    case 'add_axiom':
      return {
        ...operation,
        when: normalizeAtomList(operation.when) as PredicateAtom[],
        then: normalizeAtomList(operation.then) as PredicateAtom[],
      }
    case 'define_action':
      return {
        ...operation,
        preconditions: normalizeAtomList(operation.preconditions),
        effects: normalizeAtomList(operation.effects),
      }
    default:
      return operation
  }
}

/** Provenance markers the board assigns; user ops may not claim them.
 * (External review P0: assert_fact with summary "Rule-derived fact: ..."
 * passed isDerivedFactNode and sailed through the record_result gate.) */
const RESERVED_SUMMARY_PREFIXES = ['Rule-derived fact:', 'Derived fact:', 'Action-effect fact:']

function assertNoForgedProvenance(operation: WorkingMemoryOperation): void {
  const id = (operation as { id?: unknown }).id
  if (typeof id === 'string' && id.startsWith('derived:')) {
    throw new Error(
      `${operation.op}: "derived:" is a reserved id prefix - closure provenance is assigned by ` +
        `the board, never claimed. Use a plain id (e.g. "F1"); if you want the fact DERIVED, ` +
        `add a rule and let the closure produce it.`,
    )
  }
  const summary = (operation as { summary?: unknown }).summary
  if (typeof summary === 'string') {
    const forged = RESERVED_SUMMARY_PREFIXES.find((prefix) => summary.startsWith(prefix))
    if (forged !== undefined) {
      throw new Error(
        `${operation.op}: summary may not start with "${forged}" - that is a reserved provenance ` +
          `marker the board assigns. Describe the node in your own words; derivation status comes ` +
          `from the closure, not from labels.`,
      )
    }
  }
}

const ATOM_SHAPE_HINT =
  'each atom must be an object like {"predicate":"at","args":{"object":"car","location":"home"}}'

function assertValidAtoms(operation: WorkingMemoryOperation): void {
  const check = (atom: unknown, field: string): void => {
    if (
      typeof atom !== 'object' ||
      atom === null ||
      typeof (atom as PredicateAtom).predicate !== 'string' ||
      (atom as PredicateAtom).predicate.length === 0
    ) {
      throw new Error(
        `invalid atom in ${operation.op}.${field}: got ${JSON.stringify(atom)}; ${ATOM_SHAPE_HINT}`,
      )
    }
  }

  switch (operation.op) {
    case 'assert_fact':
    case 'declare_hypothesis':
    case 'revise_fact': {
      check(
        { predicate: (operation as { predicate?: unknown }).predicate, args: operation.args },
        'predicate',
      )
      return
    }
    case 'declare_goal': {
      if (!Array.isArray(operation.desired) || operation.desired.length === 0) {
        throw new Error(`declare_goal.desired must be a non-empty atom array; ${ATOM_SHAPE_HINT}`)
      }
      operation.desired.forEach((atom) => check(atom, 'desired'))
      return
    }
    case 'add_axiom': {
      ;(operation.when ?? []).forEach((atom) => check(atom, 'when'))
      ;(operation.then ?? []).forEach((atom) => check(atom, 'then'))
      return
    }
    case 'define_action': {
      ;(operation.preconditions ?? []).forEach((atom) => check(atom, 'preconditions'))
      ;(operation.effects ?? []).forEach((atom) => check(atom, 'effects'))
      return
    }
    default:
      return
  }
}

const KNOWN_OPERATIONS = new Set([
  'declare_goal',
  'assert_fact',
  'declare_hypothesis',
  'derive_aggregate',
  'add_axiom',
  'define_action',
  'record_result',
  'record_conflict',
  'retract_node',
  'revise_fact',
])

/** Tell-tale fields → the op the model most likely meant when it left "op" out. */
function inferLikelyOp(raw: Record<string, unknown>): string | undefined {
  if (Array.isArray(raw.desired)) return 'declare_goal'
  if (Array.isArray(raw.when) && Array.isArray(raw.then)) return 'add_axiom'
  if (Array.isArray(raw.effects)) return 'define_action'
  if (raw.source !== undefined && raw.into !== undefined) return 'derive_aggregate'
  if (typeof raw.nodeId === 'string') return typeof raw.predicate === 'string' ? 'revise_fact' : 'retract_node'
  if (Array.isArray(raw.evidenceRefs)) return 'record_result'
  if (typeof raw.predicate === 'string') return 'assert_fact'
  return undefined
}

function exampleForOp(op: string): string {
  switch (op) {
    case 'declare_goal':
      return '{"op":"declare_goal","id":"G1","desired":[{"predicate":"cost","args":{"item":"x"}}]}'
    case 'assert_fact':
      return '{"op":"assert_fact","id":"F1","predicate":"line","args":{"item":"x","unit":3,"qty":2}}'
    case 'add_axiom':
      return '{"op":"add_axiom","id":"R1","when":[{"predicate":"line","args":{"unit":"?u","qty":"?q"}},{"predicate":"mul","args":{"left":"?u","right":"?q","result":"?t"}}],"then":[{"predicate":"cost","args":{"total":"?t"}}]}'
    case 'derive_aggregate':
      return '{"op":"derive_aggregate","id":"AGG","kind":"sum","source":{"predicate":"cost","valueArg":"total"},"into":{"predicate":"grand_total","valueArg":"value"}}'
    case 'record_result':
      return '{"op":"record_result","id":"R1","label":"done","evidenceRefs":["F1"]}'
    case 'revise_fact':
      return '{"op":"revise_fact","nodeId":"F1","predicate":"line","args":{"item":"x"}}'
    case 'retract_node':
      return '{"op":"retract_node","nodeId":"F1"}'
    default:
      return '{"op":"assert_fact","id":"F1","predicate":"at","args":{"object":"car"}}'
  }
}

function assertKnownOperation(operation: WorkingMemoryOperation): void {
  if (KNOWN_OPERATIONS.has(operation.op)) return
  const valid = [...KNOWN_OPERATIONS].join(', ')
  const raw = operation as unknown as Record<string, unknown>

  // op MISSING entirely is a high-frequency weak-model slip (qwen arith p1): the model wrote the ATOM
  // fields (predicate/args, or desired, or when+then) directly on the operation and left out the "op"
  // discriminator — often using predicate:"line" AS IF it were the op type. The opaque
  // `unknown op "undefined"` didn't connect to that mistake (10 turns, no self-correction), so name it:
  // infer the intended op from the tell-tale fields and show the exact corrected shape. Still throws —
  // the batch is rejected (atomic), never guessed into existence (no masking of the protocol error).
  if (typeof raw.op !== 'string') {
    const guess = inferLikelyOp(raw)
    if (guess !== undefined) {
      const predicateHint =
        typeof raw.predicate === 'string' && (guess === 'assert_fact' || guess === 'revise_fact')
          ? `You wrote predicate:"${String(raw.predicate)}", which names the ATOM, not the operation. `
          : ''
      throw new Error(
        `operation is missing the required "op" field. ${predicateHint}` +
          `From its shape you most likely meant op:"${guess}", e.g. ${exampleForOp(guess)}. ` +
          `Every operation needs an "op" (one of: ${valid}).`,
      )
    }
    throw new Error(
      `operation is missing the required "op" field (one of: ${valid}), ` +
        'e.g. {"op":"assert_fact","id":"F1","predicate":"at","args":{"object":"car"}}.',
    )
  }

  throw new Error(
    `unknown op "${operation.op}"; valid ops: ${valid}. ` +
      'Each operation needs an "op" key, e.g. {"op":"assert_fact","id":"F1","predicate":"at","args":{"object":"car"}}',
  )
}

function operationResult(
  index: number,
  op: WorkingMemoryOperation['op'],
  nodeIds: string[],
): WorkingMemoryOperationResult {
  return {
    index,
    op,
    nodeIds,
    retractedNodeIds: [],
  }
}
