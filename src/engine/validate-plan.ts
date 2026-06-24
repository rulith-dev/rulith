import type { PredicateAtom, SpaceNode, CreateNodeInput } from '../model/types.js'
import type { SpaceStore } from '../storage/space-store.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { simulateActionEffects } from './simulate.js'
import { deriveActionEffects } from './semantic-derivation.js'
import { getLogicContext } from './logic-context.js'

/**
 * validate_plan — dry-run a WHOLE action sequence on a throwaway clone and
 * report where it breaks, without touching the real board.
 *
 * Planning is "model proposes a sequence, board grounds it." simulate checks
 * ONE action against NOW; a plan is a sequence, and the precondition of step
 * k only holds after steps 1..k-1 have run. validatePlan walks the sequence
 * on a cloned board: simulate step k against the evolving clone, and if
 * applicable, apply it to advance the clone so step k+1 sees the right state.
 * It stops at the first step whose preconditions fail and names it — the plan
 * is coherent iff every step applied AND the goals are reached at the end.
 *
 * The real board is never mutated (the clone is discarded), so a model can
 * validate a candidate plan before committing a single step.
 */
export type PlanStepResult = {
  index: number
  actionNodeId: string
  applicable: boolean
  /** First failing precondition (partial binding substituted), when blocked. */
  failedPrecondition?: PredicateAtom
  /** All unsatisfied preconditions, when blocked. */
  unsatisfied: PredicateAtom[]
  /** Set when the action id is not a usable action on the board. */
  error?: string
}

export type PlanValidation = {
  /** Every step applied in order AND, if goals exist, all of them end satisfied. */
  ok: boolean
  steps: PlanStepResult[]
  /** Index of the first step that could not run (undefined = none failed). */
  firstFailureIndex?: number
  /** Goal node ids satisfied on the clone after the sequence ran as far as it could. */
  satisfiedGoalIds: string[]
  /** Goal node ids that exist but are NOT satisfied at the end. */
  unmetGoalIds: string[]
  /**
   * Minimal number of LEADING steps after which ALL goals are first satisfied —
   * i.e. the shortest prefix of the plan that already reaches the goal. The model
   * can drop everything after it. undefined when there are no goals, or when the
   * goals are never all satisfied during the run. 0 means the goals held before
   * any step ran (the plan is entirely redundant).
   */
  shortestPrefixLength?: number
  /**
   * Step indices that ran AFTER the goal was already reached (index >=
   * shortestPrefixLength). These are the prune candidates. Empty unless a
   * shorter prefix sufficed.
   */
  redundantStepIndices: number[]
}

function cloneInput(node: SpaceNode): CreateNodeInput & { id: string } {
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
  }
}

/** A throwaway MemorySpaceStore carrying a copy of spaceId's nodes. */
export function cloneSpace(store: SpaceStore, spaceId: string): { clone: MemorySpaceStore; spaceId: string } {
  const source = store.getSpace(spaceId)
  const clone = new MemorySpaceStore()
  clone.createSpace({ id: source.id, title: source.title, summary: source.summary, scopes: source.scopes })
  for (const node of store.listNodes(spaceId)) clone.addNode(spaceId, cloneInput(node))
  return { clone, spaceId }
}

function goalIds(store: SpaceStore, spaceId: string): string[] {
  return store
    .listNodes(spaceId)
    .filter((node) => node.type === 'goal' && node.semantic?.kind === 'goal')
    .map((node) => node.id)
}

/** True iff every goal on the clone is satisfied right now (closure-backed). */
function allGoalsSatisfied(store: SpaceStore, spaceId: string, totalGoals: number): boolean {
  if (totalGoals === 0) return false // no goals => nothing to "reach"; prefix stays undefined
  const ctx = getLogicContext(store, spaceId)
  return ctx.goals.length === totalGoals && ctx.goals.every((g) => g.satisfied)
}

export function validatePlan(
  store: SpaceStore,
  spaceId: string,
  actionNodeIds: string[],
): PlanValidation {
  const { clone } = cloneSpace(store, spaceId)
  const steps: PlanStepResult[] = []
  let firstFailureIndex: number | undefined

  // How many goals exist on the (frozen) board; goals are asserted up front, so
  // their count does not change as actions run — only their satisfaction does.
  const totalGoals = goalIds(clone, spaceId).length
  let shortestPrefixLength: number | undefined
  // 0-step prefix: were the goals already satisfied before any action ran?
  if (allGoalsSatisfied(clone, spaceId, totalGoals)) shortestPrefixLength = 0

  for (let index = 0; index < actionNodeIds.length; index += 1) {
    const actionNodeId = actionNodeIds[index]!
    let sim
    try {
      sim = simulateActionEffects(clone, spaceId, actionNodeId)
    } catch (error) {
      steps.push({ index, actionNodeId, applicable: false, unsatisfied: [], error: String(error).slice(0, 160) })
      firstFailureIndex = index
      break // a missing/invalid action breaks the rest of the sequence
    }
    if (!sim.applicable) {
      steps.push({
        index,
        actionNodeId,
        applicable: false,
        failedPrecondition: sim.failedPrecondition,
        unsatisfied: sim.unsatisfiedPreconditions,
      })
      firstFailureIndex = index
      break // step k's failure means k+1.. never get their preconditions
    }
    steps.push({ index, actionNodeId, applicable: true, unsatisfied: [] })
    // Advance the clone so the next step sees the post-action state.
    deriveActionEffects(clone, spaceId, actionNodeId)
    // First prefix that reaches every goal: this step's run (index+1 steps) suffices.
    if (shortestPrefixLength === undefined && allGoalsSatisfied(clone, spaceId, totalGoals)) {
      shortestPrefixLength = index + 1
    }
  }

  // Read goal satisfaction on the clone in its final (as-far-as-it-ran) state.
  const allGoals = goalIds(clone, spaceId)
  const satisfiedGoalIds: string[] = []
  const unmetGoalIds: string[] = []
  if (allGoals.length > 0) {
    const ctx = getLogicContext(clone, spaceId)
    for (const goal of ctx.goals) {
      if (goal.satisfied) satisfiedGoalIds.push(goal.nodeId)
      else unmetGoalIds.push(goal.nodeId)
    }
  }

  const allApplied = firstFailureIndex === undefined && steps.length === actionNodeIds.length
  const goalsOk = allGoals.length === 0 || unmetGoalIds.length === 0

  // Steps that executed after the goal was already reached are prune candidates.
  const redundantStepIndices =
    shortestPrefixLength === undefined
      ? []
      : steps.filter((s) => s.applicable && s.index >= shortestPrefixLength!).map((s) => s.index)

  return {
    ok: allApplied && goalsOk,
    steps,
    firstFailureIndex,
    satisfiedGoalIds,
    unmetGoalIds,
    shortestPrefixLength,
    redundantStepIndices,
  }
}
