import type { PredicateAtom } from '../model/types.js'
import type { SpaceStore } from '../storage/space-store.js'
import type { PredicateFact } from '../kernel/predicate.js'
import { validatePlan, cloneSpace, type PlanValidation } from './validate-plan.js'
import { deriveActionEffects } from './semantic-derivation.js'
import { abduceProducingActions, collectActionDefinitions } from './abduction.js'
import { getLogicContext } from './logic-context.js'

/**
 * suggest_plan_repairs — when validate_plan reports a broken plan, propose
 * action insertions that would unblock it, by SEARCHING THE BOARD'S OWN
 * actions (no general planner in the kernel).
 *
 * The loop so far: validate_plan tells you WHERE a plan breaks and on WHICH
 * unmet precondition; the model then has to hand-search its actions for one
 * that produces the missing atom. suggestPlanRepairs does that search: it
 * finds defined actions whose positive effect would produce the failed
 * precondition and, multi-hop, chases each producer's own unmet preconditions
 * up to a depth bound. Each candidate is RE-VALIDATED on a clone before being
 * offered, so a suggestion never claims to fix a plan it does not.
 *
 * Discipline (the maintainer hates a feature that bloats into a mini-planner):
 * - SUGGESTION ONLY: returns copyable candidate plans; never mutates the board,
 *   never auto-applies. The model re-runs validate_plan / apply_plan itself.
 * - BOUNDED: maxDepth caps the producer-chain hops; maxActions caps inserted
 *   actions; a visited set forbids cycles. No unbounded search.
 * - GROUNDED: validatePlan + abduceProducingActions are reused verbatim, so a
 *   suggestion agrees with what apply would actually do - no parallel logic.
 *
 * (Converges with Codex's exported suggestPlanRepairs API; built here on main's
 * validatePlan + abduceProducingActions + cloneSpace rather than re-imported.)
 */
export type PlanRepair = {
  /** The full repaired plan, ready to copy into validate_plan / apply_plan. */
  actionNodeIds: string[]
  /** Just the producer actions inserted (in execution order). */
  insertedActionNodeIds: string[]
  /** The failed precondition this insertion was chosen to satisfy. */
  resolvedPrecondition: PredicateAtom
  /** The repaired plan validates fully (every step runs AND goals are reached). */
  validates: boolean
  /** The repaired plan at least runs past the original failure point. */
  runsPastFailure: boolean
}

export type PlanRepairResult = {
  /** The step the original plan first failed at (undefined = plan was already ok). */
  failedIndex?: number
  /** The unmet precondition at the failure (undefined = no actionable gap). */
  failedPrecondition?: PredicateAtom
  /** Candidate repairs, best (fully-validating, shortest) first. */
  repairs: PlanRepair[]
  /** Teaching note when there is nothing to repair or no repair was found. */
  note?: string
}

export function suggestPlanRepairs(
  store: SpaceStore,
  spaceId: string,
  actionNodeIds: string[],
  validation?: PlanValidation,
  options: { maxDepth?: number; maxActions?: number } = {},
): PlanRepairResult {
  const maxDepth = Math.max(1, options.maxDepth ?? 5)
  const maxActions = Math.max(1, options.maxActions ?? maxDepth)

  const v = validation ?? validatePlan(store, spaceId, actionNodeIds)
  if (v.firstFailureIndex === undefined) {
    return { repairs: [], note: 'plan already validates - nothing to repair (use validate_plan for prune hints).' }
  }
  const failedIndex = v.firstFailureIndex
  const failedStep = v.steps[failedIndex]
  const target = failedStep?.failedPrecondition ?? failedStep?.unsatisfied[0]
  if (!target) {
    return {
      failedIndex,
      repairs: [],
      note:
        `step #${failedIndex} failed on ${failedStep?.error ? 'an invalid/unusable action' : 'a guard or arithmetic literal'}, ` +
        `not a missing fact - no action can "produce" it. Fix the action definition or the ordering.`,
    }
  }

  // Reconstruct the board state the inserted actions would run against: clone,
  // then apply the applicable prefix (steps 0..failedIndex-1 all ran in v).
  const { clone } = cloneSpace(store, spaceId)
  for (let i = 0; i < failedIndex; i += 1) {
    deriveActionEffects(clone, spaceId, actionNodeIds[i]!)
  }
  const ctx = getLogicContext(clone, spaceId)
  const facts: PredicateFact[] = [...ctx.facts, ...ctx.findings].map((f) => ({ id: f.nodeId, atom: f.atom }))
  const actionDefs = collectActionDefinitions(clone.listNodes(spaceId))
  const planActionIds = new Set(actionNodeIds)

  // Greedy producer chain for an atom: an applicable producer ends the chain;
  // a blocked producer recurses on its first unmet precondition (multi-hop),
  // executing the deeper producer first. visited forbids cycles; depth bounds it.
  const chainFor = (atom: PredicateAtom): string[] | null => {
    const visited = new Set<string>()
    const dfs = (t: PredicateAtom, depth: number): string[] | null => {
      if (depth > maxDepth) return null
      for (const hint of abduceProducingActions(t, actionDefs, facts)) {
        // Do not re-insert an action the plan already runs, nor revisit within a chain.
        if (visited.has(hint.actionNodeId) || planActionIds.has(hint.actionNodeId)) continue
        if (hint.applicable) return [hint.actionNodeId]
        if (hint.blockedOn) {
          visited.add(hint.actionNodeId)
          const sub = dfs(hint.blockedOn, depth + 1)
          visited.delete(hint.actionNodeId)
          if (sub && sub.length + 1 <= maxActions) return [...sub, hint.actionNodeId]
        }
      }
      return null
    }
    return dfs(atom, 1)
  }

  // Offer one candidate per distinct producer of the failed precondition (so the
  // model sees real alternatives), each extended multi-hop. Re-validate each.
  const repairs: PlanRepair[] = []
  const seenChains = new Set<string>()
  for (const top of abduceProducingActions(target, actionDefs, facts)) {
    if (planActionIds.has(top.actionNodeId)) continue
    let chain: string[] | null
    if (top.applicable) {
      chain = [top.actionNodeId]
    } else if (top.blockedOn) {
      const sub = chainFor(top.blockedOn)
      chain = sub && sub.length + 1 <= maxActions ? [...sub, top.actionNodeId] : null
    } else {
      chain = null
    }
    if (!chain) continue
    const key = chain.join('>')
    if (seenChains.has(key)) continue
    seenChains.add(key)

    const candidate = [...actionNodeIds.slice(0, failedIndex), ...chain, ...actionNodeIds.slice(failedIndex)]
    const cv = validatePlan(store, spaceId, candidate)
    const runsPastFailure = cv.firstFailureIndex === undefined || cv.firstFailureIndex > failedIndex
    if (!runsPastFailure) continue // only offer insertions that actually help
    repairs.push({
      actionNodeIds: candidate,
      insertedActionNodeIds: chain,
      resolvedPrecondition: target,
      validates: cv.ok,
      runsPastFailure,
    })
  }

  // Best first: fully-validating before partial, then fewest insertions.
  repairs.sort((a, b) =>
    a.validates === b.validates
      ? a.insertedActionNodeIds.length - b.insertedActionNodeIds.length
      : Number(b.validates) - Number(a.validates),
  )

  return {
    failedIndex,
    failedPrecondition: target,
    repairs,
    note:
      repairs.length === 0
        ? `no defined action produces ${formatTarget(target)} (within depth ${maxDepth}). ` +
          `Define an action whose effect asserts it, or fix the failing step's ordering.`
        : undefined,
  }
}

function formatTarget(atom: PredicateAtom): string {
  const args = atom.args
    ? Object.entries(atom.args)
        .map(([k, val]) => `${k}=${String(val)}`)
        .join(', ')
    : ''
  return `${atom.predicate}(${args})`
}
