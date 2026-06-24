import type { SpaceStore } from '../storage/space-store.js'
import { validatePlan, type PlanValidation } from './validate-plan.js'
import { deriveActionEffects } from './semantic-derivation.js'
import { boardRevision } from './board-revision.js'

/**
 * apply_plan — validate a WHOLE action sequence, then commit it to the real
 * board step by step, each step guarded against board drift.
 *
 * The planning loop so far: validate_plan dry-runs a sequence on a clone, and
 * apply_action commits ONE action. apply_plan closes the gap - commit the
 * whole validated plan in order, with every step pinned by the board revision
 * the prior step produced, so a concurrent mutation between validation and
 * commit is caught rather than silently applied (C4 discipline at plan scope).
 *
 * Discipline (anti-slop, vs the rejected variant): a mid-sequence step that
 * cannot apply returns a STRUCTURED failure (applied:false + the failing index
 * + what committed so far), NEVER throws. apply_plan returns a value its caller
 * can branch on, exactly like apply_action - no exception path the tests can't
 * cover. (Absorbed concept from Codex's guarded apply_plan; reworked to return
 * structured failure and to build on main's validatePlan + boardRevision.)
 */
export type AppliedPlanStep = {
  index: number
  actionNodeId: string
  /** Board revision AFTER this step committed. */
  revision: string
}

export type PlanApplication = {
  /** The whole validated plan committed in order. */
  applied: boolean
  /** Pre-commit validation; when not ok, nothing is committed. */
  validation: PlanValidation
  steps: AppliedPlanStep[]
  /** Action ids that actually committed (prefix of the plan). */
  appliedActionNodeIds: string[]
  /** Set when a step failed at COMMIT despite passing validation (board drift,
   *  or expectedRevision mismatch) - structured, never thrown. */
  failedIndex?: number
  failureReason?: string
  /** Board revision after the last committed step (or the start revision). */
  finalRevision: string
}

export function applyPlan(
  store: SpaceStore,
  spaceId: string,
  actionNodeIds: string[],
  options: { requireGoals?: boolean } = {},
): PlanApplication {
  const startRevision = boardRevision(store.listNodes(spaceId))

  // 1. Validate the whole sequence on a clone first. If it cannot run in order
  //    (or, when requireGoals, does not reach the goals), commit nothing.
  const validation = validatePlan(store, spaceId, actionNodeIds)
  const validationOk = options.requireGoals ? validation.ok : validation.firstFailureIndex === undefined
  if (!validationOk) {
    return {
      applied: false,
      validation,
      steps: [],
      appliedActionNodeIds: [],
      failedIndex: validation.firstFailureIndex,
      failureReason:
        validation.firstFailureIndex !== undefined
          ? `plan does not validate: step #${validation.firstFailureIndex} cannot run`
          : `plan validates but does not reach all goals (unmet: ${validation.unmetGoalIds.join(', ')})`,
      finalRevision: startRevision,
    }
  }

  // 2. Commit each step to the REAL board, pinning every step to the revision
  //    the previous step produced. A drift between validation and commit (or
  //    between steps) trips the guard - return structured failure, never throw.
  const steps: AppliedPlanStep[] = []
  const appliedActionNodeIds: string[] = []
  let revision = startRevision
  for (let index = 0; index < actionNodeIds.length; index += 1) {
    const actionNodeId = actionNodeIds[index]!
    let result
    try {
      result = deriveActionEffects(store, spaceId, actionNodeId, { expectedRevision: revision })
    } catch (error) {
      return {
        applied: false, validation, steps, appliedActionNodeIds,
        failedIndex: index,
        failureReason: `step #${index} ${actionNodeId} threw at commit: ${String(error).slice(0, 160)}`,
        finalRevision: revision,
      }
    }
    if (!result.applied) {
      return {
        applied: false, validation, steps, appliedActionNodeIds,
        failedIndex: index,
        failureReason:
          `step #${index} ${actionNodeId} did not apply at commit (board drift since validation?). ` +
          `Re-validate from the current board.`,
        finalRevision: revision,
      }
    }
    revision = boardRevision(store.listNodes(spaceId))
    steps.push({ index, actionNodeId, revision })
    appliedActionNodeIds.push(actionNodeId)
  }

  return { applied: true, validation, steps, appliedActionNodeIds, finalRevision: revision }
}
