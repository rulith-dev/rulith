import type { SpaceStore } from '../storage/space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import { cloneSpace } from './validate-plan.js'
import { solveFiniteDomain, type FdVariable, type FdValue } from './finite-domain.js'

/**
 * constraint-solve — the board-integrated DRIVER that turns the bare
 * `solveFiniteDomain` search into a usable board capability (慢轨 #2).
 *
 * This is the "external driver" the capability needs: a search loop in which the
 * BOARD CLOSURE is the adjudicator. For each candidate assignment the search
 * proposes, the driver lays it down as `assignment(var, value)` facts on a
 * THROWAWAY CLONE of the board (which already carries the model's constraint
 * rules + base facts), runs the closure, and accepts the candidate iff no
 * `conflict` predicate is DERIVED. The real board is never touched during the
 * search. On success the certified assignment is committed to the real board and
 * the closure RE-VERIFIES it there.
 *
 * propose/adjudicate, intact: the model writes the constraint rules in its normal
 * language (a rule that derives `conflict(...)` on a violation — exactly the
 * board-as-checker pattern of constraint-agent); this driver only searches the
 * finite domains and lets the closure judge. A wrong search can only get a
 * candidate rejected or honestly report no solution — the committed answer is
 * always closure-certified.
 *
 * REPRESENTATION (minimal new surface): each decision variable's chosen value is
 * carried by one bridge fact `assignment(var=<name>, value=<chosen>)`. The model
 * writes its constraint rules over `assignment(...)` (plus any base facts like
 * `edge`, `requires`, capacities). Defaults: bridge predicate `assignment`,
 * violation predicate `conflict` — both overridable.
 *
 * BOUNDARY: finite domains only; `unsat` = no assignment in the declared domain
 * product passes the closure (not a global proof); bounded by `maxNodes`
 * (-> `budget`, fail-visible); one clone + closure per visited node (naive — fine
 * for small problems, forward-checking is a later optimization). Assumes the
 * board has no pre-existing `assignment(var=...)` facts for these variables.
 */

export type ConstraintSolveSpec = {
  /** Decision variables with their finite candidate domains. */
  variables: FdVariable[]
  /** Predicate whose DERIVATION marks a violated constraint. Default 'conflict'. */
  conflictPredicate?: string
  /** Bridge predicate carrying a chosen value, args {var, value}. Default 'assignment'. */
  assignPredicate?: string
  /** Search budget (assignment attempts). */
  maxNodes?: number
  /** Commit the certified assignment to the real board. Default true. */
  commit?: boolean
}

export type ConstraintSolveResult =
  | { sat: true; assignment: Record<string, FdValue>; nodes: number; committed: boolean }
  | { sat: false; reason: 'unsat' | 'budget'; nodes: number }

function assignmentOps(
  assignPredicate: string,
  assignment: Readonly<Record<string, FdValue>>,
): WorkingMemoryOperation[] {
  return Object.entries(assignment).map(
    ([name, value]) =>
      ({
        op: 'assert_fact',
        id: `__solve_${name}`,
        predicate: assignPredicate,
        args: { var: name, value },
      }) as WorkingMemoryOperation,
  )
}

function derivesConflict(
  store: SpaceStore,
  spaceId: string,
  conflictPredicate: string,
): boolean {
  return getLogicContext(store, spaceId).facts.some(
    (f) => f.atom.predicate === conflictPredicate && f.derived,
  )
}

/**
 * Solve the finite-domain CSP described by the board's constraint rules, with the
 * closure adjudicating every candidate. Returns the committed, closure-certified
 * assignment (`sat`), or an honest `unsat` / `budget`.
 */
export function solveConstraintsOnBoard(
  store: SpaceStore,
  spaceId: string,
  spec: ConstraintSolveSpec,
): ConstraintSolveResult {
  const conflictPredicate = spec.conflictPredicate ?? 'conflict'
  const assignPredicate = spec.assignPredicate ?? 'assignment'
  const commit = spec.commit ?? true

  // Adjudicator = the board closure on a throwaway clone: lay the candidate as
  // assignment facts, run closure, accept iff no conflict is derived.
  const isConsistent = (partial: Readonly<Record<string, FdValue>>): boolean => {
    const { clone } = cloneSpace(store, spaceId)
    const ops = assignmentOps(assignPredicate, partial)
    if (ops.length > 0) applyWorkingMemoryOperations(clone, spaceId, ops, { source: 'system' })
    return !derivesConflict(clone, spaceId, conflictPredicate)
  }

  const result = solveFiniteDomain(spec.variables, isConsistent, { maxNodes: spec.maxNodes })
  if (!result.sat) return result

  if (!commit) {
    return { sat: true, assignment: result.assignment, nodes: result.nodes, committed: false }
  }

  // Commit to the real board, then have the closure RE-CERTIFY there.
  applyWorkingMemoryOperations(
    store,
    spaceId,
    assignmentOps(assignPredicate, result.assignment),
    { source: 'system' },
  )
  if (derivesConflict(store, spaceId, conflictPredicate)) {
    throw new Error(
      `solveConstraintsOnBoard: the committed assignment derived a ${conflictPredicate} on the ` +
        `real board — clone/real divergence (pre-existing ${assignPredicate} facts, or a rule that ` +
        `reads state the clone lacked). Refusing to certify.`,
    )
  }
  return { sat: true, assignment: result.assignment, nodes: result.nodes, committed: true }
}
