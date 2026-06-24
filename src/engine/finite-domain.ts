/**
 * finite-domain — 慢轨 #2, first slice: the board as a constraint SOLVER.
 *
 * `constraint-agent` already shows the board as a constraint CHECKER: the model
 * lays down an assignment + a rule, and the closure DERIVES `conflict(...)` when
 * a constraint is violated. The missing half is SOLVE: given variables with
 * finite candidate domains, FIND an assignment with no violation — instead of the
 * model guessing it (exact search is a known model weakness).
 *
 * This is a bounded backtracking search. Critically, it is a PROPOSER, not an
 * oracle: every candidate is judged by the `isConsistent` adjudicator the caller
 * supplies, and the intended adjudicator is the BOARD CLOSURE itself (apply the
 * partial assignment to a board, run closure, accept iff no `conflict` is
 * derived). So:
 *
 *   - The search lives OUTSIDE the deductive core (heuristic layer, like
 *     plan-search) — it never decides truth.
 *   - A buggy or incomplete search can only return a candidate the board then
 *     REJECTS, or honestly report "no solution found" — it can NEVER certify a
 *     wrong answer, because correctness rests on the closure's exact check.
 *
 * That is the propose/adjudicate guarantee applied to constraint solving: the
 * solver makes the creative search cheap; the board keeps it honest.
 *
 * BOUNDARY (do not over-claim):
 *   - FINITE domains only — every variable ranges over an enumerable candidate
 *     list. Infinite/continuous/rational domains (true linear arithmetic) are a
 *     later slice and would touch the core's number model; deliberately not here.
 *   - `unsat` means "no assignment in the PRODUCT OF THE DECLARED DOMAINS
 *     satisfies the adjudicator", not a global unsatisfiability proof.
 *   - BOUNDED: `maxNodes` caps the search; exhausting it returns `budget`
 *     (fail-visible escalation), never a hang and never a false `unsat`.
 *   - Naive backtracking with partial-assignment pruning. Forward-checking /
 *     AC-3 / heuristics are future optimizations, not correctness.
 */

/** A constraint-variable value: the board's scalar shapes. */
export type FdValue = string | number | boolean

export type FdVariable = {
  name: string
  /** The finite list of candidate values this variable may take. */
  domain: readonly FdValue[]
}

/**
 * Adjudicator: is this (partial) assignment still consistent? Called on partial
 * assignments to allow pruning — it must return true for a partial assignment
 * that no constraint can yet falsify, and false only once a constraint is
 * actually violated. The board-backed form runs the closure and returns
 * `no conflict derived`.
 */
export type FdConsistency = (partial: Readonly<Record<string, FdValue>>) => boolean

export type FdSolveOptions = {
  /** Search budget in assignment attempts. Default 100_000. */
  maxNodes?: number
}

export type FdSolveResult =
  | { sat: true; assignment: Record<string, FdValue>; nodes: number }
  | { sat: false; reason: 'unsat' | 'budget'; nodes: number }

/**
 * Search the product of the variables' finite domains for an assignment the
 * adjudicator accepts. Returns the first such assignment (`sat`), or `unsat`
 * when the whole product is exhausted with none, or `budget` when the node cap
 * is hit first.
 */
export function solveFiniteDomain(
  variables: readonly FdVariable[],
  isConsistent: FdConsistency,
  opts: FdSolveOptions = {},
): FdSolveResult {
  const maxNodes = opts.maxNodes ?? 100_000
  const seen = new Set<string>()
  for (const [index, variable] of variables.entries()) {
    if (!variable.name) {
      throw new Error(`solveFiniteDomain: variable at index ${index} has an empty name`)
    }
    if (seen.has(variable.name)) {
      throw new Error(
        `solveFiniteDomain: duplicate variable name "${variable.name}" would overwrite an assignment; ` +
          `variable names must be unique.`,
      )
    }
    seen.add(variable.name)
  }
  const assignment: Record<string, FdValue> = {}
  let nodes = 0
  let budgetHit = false

  if (variables.length === 0) {
    return isConsistent(assignment)
      ? { sat: true, assignment: {}, nodes }
      : { sat: false, reason: 'unsat', nodes }
  }

  const backtrack = (i: number): boolean => {
    if (budgetHit) return false
    if (i === variables.length) return true // all variables assigned and consistent
    const v = variables[i]
    for (const value of v.domain) {
      nodes += 1
      if (nodes > maxNodes) {
        budgetHit = true
        return false
      }
      assignment[v.name] = value
      if (isConsistent(assignment) && backtrack(i + 1)) return true
      delete assignment[v.name]
      if (budgetHit) return false
    }
    return false
  }

  const solved = backtrack(0)
  if (solved) return { sat: true, assignment: { ...assignment }, nodes }
  if (budgetHit) return { sat: false, reason: 'budget', nodes }
  return { sat: false, reason: 'unsat', nodes }
}
