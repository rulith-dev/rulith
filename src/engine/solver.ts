import type { PredicateAtom } from '../model/types.js'
import type { SpaceStore } from '../storage/space-store.js'
import { planToGoal } from './plan-search.js'
import { solveConstraintsOnBoard, type ConstraintSolveSpec } from './constraint-solve.js'
import { validatePlan, cloneSpace } from './validate-plan.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'

/**
 * Solver — the delegation seam (➕, open-core; the federation §7 plug point).
 *
 * A Solver routes a subgoal to a backend that returns a CHECKABLE CERTIFICATE or
 * an HONEST GAP. The core RE-VERIFIES every certificate by closure
 * (`verifySolverCertificate`) — a forged one is rejected — so a future external
 * backend (Z3 / ATP / Lean, see theory §7) can be delegated to WITHOUT trusting
 * it: "search proposes, core adjudicates", abstracted into one interface. rulith
 * records a verified certificate as `attested(provenance=<backend>)` or an honest
 * gap; it never claims a result it could not re-check.
 *
 * Thin by design (architecture 三铁律). The built-in `BoardSolver` wraps the
 * existing board search (`planToGoal` / `solveConstraintsOnBoard`), which already
 * re-verify; its certificates always pass `verifySolverCertificate`. OPT-IN.
 */

export type SolverCertificate =
  | { kind: 'plan'; plan: string[]; label: string }
  | { kind: 'assignment'; facts: PredicateAtom[]; label: string }

export type SolveRequest =
  | { kind: 'plan'; options?: { maxDepth?: number; maxBeam?: number } }
  | { kind: 'constraint'; spec: ConstraintSolveSpec }

export type SolverResult =
  | { ok: true; certificate: SolverCertificate; stats?: { nodes?: number } }
  | { ok: false; reason: 'unsat' | 'budget' | 'error'; note?: string }

export interface Solver {
  solve(store: SpaceStore, spaceId: string, request: SolveRequest): SolverResult
}

/**
 * Re-verify a certificate by the CORE — the adjudication an external solver cannot
 * skip. Plan: `validatePlan` must apply every step AND reach the goals. Assignment:
 * lay the facts on a throwaway clone, run closure, accept iff no `conflict` derives.
 * This is what keeps a delegated backend honest: it can only hand back a
 * certificate the closure itself accepts.
 */
export function verifySolverCertificate(
  store: SpaceStore,
  spaceId: string,
  cert: SolverCertificate,
  opts: { conflictPredicate?: string } = {},
): { verified: boolean; detail?: string } {
  if (cert.kind === 'plan') {
    const v = validatePlan(store, spaceId, cert.plan)
    return v.ok
      ? { verified: true }
      : { verified: false, detail: `plan does not validate to the goals (unmet: ${v.unmetGoalIds.join(', ') || 'none'})` }
  }
  const conflictPredicate = opts.conflictPredicate ?? 'conflict'
  const { clone } = cloneSpace(store, spaceId)
  const ops: WorkingMemoryOperation[] = cert.facts.map((atom, i) => ({
    op: 'assert_fact',
    id: `__cert_${i}`,
    predicate: atom.predicate,
    args: atom.args ?? {},
  }))
  if (ops.length > 0) applyWorkingMemoryOperations(clone, spaceId, ops, { source: 'system' })
  const conflict = getLogicContext(clone, spaceId).facts.some(
    (f) => f.atom.predicate === conflictPredicate && f.derived,
  )
  return conflict
    ? { verified: false, detail: `certificate derives ${conflictPredicate} — not a valid solution` }
    : { verified: true }
}

/**
 * Built-in Solver over the board's own bounded search. Its certificates are
 * already core-verified (planToGoal / solveConstraintsOnBoard re-verify), so they
 * always pass `verifySolverCertificate`. The constraint path solves WITHOUT
 * committing — the seam returns a certificate; the caller commits after
 * re-verifying, keeping "core adjudicates" explicit.
 */
export class BoardSolver implements Solver {
  solve(store: SpaceStore, spaceId: string, request: SolveRequest): SolverResult {
    if (request.kind === 'plan') {
      const r = planToGoal(store, spaceId, request.options)
      if (!r.found) return { ok: false, reason: 'unsat', note: r.note }
      return {
        ok: true,
        certificate: { kind: 'plan', plan: r.plan, label: `plan of ${r.plan.length} action(s)` },
      }
    }
    const r = solveConstraintsOnBoard(store, spaceId, { ...request.spec, commit: false })
    if (!r.sat) return { ok: false, reason: r.reason, note: `no assignment (${r.nodes} nodes)` }
    const assignPredicate = request.spec.assignPredicate ?? 'assignment'
    const facts: PredicateAtom[] = Object.entries(r.assignment).map(([name, value]) => ({
      predicate: assignPredicate,
      args: { var: name, value },
    }))
    return {
      ok: true,
      certificate: { kind: 'assignment', facts, label: `assignment (${r.nodes} nodes)` },
      stats: { nodes: r.nodes },
    }
  }
}
