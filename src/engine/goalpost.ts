/**
 * goalpost — the committed-baseline (I12 no-goalpost-moving) primitives, factored out so BOTH the
 * working-memory batch gate (assertNoGoalpostMoving) and the action layer (deriveActionEffects /
 * simulateActionEffects) can enforce the same protection without an import cycle (working-memory
 * already imports semantic-derivation, so semantic-derivation must not import working-memory).
 *
 * A committed baseline is a trusted `committed_*` marker that pins a target fact: a model may not
 * silently retract/revise/consume it to make the task easier. The batch gate guards model-sourced
 * retract_node/revise_fact; the action layer guards an action whose negated (consume) effect would
 * archive a committed target OR the marker itself (#B7) — real deepseek dodged the batch gate by
 * wrapping the consume in a define_action + apply_action (the action version of #101 rule-laundering).
 */
import type { PredicateAtom, SemanticArgs, SpaceNode } from '../model/types.js'
import { logicallyUsableNodes } from './semantic-active.js'

export class GoalpostMovingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GoalpostMovingError'
  }
}

export type ProtectedGoalpost = {
  marker: string
  target: string
  /** Key used on the target fact being changed. Defaults to `key`. */
  key: string[]
  /** Key used on the committed marker. Defaults to `key`. */
  markerKey?: string[]
  /** Key used on amendment_result. Defaults to `key`. */
  amendmentKey?: string[]
  amendmentKind: string
  allowMissingAmendmentKind?: boolean
}

export const PROTECTED_GOALPOSTS: ProtectedGoalpost[] = [
  { marker: 'committed_acceptance', target: 'acceptance', key: ['node'], amendmentKind: 'acceptance', allowMissingAmendmentKind: true },
  { marker: 'committed_constraint', target: 'constraint', key: ['node', 'tag'], amendmentKind: 'constraint' },
  { marker: 'committed_constraint', target: 'constraint_scope', key: ['tag'], markerKey: ['tag'], amendmentKey: ['tag'], amendmentKind: 'constraint' },
  { marker: 'committed_goal', target: 'goal_node', key: ['node'], amendmentKind: 'structure' },
  { marker: 'committed_subgoal', target: 'subgoal_of', key: ['child', 'parent'], amendmentKind: 'structure' },
  // ③ self-driving phase (theory §3.0 object 1): the agent's current lifetime phase goal. Keyed on the
  // constant `slot` (the IDENTITY, e.g. 'current'), NOT `phase` (the protected value) — a switch CHANGES
  // the phase value, so keying on phase would unhook the gate right after the first authorized switch and
  // break one-shot. An APPENDED second current_phase is caught separately by a domain-declared
  // functional_dependency(current_phase, slot). Lifetime-scale no-goalpost: no silent self-switch to an
  // easier phase; a real switch needs a trusted, one-shot amendment_result(kind='phase').
  { marker: 'committed_phase', target: 'current_phase', key: ['slot'], amendmentKind: 'phase' },
  // ② 2c intent-coverage: the domain's required-coverage checklist (required_coverage(root, item)) is a
  // committed baseline — a model may not retract/revise it to drop a requirement and dodge the coverage
  // gate (coverage_gap_in → blocks task_done). Pinned by committed_coverage(root, item); one-shot
  // amendment_result(kind='coverage'). Opt-in (inert without the marker).
  { marker: 'committed_coverage', target: 'required_coverage', key: ['root', 'item'], amendmentKind: 'coverage' },
]

export function goalpostKey(args: SemanticArgs | undefined, fields: string[]): string | null {
  const values: string[] = []
  for (const field of fields) {
    const value = args?.[field]
    if (typeof value !== 'string' || value.length === 0) return null
    values.push(value)
  }
  return JSON.stringify(values)
}

export function amendmentMatches(args: SemanticArgs | undefined, spec: ProtectedGoalpost): boolean {
  const kind = args?.kind
  if (kind === undefined && spec.allowMissingAmendmentKind) return true
  return kind === spec.amendmentKind
}

export function formatGoalpostKey(spec: ProtectedGoalpost, key: string): string {
  const values = JSON.parse(key) as string[]
  return spec.key.map((field, index) => `${field}="${values[index] ?? ''}"`).join(', ')
}

/** A committed-baseline identity an action effect may not consume: a (predicate, key-fields, key-value)
 *  triple. Includes BOTH the target a marker pins and the marker itself (#B7). */
export type CommittedTarget = { predicate: string; key: string[]; keyValue: string }

/**
 * The committed-baseline identities currently on the board — every trusted `committed_*` marker yields
 * its TARGET identity (the fact it pins) and its MARKER identity (the marker fact itself). An action
 * effect that consumes any of these is goalpost-moving via the action layer. Mirrors the marker scan
 * in assertNoGoalpostMoving; keyValue uses `markerKey ?? key` from the marker's own args.
 */
export function committedBaselineTargets(nodes: SpaceNode[]): CommittedTarget[] {
  const usable = logicallyUsableNodes(nodes)
  const trusted = (n: SpaceNode): boolean => n.createdBy === 'tool' || n.createdBy === 'system'
  const out: CommittedTarget[] = []
  for (const node of usable) {
    if (node.type !== 'fact' || node.semantic?.kind !== 'predicate' || !trusted(node)) continue
    for (const spec of PROTECTED_GOALPOSTS) {
      if (node.semantic.predicate !== spec.marker) continue
      const keyValue = goalpostKey(node.semantic.args, spec.markerKey ?? spec.key)
      if (keyValue === null) continue
      out.push({ predicate: spec.target, key: spec.key, keyValue }) // the pinned target
      out.push({ predicate: spec.marker, key: spec.markerKey ?? spec.key, keyValue }) // the marker itself (#B7)
    }
  }
  return out
}

/** Does `atom` (an action's consume/negated effect) hit a committed-baseline identity? */
export function committedBaselineHit(atom: PredicateAtom, targets: CommittedTarget[]): boolean {
  return targets.some((t) => atom.predicate === t.predicate && goalpostKey(atom.args, t.key) === t.keyValue)
}
