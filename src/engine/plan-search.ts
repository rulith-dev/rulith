import type { SpaceStore } from '../storage/space-store.js'
import { formatAtom } from '../kernel/predicate.js'
import { validatePlan, cloneSpace, type PlanValidation } from './validate-plan.js'
import { deriveActionEffects } from './semantic-derivation.js'
import { collectActionDefinitions } from './abduction.js'
import { getLogicContext } from './logic-context.js'

/**
 * plan_to_goal — a BOUNDED forward search for an action sequence that reaches
 * the board's declared goals (roadmap P4). This is the heuristic layer: the
 * search PROPOSES a plan, the core JUDGES it. Every returned plan is re-checked
 * by validatePlan, so a suggestion never claims to reach a goal it does not.
 *
 * Discipline (foundations.md layer 2; the maintainer hates a kernel mini-planner):
 * - BOUNDED: maxDepth caps plan length, maxBeam caps states per level, and a
 *   visited set keyed by the LOGICAL state (the canonical set of active fact
 *   atoms - NOT boardRevision, which also fingerprints per-apply event nodes and
 *   uniquely-ided effect facts, so the same done-set would never dedup) prunes
 *   revisited states - no unbounded search.
 * - READ-ONLY: search runs on throwaway clones (cloneSpace); the real board is
 *   never mutated.
 * - GROUNDED: reuses cloneSpace + deriveActionEffects + collectActionDefinitions
 *   + boardRevision + validatePlan verbatim - no parallel apply/validate logic.
 * - SUGGESTION ONLY: returns a validated candidate; the caller commits it with
 *   apply_plan. The kernel stays the world model and the judge.
 */
export type PlanSearchResult = {
  /** A validated plan reaching every declared goal was found within the bound. */
  found: boolean
  /** The action node ids in execution order (empty when not found, or when the
   *  goals already hold before any action). */
  plan: string[]
  /** The family-consistent validation of `plan` (present when found). */
  validation?: PlanValidation
  /** Why no plan was found, when found is false (teaching note). */
  note?: string
}

/** total declared goals and how many hold right now on this store. */
function goalProgress(store: SpaceStore, spaceId: string): { total: number; satisfied: number } {
  const goals = getLogicContext(store, spaceId).goals
  return { total: goals.length, satisfied: goals.filter((g) => g.satisfied).length }
}

/** Canonical key for the LOGICAL state: the sorted set of active fact + finding
 *  atoms. Order-independent and id-independent, so two paths reaching the same
 *  world map to one key (boardRevision would not - it also hashes event nodes). */
function stateKey(store: SpaceStore, spaceId: string): string {
  const ctx = getLogicContext(store, spaceId)
  return [...ctx.facts, ...ctx.findings]
    .map((f) => formatAtom(f.atom))
    .sort()
    .join('|')
}

/** Replay an action sequence onto a fresh clone; returns the advanced clone. */
function cloneAfter(store: SpaceStore, spaceId: string, sequence: string[]): SpaceStore {
  const { clone } = cloneSpace(store, spaceId)
  for (const actionNodeId of sequence) {
    try {
      deriveActionEffects(clone, spaceId, actionNodeId)
    } catch {
      // a stale/inactive action in a candidate sequence just stops that replay
      break
    }
  }
  return clone
}

export function planToGoal(
  store: SpaceStore,
  spaceId: string,
  options: { maxDepth?: number; maxBeam?: number } = {},
): PlanSearchResult {
  const maxDepth = Math.max(1, options.maxDepth ?? 8)
  const maxBeam = Math.max(1, options.maxBeam ?? 16)

  const start = goalProgress(store, spaceId)
  if (start.total === 0) {
    return { found: false, plan: [], note: 'no declared goal to plan toward - declare_goal first.' }
  }
  if (start.satisfied === start.total) {
    // Already there: the empty plan is the (validated) answer.
    return { found: true, plan: [], validation: validatePlan(store, spaceId, []) }
  }

  const actionIds = collectActionDefinitions(store.listNodes(spaceId)).map((a) => a.id)
  if (actionIds.length === 0) {
    return { found: false, plan: [], note: 'no defined action can change the board - define_action first.' }
  }

  // Beam BFS over action sequences. State identity = boardRevision of the clone
  // the sequence produces (so two orders reaching the same world are not both
  // expanded). Each frontier entry carries its sequence + a goal-progress score.
  const visited = new Set<string>([stateKey(store, spaceId)])
  let frontier: { sequence: string[]; score: number }[] = [{ sequence: [], score: start.satisfied }]

  for (let depth = 0; depth < maxDepth; depth += 1) {
    const next: { sequence: string[]; score: number }[] = []
    for (const node of frontier) {
      const base = cloneAfter(store, spaceId, node.sequence)
      for (const actionNodeId of actionIds) {
        const { clone } = cloneSpace(base, spaceId)
        let applied = false
        try {
          applied = deriveActionEffects(clone, spaceId, actionNodeId).applied
        } catch {
          applied = false
        }
        if (!applied) continue
        const key = stateKey(clone, spaceId)
        if (visited.has(key)) continue
        visited.add(key)

        const sequence = [...node.sequence, actionNodeId]
        const progress = goalProgress(clone, spaceId)
        if (progress.satisfied === progress.total) {
          // Reached on the clone - now let the CORE judge the real plan.
          const validation = validatePlan(store, spaceId, sequence)
          if (validation.ok) return { found: true, plan: sequence, validation }
          // validatePlan disagreed (shouldn't normally) - keep searching.
        }
        next.push({ sequence, score: progress.satisfied })
      }
    }
    if (next.length === 0) break
    // Beam: keep the most goal-complete states (then shortest sequence).
    next.sort((a, b) => (b.score - a.score) || (a.sequence.length - b.sequence.length))
    frontier = next.slice(0, maxBeam)
  }

  return {
    found: false,
    plan: [],
    note:
      `no plan within depth ${maxDepth} reaches every goal (explored ${visited.size} states). ` +
      `Add the missing action/rule, or raise maxDepth.`,
  }
}
