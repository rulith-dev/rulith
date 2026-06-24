import type { RuleDefinition } from '../kernel/predicate.js'

/**
 * derive-stages (pure layering core) — the deterministic dependency-DAG layering
 * of a rule program. This is FIRST-LAYER analysis (topological ordering of the
 * dependency DAG, NOT search): a derived predicate sits at depth = 1 + max depth
 * of the predicates its rules read; base inputs are depth 0. So "establish
 * depth-1 predicates, then depth-2, ..." is a deterministic, board-derived
 * ordering, not a guess. (See foundations.md "拓扑 vs 搜索".)
 *
 * It lives in ENGINE (not agent) so the compile backends (compile-board /
 * compile-board-sql) can order their layers from it WITHOUT importing upward into
 * agent. The MODEL-FACING staging wrapper (`deriveStages` → `Stage[]` with goalText
 * and per-layer board gates) lives in `src/agent/derive-stages.ts` and delegates
 * here for the layering, then wraps each layer in a Stage.
 */

export type DeriveStageLayers = {
  /** Goal-ancestor predicates that are NOT any rule's head = raw inputs (must be
   *  ingested before layer 0; not a derivation layer). */
  baseInputs: string[]
  /** Dependency-depth layers (ascending) the goal actually depends on. */
  layers: number[]
  /** Per-layer predicate grouping, ascending layer (predicates sorted). Lets a
   *  caller seed each layer's rules or build a custom gate over its targets. */
  groups: { layer: number; predicates: string[] }[]
  note: string
}

/** A predicate is DERIVABLE iff it is some rule's head. */
function headPredicates(rules: RuleDefinition[]): Set<string> {
  const heads = new Set<string>()
  for (const rule of rules) for (const h of rule.then ?? []) heads.add(h.predicate)
  return heads
}

/** Map each head predicate -> the body predicates its rules read (its parents). */
function dependencyParents(rules: RuleDefinition[]): Map<string, Set<string>> {
  const parents = new Map<string, Set<string>>()
  for (const rule of rules) {
    const body = (rule.when ?? []).map((a) => a.predicate)
    for (const h of rule.then ?? []) {
      const set = parents.get(h.predicate) ?? new Set<string>()
      for (const b of body) set.add(b)
      parents.set(h.predicate, set)
    }
  }
  return parents
}

/**
 * Dependency depth of each predicate: base inputs (never a rule head) = 0; a head
 * predicate = 1 + max depth of the predicates its rules read. Fixpoint relaxation
 * capped at #predicates: predicates still rising at the cap are in a dependency
 * cycle (e.g. a recursive rule) and are clamped to the cap — a recursive fixpoint
 * is ONE layer, not a sequence (honest: the board derives it in a single closure,
 * there is nothing to stage inside it).
 */
function dependencyDepth(
  rules: RuleDefinition[],
  heads: Set<string>,
  parents: Map<string, Set<string>>,
): Map<string, number> {
  const depth = new Map<string, number>()
  const allPreds = new Set<string>()
  for (const rule of rules) {
    for (const a of rule.when ?? []) allPreds.add(a.predicate)
    for (const a of rule.then ?? []) allPreds.add(a.predicate)
  }
  for (const p of allPreds) depth.set(p, heads.has(p) ? 1 : 0)

  const cap = allPreds.size + 1
  for (let i = 0; i < cap; i += 1) {
    let changed = false
    for (const p of heads) {
      let maxParent = 0
      for (const parent of parents.get(p) ?? []) {
        maxParent = Math.max(maxParent, depth.get(parent) ?? 0)
      }
      const next = maxParent + 1
      if (next > (depth.get(p) ?? 0)) {
        depth.set(p, next)
        changed = true
      }
    }
    if (!changed) break
  }
  // Clamp cycle-inflated depths to the cap (recursive predicates share a layer).
  for (const [p, d] of depth) if (d > cap) depth.set(p, cap)
  return depth
}

/**
 * Layer a rule program's derivation DAG for `goalPredicate` (ascending dependency
 * depth). Returns the base inputs, the layers, and the per-layer predicate groups
 * — the deterministic schedule. The model-facing `deriveStages` (agent) wraps this
 * in stages; the compile backends order their pipeline/views by it.
 */
export function deriveStageLayers(
  rules: RuleDefinition[],
  goalPredicate: string,
): DeriveStageLayers {
  const heads = headPredicates(rules)
  const parents = dependencyParents(rules)
  const predDepth = dependencyDepth(rules, heads, parents)

  // Transitive ancestors of the goal (including itself), split into derivable
  // (rule heads) vs base inputs (read but never derived).
  const derivable = new Set<string>()
  const baseInputs = new Set<string>()
  const seen = new Set<string>()
  const stack = [goalPredicate]
  while (stack.length > 0) {
    const p = stack.pop()!
    if (seen.has(p)) continue
    seen.add(p)
    if (heads.has(p)) {
      derivable.add(p)
      for (const parent of parents.get(p) ?? []) stack.push(parent)
    } else {
      baseInputs.add(p)
    }
  }

  // Group derivable goal-ancestors by dependency depth, ascending.
  const byLayer = new Map<number, string[]>()
  for (const p of derivable) {
    const d = predDepth.get(p) ?? 1
    const list = byLayer.get(d) ?? []
    list.push(p)
    byLayer.set(d, list)
  }
  const layers = [...byLayer.keys()].sort((a, b) => a - b)
  const groups = layers.map((layer) => ({
    layer,
    predicates: (byLayer.get(layer) ?? []).slice().sort(),
  }))

  const note = derivable.has(goalPredicate)
    ? `Derived ${groups.length} derivation stage(s) from the dependency layering for ` +
      `goal "${goalPredicate}". Base inputs to ingest first: ` +
      `${[...baseInputs].sort().join(', ') || '(none)'}. ` +
      `Action/effect and judgment stages (if any) are NOT inferred here.`
    : `"${goalPredicate}" is not derived by any rule (it is a base input). ` +
      `Nothing to stage — provide it as a fact, or supply rules that derive it.`

  return { baseInputs: [...baseInputs].sort(), layers, groups, note }
}
