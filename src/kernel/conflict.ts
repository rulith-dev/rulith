import type { PredicateAtom, SemanticArgs } from '../model/types.js'
import { atomKey, type PredicateFact } from './predicate.js'

export type PredicateConflict = {
  atom: PredicateAtom
  positiveFactId: string
  negativeFactId: string
}

export function detectPredicateConflicts(facts: PredicateFact[]): PredicateConflict[] {
  const positives = new Map<string, PredicateFact[]>()
  const negatives = new Map<string, PredicateFact[]>()

  for (const fact of facts) {
    const key = atomKey(asPositiveAtom(fact.atom))
    const bucket = fact.atom.negated === true ? negatives : positives
    bucket.set(key, [...(bucket.get(key) ?? []), fact])
  }

  const conflicts: PredicateConflict[] = []
  for (const [key, negativeFacts] of negatives) {
    const positiveFacts = positives.get(key) ?? []
    for (const positive of positiveFacts) {
      for (const negative of negativeFacts) {
        conflicts.push({
          atom: asPositiveAtom(positive.atom),
          positiveFactId: positive.id,
          negativeFactId: negative.id,
        })
      }
    }
  }

  return conflicts
}

function asPositiveAtom(atom: PredicateAtom): PredicateAtom {
  return {
    predicate: atom.predicate,
    args: atom.args,
  }
}

/** A domain-declared functional dependency: within `predicate`, the `key` args
 *  determine all the rest. e.g. { predicate: 'cost', key: ['item'] } says a given
 *  item has ONE cost. The kernel only ADJUDICATES; which predicates are functional
 *  is the domain's call (foundations: kernel decides, domain supplies semantics). */
export type FunctionalDependency = { predicate: string; key: string[] }

export type FunctionalConflict = {
  predicate: string
  /** the shared key-arg values the conflicting facts agree on. */
  key: Record<string, unknown>
  /** ids of the facts that share the key but disagree on the rest (>= 2). */
  factIds: string[]
  /** their differing atoms, for the message. */
  values: PredicateAtom[]
}

/**
 * Functional-dependency conflicts: facts that share a declared key but disagree
 * on the remaining args cannot both be true. This is exactly the pollution that
 * sank arith p10 — a board-DERIVED cost(item=sensor,total=3886450604850) and a
 * bare-ASSERTED cost(item=sensor,total=388752151850) coexisting, then both summed.
 * Identical facts (same key AND same value, e.g. a re-derivation) are NOT a
 * conflict. Negated facts are left to detectPredicateConflicts.
 */
export function detectFunctionalConflicts(
  facts: PredicateFact[],
  dependencies: FunctionalDependency[],
): FunctionalConflict[] {
  const out: FunctionalConflict[] = []
  for (const dep of dependencies) {
    if (dep.key.length === 0) continue
    const groups = new Map<string, { keyArgs: SemanticArgs; byValue: Map<string, PredicateFact> }>()
    for (const fact of facts) {
      if (fact.atom.predicate !== dep.predicate || fact.atom.negated === true) continue
      const args = fact.atom.args ?? {}
      if (dep.key.some((k) => args[k] === undefined)) continue // missing a key arg → not in scope
      const keyArgs: SemanticArgs = {}
      for (const k of dep.key) keyArgs[k] = args[k]!
      const groupKey = atomKey({ predicate: dep.predicate, args: keyArgs })
      const group = groups.get(groupKey) ?? { keyArgs, byValue: new Map<string, PredicateFact>() }
      // de-dupe by FULL atom: identical (key+value) facts are one value, not a conflict.
      group.byValue.set(atomKey(asPositiveAtom(fact.atom)), fact)
      groups.set(groupKey, group)
    }
    for (const group of groups.values()) {
      if (group.byValue.size <= 1) continue // one agreed value (or none) → no conflict
      const members = [...group.byValue.values()]
      out.push({
        predicate: dep.predicate,
        key: group.keyArgs,
        factIds: members.map((f) => f.id).sort(),
        values: members.map((f) => asPositiveAtom(f.atom)),
      })
    }
  }
  return out
}
