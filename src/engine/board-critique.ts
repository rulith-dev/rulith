import { detectVacuousRule, detectUnfirableRule } from '../kernel/rule-quality.js'
import { isBuiltinPredicate } from '../kernel/builtins.js'
import { atomKey } from '../kernel/predicate.js'
import type { LogicContext, LogicContextGoal, LogicContextFact } from './logic-context.js'

/**
 * A standing board-health item. Unlike the per-apply teaching warnings
 * (unfirable rule, self-recursive arithmetic, self-sealed goal) - which fire
 * ONCE, on the batch that introduced them - a critique is re-derived from
 * board STATE every time the context is read. A problem the model left on
 * the board therefore stays visible turn after turn until fixed: the
 * "failures become learnable" half of the agent mandate.
 */
export type BoardCritique = {
  kind:
    | 'self_sealed_goal'
    | 'asserted_finding'
    | 'vacuous_rule'
    | 'unfirable_rule'
    | 'contradiction'
    | 'unsatisfiable_action'
    | 'unreachable_goal'
    | 'conflicting_goals'
    | 'unexplained_reopen'
    | 'functional_conflict'
  nodeId: string
  message: string
}

/** A grant that vanished without a trusted `revocation_result`, while its (trusted)
 *  `approval_result(granted:true)` still stands — an unexplained reopen (I6.R6d).
 *  Pure over facts; re-exported by src/agent/reopen-audit.ts for agent-side consumers. */
export type UnexplainedReopen = { action: string; approvalResultId: string; reason: string }

const REOPEN_TRUSTED = new Set(['tool', 'system'])
const reopenStr = (f: LogicContextFact, k: string): string | undefined => {
  const v = (f.atom.args as Record<string, unknown> | undefined)?.[k]
  return typeof v === 'string' && v.trim() ? v : undefined
}
const reopenTrustedHas = (facts: LogicContextFact[], predicate: string, action: string): boolean =>
  facts.some((f) => f.atom.predicate === predicate && reopenStr(f, 'action') === action && REOPEN_TRUSTED.has(f.createdBy))

export function unexplainedReopens(facts: LogicContextFact[]): UnexplainedReopen[] {
  const out: UnexplainedReopen[] = []
  for (const ar of facts) {
    if (ar.atom.predicate !== 'approval_result' || !REOPEN_TRUSTED.has(ar.createdBy)) continue
    if ((ar.atom.args as Record<string, unknown> | undefined)?.granted !== true) continue
    const action = reopenStr(ar, 'action')
    if (!action) continue
    if (reopenTrustedHas(facts, 'permission_granted', action)) continue // a real grant still present → fine
    if (reopenTrustedHas(facts, 'revocation_result', action)) continue // a real revocation explains it → fine
    out.push({
      action,
      approvalResultId: ar.nodeId,
      reason: `permission grant for "${action}" disappeared without a revocation_result — unexplained reopen`,
    })
  }
  return out
}

/**
 * Re-derive the standing problems on a board from its already-built context.
 * Pure over LogicContext, so it reuses the provenance the context computed
 * (goal.selfSealed, fact.derived) instead of re-walking nodes.
 */
export function critiqueBoard(context: LogicContext): BoardCritique[] {
  const items: BoardCritique[] = []

  // 1. A goal "satisfied" only by a bare assertion the model wrote.
  for (const goal of context.goals) {
    if (goal.selfSealed) {
      items.push({
        kind: 'self_sealed_goal',
        nodeId: goal.nodeId,
        message:
          `goal "${goal.label}" reads satisfied but nothing derives it - it rests on a bare ` +
          `assertion you wrote. Add a rule that derives it from more primitive facts, or treat ` +
          `it as an unproven note.`,
      })
    }
  }

  // 2. A finding(...) that is not closure-derived: a conclusion asserted
  //    rather than earned. The record_result gate blocks the obvious route;
  //    this catches one that reached the board another way and still stands.
  for (const finding of context.findings) {
    if (!finding.derived && !finding.effect) {
      items.push({
        kind: 'asserted_finding',
        nodeId: finding.nodeId,
        message:
          `finding "${finding.atom.predicate}(...)" is asserted, not derived. A finding should ` +
          `be the head of a rule whose body holds - add that rule so the closure stands behind it.`,
      })
    }
  }

  // 3. A rule whose body merely renames its conclusion (no filtering power).
  for (const axiom of context.axioms) {
    const vacuous = detectVacuousRule({ id: axiom.nodeId, when: axiom.when, then: axiom.then })
    if (vacuous) {
      items.push({ kind: 'vacuous_rule', nodeId: axiom.nodeId, message: vacuous })
    }
    // A rule whose body can provably never be satisfied — dead, fires nothing.
    // The rule analog of the unsatisfiable_action item below (and NOT
    // done-blocking: dead code is surfaced, not a correctness fault).
    const unfirable = detectUnfirableRule({ id: axiom.nodeId, when: axiom.when, then: axiom.then })
    if (unfirable) {
      items.push({ kind: 'unfirable_rule', nodeId: axiom.nodeId, message: unfirable })
    }
  }

  // 4. p AND not-p both standing on the board (paraconsistent taint).
  //    Reuses the conflicts the context already detected - one critique per
  //    contradicting pair, anchored on the positive fact.
  for (const conflict of context.predicateConflicts) {
    const args = conflict.atom.args
      ? Object.entries(conflict.atom.args).map(([key, value]) => `${key}=${value}`).join(', ')
      : ''
    const positive = `${conflict.atom.predicate}(${args})`
    items.push({
      kind: 'contradiction',
      nodeId: conflict.positiveFactId,
      message:
        `contradiction: ${conflict.positiveFactId} asserts "${positive}" while ` +
        `${conflict.negativeFactId} asserts "not ${positive}" - both stand at once. ` +
        `Anything whose evidence touches "${positive}" is tainted (disputed) and cannot be trusted. ` +
        `Resolve it: retract whichever of ${conflict.positiveFactId} / ${conflict.negativeFactId} ` +
        `is the bare assertion, or - if one is closure-derived - fix the rule that produces it.`,
    })
  }

  // 4b. Functional-dependency conflict: same declared key, different value (e.g. a derived
  //     cost and a bare-asserted cost for one item). Same paraconsistent-taint discipline as 4.
  for (const conflict of context.functionalConflicts) {
    const keyStr = Object.entries(conflict.key).map(([key, value]) => `${key}=${value}`).join(', ')
    items.push({
      kind: 'functional_conflict',
      nodeId: conflict.factIds[0]!,
      message:
        `functional conflict: ${conflict.predicate}(${keyStr}) has ${conflict.factIds.length} disagreeing ` +
        `values [${conflict.factIds.join(', ')}] - the declared key fixes one value, so these cannot all ` +
        `be true (usually a bare assertion contradicting a closure-derivation, or a stale value). Anything ` +
        `summing or relying on them is tainted (disputed). Resolve it: retract the wrong one - usually the ` +
        `bare assertion that should have been left to the rule.`,
    })
  }

  // 5. An action whose precondition references a predicate that NOTHING on
  //    the board can supply - no fact has it, no rule head derives it, no
  //    other action's effect produces it. The action is dead: it can never
  //    become applicable, so a model gating on it waits forever. The action
  //    analog of the unfirable-rule warning, surfaced as a STANDING item.
  const suppliable = new Set<string>()
  for (const fact of context.facts) suppliable.add(fact.atom.predicate)
  for (const finding of context.findings) suppliable.add(finding.atom.predicate)
  for (const axiom of context.axioms) for (const head of axiom.then) suppliable.add(head.predicate)
  for (const action of context.actions) {
    for (const effect of action.effects) {
      if (effect.negated !== true) suppliable.add(effect.predicate)
    }
  }
  for (const action of context.actions) {
    for (const pre of action.preconditions) {
      if (pre.naf === true) continue
      if (isBuiltinPredicate(pre.predicate)) continue
      if (suppliable.has(pre.predicate)) continue
      items.push({
        kind: 'unsatisfiable_action',
        nodeId: action.nodeId,
        message:
          `action "${action.action}" can never become applicable: its precondition ` +
          `"${pre.predicate}(...)" is supplied by nothing on the board - no fact asserts it, no rule ` +
          `derives it, no action produces it. Add a rule whose "then" yields ${pre.predicate}(...), ` +
          `assert it as a fact, or fix the precondition predicate name.`,
      })
    }
  }

  // 6. An open goal the board carries no machinery to close: NO rule head
  //    derives its desired predicate (zero abduction hints) AND no action
  //    effect produces it (zero action hints). The model declared a target
  //    but built no inference path or producing action toward it, so it sits
  //    open forever. The standing form of the per-turn "no rule derives this
  //    yet" guidance - it persists until the model adds the missing machinery
  //    (or, for a primitive observation, asserts the fact directly).
  for (const goal of context.goals) {
    if (goal.satisfied) continue
    if (goal.hints.length > 0 || goal.actionHints.length > 0) continue
    const wanted = goal.desired.map((atom) => atom.predicate).join(', ') || goal.label
    items.push({
      kind: 'unreachable_goal',
      nodeId: goal.nodeId,
      message:
        `goal "${goal.label}" is open and the board carries no way to close it: no rule derives it ` +
        `and no action produces "${wanted}". Add an add_axiom whose "then" matches the desired atom, ` +
        `define_action whose effect produces it, or - if it is a primitive observation - assert the ` +
        `fact directly.`,
    })
  }

  // 7. Two declared goals whose desired atoms directly contradict each other:
  //    one desires p(args) and the other desires the strong-negation of p(args)
  //    with the SAME predicate and SAME args. No plan can satisfy both. Emit one
  //    critique per conflicting pair, anchored on the goal that appears first.
  //    Scope: strong negation (negated: true) only. NAF is a rule-body guard,
  //    not a goal desire - it does not appear on declared goal desired atoms in
  //    practice and would have no planning-level meaning here.
  {
    // For each desired atom, build a map from base key (predicate+args, polarity-
    // stripped) to list of { goal, positiveKey } so we can match opposite poles.
    type DesiredEntry = { goal: LogicContextGoal; positiveKey: string; negated: boolean }
    const byBase = new Map<string, DesiredEntry[]>()
    for (const goal of context.goals) {
      for (const atom of goal.desired) {
        // Base key: same predicate+args regardless of polarity.
        const positiveKey = atomKey({ predicate: atom.predicate, args: atom.args })
        const entries = byBase.get(positiveKey) ?? []
        entries.push({ goal, positiveKey, negated: atom.negated === true })
        byBase.set(positiveKey, entries)
      }
    }
    // Pairs already emitted (to avoid double-reporting A↔B and B↔A).
    const emitted = new Set<string>()
    for (const entries of byBase.values()) {
      const positives = entries.filter((e) => !e.negated)
      const negatives = entries.filter((e) => e.negated)
      for (const pos of positives) {
        for (const neg of negatives) {
          if (pos.goal.nodeId === neg.goal.nodeId) continue
          const pairKey =
            pos.goal.nodeId < neg.goal.nodeId
              ? `${pos.goal.nodeId}:${neg.goal.nodeId}:${pos.positiveKey}`
              : `${neg.goal.nodeId}:${pos.goal.nodeId}:${pos.positiveKey}`
          if (emitted.has(pairKey)) continue
          emitted.add(pairKey)
          // Format the atom as predicate(k=v, ...) for the message.
          const args = pos.positiveKey.includes('|')
            ? pos.positiveKey
                .slice(pos.positiveKey.indexOf('|') + 1)
                .split('|')
                .filter(Boolean)
                .map((kv) => {
                  const colon = kv.indexOf(':')
                  return colon >= 0
                    ? `${kv.slice(0, colon)}=${JSON.parse(kv.slice(colon + 1))}`
                    : kv
                })
                .join(', ')
            : ''
          const atomStr = `${pos.goal.desired.find((a) => atomKey({ predicate: a.predicate, args: a.args }) === pos.positiveKey)?.predicate ?? '?'}(${args})`
          items.push({
            kind: 'conflicting_goals',
            nodeId: pos.goal.nodeId,
            message:
              `conflicting goals: goal "${pos.goal.label}" (${pos.goal.nodeId}) desires "${atomStr}" ` +
              `while goal "${neg.goal.label}" (${neg.goal.nodeId}) desires the strong negation "not ${atomStr}". ` +
              `No plan can satisfy both. Reconcile: retract one goal, change its desired atom, or ` +
              `split into mutually exclusive planning branches.`,
          })
        }
      }
    }
  }

  // I6.R6d: a grant that vanished without a trusted revocation_result (its approval still stands)
  for (const r of unexplainedReopens(context.facts)) {
    items.push({ kind: 'unexplained_reopen', nodeId: r.approvalResultId, message: r.reason })
  }

  return items
}

/**
 * The HARD board-health blockers: a conclusion claimed, not earned.
 *   - self_sealed_goal: a goal satisfied only by a bare assertion.
 *   - asserted_finding: a finding placed directly, not closure-derived.
 *   - vacuous_rule: a body that just renames the head (a zero-power rule that
 *     launders a claim into a "derived" fact, dodging the self_sealed check).
 * This is the SAME set the task-loop's done-gate (validateDone) treats as
 * blocking — kept in lockstep so a stage gate and a done-gate agree on what
 * counts as unearned. Borrowed from Codex's branch concept (isDoneBlockingCritique);
 * co-authored when adopting the idea. The other critique kinds
 * (contradiction / unsatisfiable_action / unreachable_goal / conflicting_goals)
 * are standing diagnostics, not hard finish-blockers.
 */
export function isDoneBlockingCritique(item: BoardCritique): boolean {
  return (
    item.kind === 'self_sealed_goal' ||
    item.kind === 'asserted_finding' ||
    item.kind === 'vacuous_rule'
  )
}

/** The subset of critique items that hard-block a finish (see isDoneBlockingCritique). */
export function doneBlockingCritiques(items: BoardCritique[]): BoardCritique[] {
  return items.filter(isDoneBlockingCritique)
}

/** Render the critique as board-text lines (empty when the board is healthy). */
export function formatCritique(items: BoardCritique[]): string[] {
  if (items.length === 0) return []
  return ['critique:', ...items.map((item) => `- [${item.kind}] ${item.nodeId}: ${item.message}`)]
}
