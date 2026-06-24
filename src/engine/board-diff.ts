import type { LogicContextFact } from './logic-context.js'

/**
 * board-diff — compute what CHANGED on the board since the model's last turn, so
 * the rendered view can HIGHLIGHT the delta instead of re-showing an
 * undifferentiated wall.
 *
 * Why this matters (measured, 2026-06): the board view is re-sent every turn
 * (O(N)); a weak / non-thinking driver re-reads all of it and often cannot tell
 * whether its last op actually landed. The #1 driving failure is an EMPTY board —
 * the op added nothing — which the model never notices, then loops blind (a3b: 9/10
 * empty boards on hard arith). A "Δ since your last op" line, and especially a loud
 * "NOTHING CHANGED" when an op produced nothing, gives the driver the feedback it
 * needs to fix/retry instead of looping.
 *
 * This module is the PURE SEMANTICS of the delta ("what is new / changed / gone").
 * Rendering it into the view (logic-context / task-loop) is the shell layer's job;
 * keeping it a pure function makes it testable and lets the view annotate however
 * it likes. Pairs with a `view: plain | highlight` bench arm: default plain (so all
 * prior P5 data stays the control), highlight measured as the delta against it.
 */

export type FactKind = 'derived' | 'effect' | 'asserted'

/** Provenance kind of a fact: closure-derived, action-effect, or bare assertion. */
export function factKind(f: LogicContextFact): FactKind {
  return f.derived ? 'derived' : f.effect ? 'effect' : 'asserted'
}

export type BoardDelta = {
  /** nodeId absent last turn — produced by the last op. */
  added: LogicContextFact[]
  /** same nodeId, but atom / derived / effect / status changed (revised, or newly derived). */
  changed: LogicContextFact[]
  /** nodeId present last turn, gone now — consumed / archived / retracted. */
  removedIds: string[]
  /** added ∪ changed nodeIds: what the view should mark fresh. */
  highlight: Set<string>
}

/** Identity-independent signature; a change to any of these fields = "changed". */
function signature(f: LogicContextFact): string {
  return JSON.stringify([f.atom.predicate, f.atom.args ?? null, f.derived, f.effect ?? false, f.status])
}

/**
 * Diff two snapshots of a board's facts (or findings — same node shape) by nodeId.
 * `prev === undefined` = the first turn (everything is "added"; callers usually
 * suppress the rendered line on turn 1 via formatDelta's `firstTurn`).
 */
export function diffFacts(
  prev: readonly LogicContextFact[] | undefined,
  curr: readonly LogicContextFact[],
): BoardDelta {
  const prevById = new Map((prev ?? []).map((f) => [f.nodeId, f] as const))
  const currIds = new Set(curr.map((f) => f.nodeId))
  const added: LogicContextFact[] = []
  const changed: LogicContextFact[] = []
  for (const f of curr) {
    const before = prevById.get(f.nodeId)
    if (before === undefined) added.push(f)
    else if (signature(before) !== signature(f)) changed.push(f)
  }
  const removedIds = [...prevById.keys()].filter((id) => !currIds.has(id))
  const highlight = new Set<string>([...added, ...changed].map((f) => f.nodeId))
  return { added, changed, removedIds, highlight }
}

function shortArgs(args: Record<string, unknown> | undefined): string {
  if (!args) return ''
  const joined = Object.values(args).map((v) => String(v)).join(',')
  return `(${joined.length > 40 ? `${joined.slice(0, 40)}…` : joined})`
}

function preview(facts: LogicContextFact[], n = 6): string {
  const shown = facts.slice(0, n).map((f) => `${f.atom.predicate}${shortArgs(f.atom.args)} [${factKind(f)}]`)
  return shown.join(', ') + (facts.length > n ? `, +${facts.length - n} more` : '')
}

/**
 * A compact "Δ since your last op" line to inject ABOVE the board view. The
 * crucial case is the EMPTY delta: it tells a weak driver its last op produced
 * nothing (the empty-board failure) so it fixes/retries instead of looping blind.
 * It also calls out whether the new facts are DERIVED (earned) or only asserted —
 * the distinction the done-gate cares about.
 */
export function formatDelta(delta: BoardDelta, opts: { firstTurn?: boolean } = {}): string {
  if (opts.firstTurn) return ''
  const { added, changed, removedIds } = delta
  if (added.length === 0 && changed.length === 0 && removedIds.length === 0) {
    return (
      'Δ since your last op: NOTHING CHANGED — your last operation put no new facts on the board. ' +
      'Likely a malformed op, a wrong predicate/args, or a rule that did not fire. Inspect and retry; do not repeat the same call.'
    )
  }
  const parts: string[] = []
  if (added.length > 0) parts.push(`+${added.length} new: ${preview(added)}`)
  if (changed.length > 0) parts.push(`~${changed.length} changed: ${preview(changed)}`)
  if (removedIds.length > 0) parts.push(`-${removedIds.length} removed (consumed/retracted)`)
  const derivedNew = added.filter((f) => f.derived).length
  const earned =
    derivedNew > 0
      ? ` (${derivedNew} of the new facts are DERIVED — closure-earned, not just asserted.)`
      : added.length > 0
        ? ' (none of the new facts are derived yet — only assertions; the gate will not certify assertions.)'
        : ''
  return `Δ since your last op: ${parts.join('; ')}.${earned}`
}
