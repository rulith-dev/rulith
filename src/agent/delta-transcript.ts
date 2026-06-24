/**
 * delta-transcript — the 'delta-history' transcript strategy (docs/product.md §3).
 *
 * History turns are rendered as their COMPLETE delta from the previous turn
 * (+added / ~changed / -removed, every change listed, consumed/archived facts
 * shown gone); the LATEST turn is rendered FULL — the authoritative current
 * state — so the model never has to integrate the deltas itself (avoids the
 * pure-delta "③" cognitive-drift trap). History deltas keep the trajectory
 * cheaply (avoids windowing "②" dropping it). Space ≈ O(board + total churn).
 *
 * Soundness (replay invariant, §5): replaying the history deltas from the
 * initial snapshot reconstructs the latest full board — delta-history is a
 * lossless, reversible decomposition of the latest full board along time, NOT a
 * second source of truth. `replayDeltas` exposes this for the loop + red tests.
 *
 * Reuses board-diff (diffFacts/factKind) verbatim and does NOT touch it — the
 * existing `highlight` bench arm (which uses the truncating formatDelta as a
 * pre-pended summary while the full board is still present) is unchanged. Here
 * the delta IS the history body, so it must be complete, not a preview.
 */
import { diffFacts, factKind } from '../engine/board-diff.js'
import { formatAtom } from '../kernel/predicate.js'
import type { LogicContextFact } from '../engine/logic-context.js'

export type TurnSnapshot = readonly LogicContextFact[]

// ③ format isomorphism: render a delta line's atom with the SAME formatAtom the full board
// uses (predicate(key=value)), so a history delta (+observed(item=x)) and the tail full board
// (observed(item=x)) read in ONE format - not the old simplified observed(x) the model had to
// reconcile against the full render. Slightly longer per delta line, but one grammar to parse.
function atomStr(f: LogicContextFact): string {
  return formatAtom(f.atom)
}

/**
 * Complete (non-truncating) delta text for one history turn. Lossless so the
 * replay invariant holds at the text layer too: +added / ~changed / -removed,
 * every change listed; a consumed/retracted fact is shown `[gone]` with the
 * atom it used to be (negative-delta faithfulness, §6 — never silently dropped).
 */
export function renderHistoryDelta(prev: TurnSnapshot | undefined, curr: TurnSnapshot): string {
  const d = diffFacts(prev, curr)
  const prevById = new Map((prev ?? []).map((f) => [f.nodeId, f] as const))
  const lines: string[] = []
  for (const f of d.added) lines.push(`+${atomStr(f)} [${factKind(f)}]`)
  for (const f of d.changed) lines.push(`~${atomStr(f)} [${factKind(f)}]`)
  for (const id of d.removedIds) {
    const gone = prevById.get(id)
    lines.push(`-${gone ? atomStr(gone) : id} [gone]`)
  }
  return lines.length > 0 ? `Δ turn:\n${lines.join('\n')}` : 'Δ turn: (nothing changed)'
}

/**
 * Transcript bodies for the 'delta-history' strategy: history turns as their
 * complete delta, the latest turn FULL (+ delta highlight set for the renderer).
 * `renderFull` is injected so the live loop can pass formatLogicContextAsText
 * while tests pass a simple stand-in.
 */
export function renderDeltaHistory(
  snapshots: readonly TurnSnapshot[],
  renderFull: (facts: TurnSnapshot, highlight: ReadonlySet<string>) => string,
): string[] {
  const bodies: string[] = []
  for (let i = 0; i < snapshots.length; i += 1) {
    const prev = i === 0 ? undefined : snapshots[i - 1]
    const curr = snapshots[i]!
    if (i === snapshots.length - 1) {
      bodies.push(renderFull(curr, diffFacts(prev, curr).highlight))
    } else {
      bodies.push(renderHistoryDelta(prev, curr))
    }
  }
  return bodies
}

/**
 * Replay the structural deltas from the initial snapshot — the soundness check
 * behind delta-history (§5). Returns the reconstructed latest board by nodeId;
 * the loop/red test asserts this equals the actual latest snapshot.
 */
export function replayDeltas(snapshots: readonly TurnSnapshot[]): Map<string, LogicContextFact> {
  const acc = new Map<string, LogicContextFact>()
  for (let i = 0; i < snapshots.length; i += 1) {
    const prev = i === 0 ? undefined : snapshots[i - 1]
    const d = diffFacts(prev, snapshots[i]!)
    for (const f of d.added) acc.set(f.nodeId, f)
    for (const f of d.changed) acc.set(f.nodeId, f)
    for (const id of d.removedIds) acc.delete(id)
  }
  return acc
}
