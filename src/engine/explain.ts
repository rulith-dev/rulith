/**
 * explain: derivation-trace utility — "why is this fact true?"
 *
 * A derived fact's evidenceRefs carry [ruleId, ...sourceFactIds].
 * An asserted (EDB) fact has evidenceRefs = [] or references only non-rule
 * nodes (e.g. a parent observation). An action-effect fact has
 * evidenceRefs = [actionId].
 *
 * This module walks the evidenceRefs DAG from a given fact node up to its
 * asserted/effect leaves, building a structured ExplanationNode tree.
 * Cycles are theoretically impossible in the stratified closure, but a
 * visited-set guard defends against corrupted stores.
 */

import type { SpaceNode } from '../model/types.js'
import type { SpaceStore } from '../storage/space-store.js'
import { formatAtom } from '../kernel/predicate.js'
import { isDerivedFactId } from '../kernel/stratify.js'
import { isActionEffectFactNode, isDerivedFactNode } from './logic-context.js'

/** The three source markers present on board facts. */
export type FactMarker = 'derived' | 'asserted' | 'effect'

/**
 * One node in the derivation tree.
 *
 * - `atom` — the predicate atom this node represents (formatted label for
 *   non-fact node types that appear as rule anchors).
 * - `marker` — how the fact was produced.
 * - `ruleId` / `ruleLabel` — set when this node was produced by a rule
 *   (derived) or an action (effect). Null for bare assertions.
 * - `parents` — the supporting facts this node rests on. Empty at leaves.
 * - `cycleDetected` — true when we hit a visited id (should not happen in a
 *   healthy store; recorded rather than thrown so the tree is still useful).
 */
export type ExplanationNode = {
  nodeId: string
  atom: string
  marker: FactMarker
  ruleId: string | null
  ruleLabel: string | null
  parents: ExplanationNode[]
  cycleDetected?: true
}

/**
 * Build a derivation tree for the fact identified by `factNodeId`.
 *
 * Throws a teaching error when the node id is unknown or is not a fact node.
 * Cycle-guards with a `visited` set threaded through recursive calls.
 */
export function explain(
  store: SpaceStore,
  spaceId: string,
  factNodeId: string,
): ExplanationNode {
  // Verify the space exists (throws if not).
  store.getSpace(spaceId)

  let node: SpaceNode
  try {
    node = store.getNode(spaceId, factNodeId)
  } catch {
    throw new Error(
      `explain: unknown node id "${factNodeId}" in space "${spaceId}". ` +
        `Check the id against the board (logic_context facts/findings lists) ` +
        `and pass a node id that exists.`,
    )
  }

  if (node.type !== 'fact') {
    throw new Error(
      `explain: node "${factNodeId}" has type "${node.type}", not "fact". ` +
        `explain only traces predicate fact nodes. ` +
        `For goals/axioms/results, inspect their evidenceRefs directly.`,
    )
  }

  return explainNode(store, spaceId, node, new Set<string>())
}

function explainNode(
  store: SpaceStore,
  spaceId: string,
  node: SpaceNode,
  visited: Set<string>,
): ExplanationNode {
  const marker = markerOf(node)

  // Cycle guard — should not occur in a healthy stratified closure.
  if (visited.has(node.id)) {
    return {
      nodeId: node.id,
      atom: atomLabel(node),
      marker,
      ruleId: null,
      ruleLabel: null,
      parents: [],
      cycleDetected: true,
    }
  }
  visited.add(node.id)

  // For derived facts: evidenceRefs = [ruleId, ...sourceFactIds].
  // For effect facts: evidenceRefs = [actionId].
  // For asserted facts: evidenceRefs = [] or arbitrary provenance refs
  //   (not guaranteed to be fact nodes themselves).
  if (marker === 'derived') {
    const [ruleId, ...sourceIds] = node.evidenceRefs
    const { ruleLabel } = resolveRuleLabel(store, spaceId, ruleId)
    const parents = sourceIds
      .map((sid) => resolveParent(store, spaceId, sid, visited))
      .filter((p): p is ExplanationNode => p !== null)
    visited.delete(node.id) // allow same node to appear in independent branches
    return {
      nodeId: node.id,
      atom: atomLabel(node),
      marker,
      ruleId: ruleId ?? null,
      ruleLabel,
      parents,
    }
  }

  if (marker === 'effect') {
    const [actionId] = node.evidenceRefs
    const { ruleLabel } = resolveRuleLabel(store, spaceId, actionId)
    visited.delete(node.id)
    return {
      nodeId: node.id,
      atom: atomLabel(node),
      marker,
      ruleId: actionId ?? null,
      ruleLabel,
      parents: [],
    }
  }

  // Asserted leaf — no derivation to walk.
  visited.delete(node.id)
  return {
    nodeId: node.id,
    atom: atomLabel(node),
    marker,
    ruleId: null,
    ruleLabel: null,
    parents: [],
  }
}

/**
 * Attempt to resolve a parent node by id. Returns null (silently) for
 * non-fact evidenceRefs (e.g. a rule id, an external source string) that
 * do not correspond to a fact in the space — the derivation tree is still
 * valid, those refs simply don't expand further.
 */
function resolveParent(
  store: SpaceStore,
  spaceId: string,
  nodeId: string,
  visited: Set<string>,
): ExplanationNode | null {
  let parent: SpaceNode
  try {
    parent = store.getNode(spaceId, nodeId)
  } catch {
    // evidenceRef points outside this space or to an external source id —
    // not walkable, skip silently.
    return null
  }
  if (parent.type !== 'fact') {
    // e.g. a rule axiom node referenced in evidenceRefs — not a fact leaf.
    return null
  }
  return explainNode(store, spaceId, parent, visited)
}

function resolveRuleLabel(
  store: SpaceStore,
  spaceId: string,
  nodeId: string | undefined,
): { ruleLabel: string | null } {
  if (!nodeId) return { ruleLabel: null }
  try {
    const n = store.getNode(spaceId, nodeId)
    return { ruleLabel: n.label ?? null }
  } catch {
    return { ruleLabel: null }
  }
}

function markerOf(node: SpaceNode): FactMarker {
  if (isDerivedFactNode(node)) return 'derived'
  if (isActionEffectFactNode(node)) return 'effect'
  return 'asserted'
}

function atomLabel(node: SpaceNode): string {
  if (node.semantic?.kind === 'predicate' && node.semantic.predicate) {
    return formatAtom({
      predicate: node.semantic.predicate,
      args: node.semantic.args,
      negated: node.semantic.negated,
    })
  }
  return node.label
}

/**
 * Render a derivation tree as an indented human-readable trace.
 *
 * Example output:
 *   finding(kind:issue, line:7) [derived by AX1 "finding from obs"]
 *     <- obs(line:7) [asserted]
 */
export function formatExplanation(node: ExplanationNode, depth = 0): string {
  const indent = '  '.repeat(depth)
  const rulePart =
    node.marker === 'derived' || node.marker === 'effect'
      ? node.ruleId !== null
        ? ` by ${node.ruleId}${node.ruleLabel ? ` "${node.ruleLabel}"` : ''}`
        : ''
      : ''
  const markerText =
    node.marker === 'derived'
      ? `[derived${rulePart}]`
      : node.marker === 'effect'
        ? `[effect${rulePart}]`
        : '[asserted]'
  const cycleSuffix = node.cycleDetected ? ' ⚠ cycle detected' : ''
  const header = `${indent}${node.atom} (${node.nodeId}) ${markerText}${cycleSuffix}`

  if (node.parents.length === 0) return header

  const childLines = node.parents
    .map((parent) => formatExplanation(parent, depth + 1))
    .join('\n')
  return `${header}\n${childLines}`
}
