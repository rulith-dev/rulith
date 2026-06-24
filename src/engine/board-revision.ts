import type { SpaceNode } from '../model/types.js'
import { logicallyUsableNodes } from './semantic-active.js'

/**
 * A deterministic fingerprint of the simulation-relevant board state (C4).
 *
 * simulate computes a preview against the world as it is NOW; apply may run
 * later, after the board moved, so the preview can be stale (the C4 gap:
 * "simulate -> apply has no consistency guarantee"). The revision lets a
 * caller pin the world it reasoned about: pass simulate's boardRevision to
 * apply, and a changed board is caught instead of silently diverging.
 *
 * It fingerprints every LOGICALLY USABLE node (facts, rules, actions, goals,
 * hypotheses) - id + type + status + semantic payload - because each can
 * change a simulation's outcome. Derived facts are excluded: they are a pure
 * function of the EDB + rules already in the fingerprint, so including them
 * would be redundant (and order-sensitive). Stable across key order via a
 * sorted, canonical serialization; compact via FNV-1a.
 */
export function boardRevision(nodes: SpaceNode[]): string {
  const usable = logicallyUsableNodes(nodes).filter((node) => !node.id.startsWith('derived:'))
  const lines = usable
    .map((node) => `${node.id}${node.type}${node.status}${stableStringify(node.semantic)}`)
    .sort()
  return `rev1:${fnv1a(lines.join(''))}`
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null'
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`
}

/** FNV-1a 32-bit, hex. Not cryptographic - a compact change-detector. */
function fnv1a(text: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}
