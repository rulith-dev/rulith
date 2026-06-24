/**
 * evidence-policy — high-value-task evidence floors as BOARD POLICY (A2 / theory §5 #7, §7 "trust inversion").
 *
 * The grounding floor (premise-provenance) reports how trustworthy a completion IS; this layer declares
 * how trustworthy it must BE for a given obligation, and surfaces the gap. Two policy sources, both board
 * facts (declared, not hardcoded in a prompt):
 *   - `required_floor(scope, tier)` — the minimum acceptable evidence tier for an obligation. `scope` is a
 *     node / slot / constraint tag, or `*` / `task` for a global floor.
 *   - `constraint_tier(tag, human)` — a human-judged constraint: its `constraint_met` witness must be at
 *     least `attested` (a model-asserted constraint_met does NOT satisfy a human-tier constraint). This is
 *     what finally makes `constraint_tier` non-inert (it was declared but read by no rule before).
 *
 * A TRUST INVERSION = a satisfied obligation whose actual grounding floor is WEAKER than its required tier
 * (high stakes leaning on flimsy evidence). `trustInversions` is a CRITIQUE: it returns the gaps; it does
 * NOT mutate `receipt.closed`. The policy gate / UI decides whether to block — keeping the "expose, don't
 * silently launder" discipline (foundations.md) and the orthogonal axes (structure vs grounding) intact.
 */
import { groundingOf, TIER_ORDER, type PremiseTier, type GroundingOptions } from '../engine/premise-provenance.js'
import type { LogicContextFact } from '../engine/logic-context.js'

const tierRank = (t: PremiseTier): number => TIER_ORDER.indexOf(t)
const isTier = (t: string): t is PremiseTier => (TIER_ORDER as readonly string[]).includes(t)
const strArg = (f: LogicContextFact, k: string): string | undefined => {
  const v = (f.atom.args as Record<string, unknown> | undefined)?.[k]
  return v === undefined || v === null ? undefined : String(v)
}

/** A POLICY fact (required_floor / constraint_tier) only counts if it entered through a TRUSTED channel.
 *  Otherwise the model could set its own evidence bar — assert a lax `required_floor` to dodge the very
 *  check this module exists to make. Policy is host/human/system territory, not the model's free word
 *  (same discipline as committed_* baselines and the trusted clock). */
const TRUSTED_POLICY_CREATORS = new Set(['system', 'tool'])
const isTrustedPolicy = (f: LogicContextFact): boolean => {
  const createdBy = (f as { createdBy?: string }).createdBy
  const trustTier = (f as { trustTier?: PremiseTier }).trustTier
  return (createdBy !== undefined && TRUSTED_POLICY_CREATORS.has(createdBy)) || trustTier === 'verified' || trustTier === 'attested'
}

export type TrustInversion = {
  /** the satisfied-obligation witness whose evidence is too weak. */
  witnessId: string
  /** the obligation it covers (node / slot / `node/tag`). */
  scope: string
  /** the minimum tier policy demands. */
  requiredTier: PremiseTier
  /** the actual grounding floor of the witness (weaker than requiredTier). */
  actualTier: PremiseTier
  source: 'required_floor' | 'human_constraint'
}

/**
 * Trust inversions on the board: satisfied obligations grounded WEAKER than policy requires.
 * Pure read over board facts (no closure of its own); a non-empty result is a structured critique.
 */
export function trustInversions(facts: LogicContextFact[], options: GroundingOptions = {}): TrustInversion[] {
  // 1. explicit policy: required_floor(scope, tier). scope = node / slot / tag, or '*'/'task' for global.
  const requiredByScope = new Map<string, PremiseTier>()
  for (const f of facts) {
    if (f.atom.predicate !== 'required_floor' || !isTrustedPolicy(f)) continue // untrusted policy is ignored
    const scope = strArg(f, 'scope')
    const tier = strArg(f, 'tier')
    if (!scope || !tier || !isTier(tier)) continue
    // STRICTEST wins, order-INDEPENDENT: keep the strongest required tier (lowest TIER_ORDER index).
    // A laxer trusted floor must not overwrite a stricter one just by being written later — lowering
    // the bar requires an explicit revise/retract of the strict floor, never write order (Codex P2).
    const existing = requiredByScope.get(scope)
    if (existing === undefined || tierRank(tier) < tierRank(existing)) requiredByScope.set(scope, tier)
  }
  const globalRequired = requiredByScope.get('*') ?? requiredByScope.get('task')

  // 2. human-judged constraints: a human-tier constraint demands at least `attested` evidence —
  //    a model-asserted constraint_met does not satisfy it. (This reads constraint_tier, making it non-inert.)
  //    The tier DECLARATION must itself be trusted, else the model could relabel a human constraint 'machine'.
  const humanTags = new Set<string>()
  for (const f of facts) {
    if (f.atom.predicate === 'constraint_tier' && strArg(f, 'tier') === 'human' && isTrustedPolicy(f)) {
      const tag = strArg(f, 'tag')
      if (tag) humanTags.add(tag)
    }
  }

  const out: TrustInversion[] = []
  const seen = new Set<string>()
  const consider = (witnessId: string, scope: string, required: PremiseTier | undefined, source: TrustInversion['source']): void => {
    if (!required) return
    const actualTier = groundingOf(facts, witnessId, options).weakestTier
    if (tierRank(actualTier) <= tierRank(required)) return // actual is at least as strong as required
    const key = `${witnessId}|${source}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({ witnessId, scope, requiredTier: required, actualTier, source })
  }

  for (const f of facts) {
    const p = f.atom.predicate
    if (p === 'acceptance_met') {
      const node = strArg(f, 'node')
      if (node) consider(f.nodeId, node, requiredByScope.get(node) ?? globalRequired, 'required_floor')
    } else if (p === 'slot_met') {
      const slot = strArg(f, 'slot')
      if (slot) consider(f.nodeId, slot, requiredByScope.get(slot) ?? globalRequired, 'required_floor')
    } else if (p === 'constraint_met') {
      const node = strArg(f, 'node') ?? ''
      const tag = strArg(f, 'tag')
      if (!tag) continue
      const scope = `${node}/${tag}`
      if (humanTags.has(tag)) consider(f.nodeId, scope, 'attested', 'human_constraint')
      const explicit = requiredByScope.get(tag) ?? globalRequired
      if (explicit) consider(f.nodeId, scope, explicit, 'required_floor')
    }
  }
  return out.sort((a, b) => a.witnessId.localeCompare(b.witnessId) || a.source.localeCompare(b.source))
}

/**
 * Render trust inversions as a model-facing critique block — this is what the execution loop
 * surfaces in the board view so the model can act on it (strengthen the evidence). Empty when
 * there are none, so a board with no policy / no inversion sees no change at all.
 */
export function formatTrustInversions(inversions: TrustInversion[]): string {
  if (inversions.length === 0) return ''
  return [
    'trust_inversions:',
    ...inversions.map(
      (t) =>
        `- [${t.source}] ${t.scope}: grounded only at "${t.actualTier}", policy requires "${t.requiredTier}" — strengthen the evidence (it cannot honestly close on this)`,
    ),
  ].join('\n')
}
