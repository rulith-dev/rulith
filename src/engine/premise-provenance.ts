/**
 * premise-provenance — the GROUNDING rollup.
 *
 * The board guarantees SOUND, EXACT INFERENCE *from* premises; it cannot
 * guarantee the PREMISES themselves — the ground facts / observations (a
 * weight, a fraud amount) — are true. Those enter from outside logic; the
 * board can verify "amount >= 100000 -> severity high", it cannot conjure the
 * amount. So a derived conclusion is only as trustworthy as its flimsiest
 * load-bearing ground fact.
 *
 * This walks a conclusion's evidence chain (evidenceRefs) down to its ground
 * premises and reports the WEAKEST tier among them, plus which premises sit at
 * that tier — the inputs a human must verify before trusting the verdict.
 *
 * Ground premises sit on an ORDERED TRUST LATTICE (strongest → weakest). The
 * default two tiers are `attested` (entered through a trusted channel — a
 * machine-attested predicate or a tool/system-created fact) and `asserted` (a
 * bare fact the model put on the board — the residual honesty the board can only
 * surface, not eliminate). Heterogeneous external ENGINES tag their outputs with
 * finer tiers via fact.trustTier — a numeric solver's result is `approximate`,
 * an ML prediction `uncertain`, perception `perceived`, a learned rule
 * `inductive`, a verified/DB read `verified`. A conclusion's floor = the WEAKEST
 * tier among its ground premises (the inference step "derived" is walked *past*,
 * never classified). See docs/product.md.
 *
 * The attested-predicate set is a PARAMETER (the domain declares which
 * predicates are machine-vouched), so the kernel stays domain-agnostic — it
 * adjudicates provenance, it does not hardcode what counts as a trusted source.
 */

export type PremiseTier =
  | 'verified' // independently checked / deterministic trusted source (DB read, closure-verified)
  | 'attested' // entered through a trusted channel (tool/system/attested predicate) — default trusted
  | 'approximate' // numeric engine: exact method, float/rounding error
  | 'inductive' // learned/statistical rule — defeasible
  | 'uncertain' // probabilistic / ML prediction — has confidence, no guarantee
  | 'perceived' // perception (OCR/vision) — pre human-check
  | 'asserted' // a bare fact the model put on the board — default residual

/** Strongest → weakest; a conclusion's floor is the weakest (max index) premise tier. */
export const TIER_ORDER: readonly PremiseTier[] = [
  'verified',
  'attested',
  'approximate',
  'inductive',
  'uncertain',
  'perceived',
  'asserted',
]
const tierRank = (t: PremiseTier): number => TIER_ORDER.indexOf(t)
const weaker = (a: PremiseTier, b: PremiseTier): PremiseTier => (tierRank(a) >= tierRank(b) ? a : b)

/** The minimal fact shape grounding needs (LogicContextFact satisfies it). */
export type ProvenanceFact = {
  nodeId: string
  atom: { predicate: string }
  evidenceRefs: string[]
  /** the rule closure stands behind this fact (an inference step). */
  derived: boolean
  /** asserted by applying an action (also an inference step, not a premise). */
  effect?: boolean
  createdBy?: string
  /** Explicit trust tier from the producing engine — overrides the createdBy /
   *  attestedPredicates default. Lets heterogeneous engine outputs carry their
   *  true reliability (numeric→approximate, ml→uncertain, db→verified, …). */
  trustTier?: PremiseTier
  /** Reserved structured provenance: carried for display + future stale/confidence
   *  checks; NOT yet folded into the floor (keep the dimensions separate). */
  sourceKind?: string // 'db' | 'numeric' | 'ml' | 'ocr' | 'model' | …
  confidence?: number // 0..1, for uncertain/ml
  errorBound?: number // for approximate/numeric
  observedAt?: string // timestamp/version, for staleness detection
  inputHash?: string // for cache / reproduction
}

export type Grounding = {
  factId: string
  /** weakest tier among the ground premises this fact rests on (the trust floor). */
  weakestTier: PremiseTier
  /** ground premises AT the weakest tier — the inputs a human must verify first. */
  weakestPremiseIds: string[]
  /** all ground premises grouped by tier. */
  premisesByTier: Partial<Record<PremiseTier, string[]>>
  /** legacy two-tier views, kept for back-compat. */
  assertedPremises: string[]
  attestedPremises: string[]
  /** the conclusion bottoms out ONLY in bare assertions — nothing of a higher tier under it. */
  ungrounded: boolean
}

export type GroundingOptions = { attestedPredicates?: Iterable<string> }

/** Creators meaning "entered through a trusted channel, not the model's free word". */
const TRUSTED_CREATORS = new Set(['system', 'tool'])

/** A ground premise = a fact nothing derives (an input/observation). A
 *  derived/effect fact is an inference STEP — we descend through it to its
 *  sources rather than treating it as a premise. */
function isGround(fact: ProvenanceFact): boolean {
  return !fact.derived && !fact.effect
}

function tierOf(fact: ProvenanceFact, attested: Set<string>): PremiseTier {
  if (fact.trustTier) return fact.trustTier // explicit tier from the producing engine
  if (fact.createdBy && TRUSTED_CREATORS.has(fact.createdBy)) return 'attested'
  if (attested.has(fact.atom.predicate)) return 'attested'
  return 'asserted'
}

/**
 * The grounding of ONE conclusion: walk its evidence chain to the ground
 * premises and report the weakest tier + the premises at issue. Refs that
 * point at non-fact nodes (rules, actions, external sources) are skipped —
 * they are "how", not "what". Cycles are guarded by a visited set.
 */
export function groundingOf(
  facts: ProvenanceFact[],
  factId: string,
  options: GroundingOptions = {},
): Grounding {
  const attested = new Set(options.attestedPredicates ?? [])
  const byId = new Map(facts.map((f) => [f.nodeId, f]))
  const byTier = new Map<PremiseTier, Set<string>>()
  const visited = new Set<string>()
  const walk = (id: string): void => {
    if (visited.has(id)) return
    visited.add(id)
    const fact = byId.get(id)
    if (!fact) return // a ref to a rule/action/external source — not a fact premise
    if (isGround(fact)) {
      const t = tierOf(fact, attested)
      if (!byTier.has(t)) byTier.set(t, new Set())
      byTier.get(t)!.add(id)
      return
    }
    for (const ref of fact.evidenceRefs) walk(ref) // descend the inference step
  }
  walk(factId)

  const premisesByTier: Partial<Record<PremiseTier, string[]>> = {}
  let weakestTier: PremiseTier = 'verified' // strongest; relaxed as weaker premises appear
  let total = 0
  for (const t of TIER_ORDER) {
    const ids = byTier.get(t)
    if (!ids || ids.size === 0) continue
    premisesByTier[t] = [...ids].sort()
    weakestTier = weaker(weakestTier, t)
    total += ids.size
  }
  if (total === 0) weakestTier = 'attested' // no ground premises (e.g. tautology): default, as before
  const assertedPremises = premisesByTier.asserted ?? []
  return {
    factId,
    weakestTier,
    weakestPremiseIds: premisesByTier[weakestTier] ?? [],
    premisesByTier,
    assertedPremises,
    attestedPremises: premisesByTier.attested ?? [],
    ungrounded: assertedPremises.length > 0 && assertedPremises.length === total,
  }
}

/** Grounding for many conclusions, weakest (most-in-need-of-verification) first. */
export function weakestPremises(
  facts: ProvenanceFact[],
  factIds: string[],
  options: GroundingOptions = {},
): Grounding[] {
  return factIds
    .map((id) => groundingOf(facts, id, options))
    .sort((a, b) => tierRank(b.weakestTier) - tierRank(a.weakestTier)) // weakest (highest rank) first
}

/** Map ground-premise ids back to readable atoms for display. */
export function describePremises(facts: ProvenanceFact[], ids: string[]): string[] {
  const byId = new Map(facts.map((f) => [f.nodeId, f]))
  return ids.map((id) => {
    const f = byId.get(id)
    return f ? `${f.atom.predicate}` : id
  })
}
