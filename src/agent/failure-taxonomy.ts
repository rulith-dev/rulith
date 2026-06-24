/**
 * failure-taxonomy — turn every way the board says "no" into a STABLE category
 * plus a copyable fix template, so failures become learnable (roadmap P3).
 *
 * rulith already fails visibly in many shapes: thrown guard errors
 * (DerivationGateError, AttestedPredicateError, StratificationError, ...),
 * standing board_critique problems (self_sealed_goal, asserted_finding, ...),
 * and characteristic teaching messages (plan ordering, exact-arithmetic). This
 * module is the SINGLE place that maps any of those to one taxonomy + the fix
 * to try - the substrate for: auto-classifying a run's failures, generating a
 * regression fixture from one, and a per-model friction profile (which model
 * trips which guard). It adds no inference of its own; it reads existing
 * signals (the propose/adjudicate boundary is unchanged).
 */
export type FailureKind =
  | 'asserted_finding' // a finding claimed instead of derived (derivation gate)
  | 'attested_violation' // tried to write a machine-attested predicate by fiat
  | 'evidence_reference' // record_result cited evidenceRefs that are not board node ids
  | 'self_sealed_goal' // goal satisfied only by a bare assertion
  | 'vacuous_rule' // a rule whose body just renames its head
  | 'unfirable_rule' // a rule whose body can provably never be satisfied (dead rule)
  | 'contradiction' // p and not-p both active
  | 'unsatisfiable_action' // an action that can never become applicable
  | 'unreachable_goal' // an open goal nothing can derive or produce
  | 'conflicting_goals' // two goals desire mutually contradictory atoms
  | 'unexplained_reopen' // a permission grant vanished without a revocation_result (I6.R6d)
  | 'unstratifiable' // negation cycle the closure cannot run
  | 'closure_divergence' // the closure did not reach a fixpoint (runaway rule)
  | 'join_explosion' // a rule body's cross-product blew up
  | 'rule_unsafe' // a rule failed the safety check (e.g. unbound head var)
  | 'action_unsafe' // an action failed the safety check
  | 'plan_ordering' // a plan step's preconditions were unmet in sequence
  | 'arithmetic_exact' // an exact-or-fail arithmetic literal failed
  | 'predicate_signature' // assert/rule missing predicate or required arg
  | 'unknown'

/** The standing board_critique kinds map 1:1 onto failure kinds. */
const CRITIQUE_TO_KIND: Record<string, FailureKind> = {
  self_sealed_goal: 'self_sealed_goal',
  asserted_finding: 'asserted_finding',
  vacuous_rule: 'vacuous_rule',
  unfirable_rule: 'unfirable_rule',
  contradiction: 'contradiction',
  unsatisfiable_action: 'unsatisfiable_action',
  unreachable_goal: 'unreachable_goal',
  conflicting_goals: 'conflicting_goals',
  unexplained_reopen: 'unexplained_reopen',
}

/** Thrown guard errors carry their class name; map by name (robust across module realms). */
const ERROR_NAME_TO_KIND: Record<string, FailureKind> = {
  DerivationGateError: 'asserted_finding',
  AttestedPredicateError: 'attested_violation',
  EvidenceReferenceError: 'evidence_reference',
  StratificationError: 'unstratifiable',
  ClosureDivergenceError: 'closure_divergence',
  JoinExplosionError: 'join_explosion',
  RuleSafetyError: 'rule_unsafe',
  ActionSafetyError: 'action_unsafe',
}

/** Concise, copyable guidance per kind: what the model should do next. */
const FIX_TEMPLATE: Record<FailureKind, string> = {
  asserted_finding: 'Assert the primitive observation you verified, add_axiom a rule deriving finding(...) from it, and let the closure produce the finding - do not assert finding(...) directly.',
  attested_violation: 'Do not assert/derive a machine-attested predicate (test_result/edited/build_status). Run the action so the harness records it, then read it in a rule body.',
  evidence_reference: 'record_result.evidenceRefs must cite exact node ids from get_logic_context. Do not cite prose labels, file refs, or invented ids; first assert/derive the evidence, then record_result with those node ids.',
  self_sealed_goal: 'The goal is satisfied only by a bare assertion. Assert the underlying observation and add a rule that DERIVES the desired atom, so the closure backs the goal.',
  vacuous_rule: 'This rule\'s body just renames its head. Give it a real body (observations/other predicates) so it derives something new.',
  unfirable_rule: 'This rule can never fire: a constant guard is always false, an arithmetic result is wrong, or a literal is required both present and absent (naf). Fix the constants/operands, or remove the dead rule.',
  contradiction: 'p and not-p are both active. retract_node the wrong one (and its evidence) so the board is consistent.',
  unsatisfiable_action: 'No fact can ever satisfy this action\'s preconditions. Add the producing rule/fact, or fix the precondition predicate/args.',
  unreachable_goal: 'Nothing can derive or produce this goal\'s atom. add_axiom a rule for it, define_action that produces it, or assert the missing fact.',
  conflicting_goals: 'Two goals want contradictory atoms (p vs not-p). Retract one, change its desired atom, or split into mutually exclusive branches.',
  unexplained_reopen: 'A permission grant disappeared without a revocation_result while its approval still stands. Record the revocation (revocation_result via the trusted channel) or re-grant — an authorization must not vanish silently.',
  unstratifiable: 'The rules form a negation cycle (e.g. recursive all-children-done over one predicate). Break it with an action-gated indirection predicate.',
  closure_divergence: 'A rule keeps producing new facts without a fixpoint. Bound it (a recursive accumulator never works); use derive_aggregate for totals.',
  join_explosion: 'A rule body\'s cross-product is too wide. Pin instances with more specific preconditions, or split the rule.',
  rule_unsafe: 'The rule is unsafe (e.g. a head/builtin variable not bound by a positive body literal). Bind every head variable in the body.',
  action_unsafe: 'The action is unsafe. Ensure effect variables are bound by preconditions and the shape is well-formed.',
  plan_ordering: 'A step\'s preconditions are not met in this order. Reorder, or insert the producer actions first (suggest_plan_repairs), then re-validate.',
  arithmetic_exact: 'An exact-or-fail arithmetic literal failed (out-of-range/NaN/Infinity, or non-numeric input). Use safe integers (|n| <= 2^53-1) and numeric operands.',
  predicate_signature: 'The operation is missing a predicate name or a required argument. Provide predicate + the args the rules expect.',
  unknown: 'Read the error/critique text; fix the cited node, then retry.',
}

export function fixTemplateFor(kind: FailureKind): string {
  return FIX_TEMPLATE[kind]
}

/** Map a standing board_critique kind to a failure kind (1:1). */
export function classifyCritiqueKind(kind: string): FailureKind {
  return CRITIQUE_TO_KIND[kind] ?? 'unknown'
}

/**
 * Classify a failure signal (a thrown guard error or a teaching-message
 * string) into a stable kind. Class name wins; otherwise characteristic
 * message phrases are matched (plan ordering / exact arithmetic / signature).
 */
export function classifyFailure(signal: unknown): { kind: FailureKind; fix: string } {
  let name = ''
  let message = ''
  if (signal instanceof Error) {
    name = signal.name
    message = signal.message
  } else if (typeof signal === 'string') {
    message = signal
  } else if (signal && typeof signal === 'object' && 'message' in signal) {
    message = String((signal as { message: unknown }).message)
  }

  const byName = ERROR_NAME_TO_KIND[name]
  const kind = byName ?? classifyByMessage(message)
  return { kind, fix: FIX_TEMPLATE[kind] }
}

function classifyByMessage(message: string): FailureKind {
  const m = message.toLowerCase()
  if (/machine-attested|attested predicate/.test(m)) return 'attested_violation'
  if (/evidencereferenceerror|evidencerefs must resolve|unresolved reference/.test(m)) return 'evidence_reference'
  if (/asserted, not derived|recorded result must rest on findings|derivation gate/.test(m)) return 'asserted_finding'
  if (/does not validate|plan stops|unmet precondition|did not apply at commit|preconditions? (fail|unmet)/.test(m)) return 'plan_ordering'
  if (/safe integer|out of range|infinity|nan|exact|arithmetic literal/.test(m)) return 'arithmetic_exact'
  if (/negation cycle|stratif/.test(m)) return 'unstratifiable'
  if (/cross-product|join width|join explosion/.test(m)) return 'join_explosion'
  if (/missing predicate|requires? a predicate|missing .*arg|needs source/.test(m)) return 'predicate_signature'
  return 'unknown'
}

/**
 * Per-model friction profile: which guard each model trips, and how often.
 * Feeds prompt adaptation ("model X keeps self-sealing -> emphasize the
 * derive-don't-assert template up front") and regression-fixture prioritization.
 */
export class FrictionProfile {
  private readonly counts = new Map<string, Map<FailureKind, number>>()

  record(model: string, kind: FailureKind): void {
    const perModel = this.counts.get(model) ?? new Map<FailureKind, number>()
    perModel.set(kind, (perModel.get(kind) ?? 0) + 1)
    this.counts.set(model, perModel)
  }

  /** Ingest a raw failure signal for a model in one step. */
  observe(model: string, signal: unknown): FailureKind {
    const { kind } = classifyFailure(signal)
    this.record(model, kind)
    return kind
  }

  /** Failure kinds for a model, most frequent first. */
  forModel(model: string): { kind: FailureKind; count: number }[] {
    const perModel = this.counts.get(model)
    if (!perModel) return []
    return [...perModel.entries()]
      .map(([kind, count]) => ({ kind, count }))
      .sort((a, b) => b.count - a.count)
  }

  /** A short adaptation hint: the top friction kinds' fix templates, to surface up front. */
  adaptationHints(model: string, top = 2): string[] {
    return this.forModel(model)
      .slice(0, top)
      .map(({ kind, count }) => `${model} often hits ${kind} (${count}x): ${FIX_TEMPLATE[kind]}`)
  }

  models(): string[] {
    return [...this.counts.keys()]
  }
}
