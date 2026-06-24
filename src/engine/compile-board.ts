import type { PredicateAtom, SemanticScalar } from '../model/types.js'
import type { RuleDefinition } from '../kernel/predicate.js'
import { atomKey, isVariable } from '../kernel/predicate.js'
import { COMPARISON_BUILTINS, isBuiltinPredicate } from '../kernel/builtins.js'
import { assertActionSafety } from '../kernel/safety.js'
import { deriveStageLayers } from './derive-stages.js'

/**
 * compile-board — compile a board rule program to a readable PURE-FUNCTION
 * pipeline, ordered by deriveStages, and prove (via roundtrip) that the
 * compiled pipeline produces the SAME derived facts as the board's closure.
 *
 * THE THESIS, MADE CONCRETE
 * -------------------------
 * The kernel's rule program (stratified-NAF Datalog) is ALREADY a declarative
 * dataflow program: deriveStages computes its dependency-DAG topological
 * schedule. So a board rule program COMPILES TO ORDINARY CODE — each dependency
 * layer becomes a pure function, evaluated in deriveStages order. Layer k reads
 * the base facts plus everything layers < k derived, scans/joins them, and
 * appends its head facts. There is no general loop across layers (the program
 * stays in the decidable fragment); that is exactly the point.
 *
 * WHAT THIS COMPILES (v1 SCOPE — honest boundaries)
 * -------------------------------------------------
 *  - Conjunctive POSITIVE literals: a relational join on shared variables.
 *  - NAF literals (`naf: true`): an anti-join — "no fact matches this pattern
 *    under the current bindings exists in the current set".
 *  - COMPARISON built-ins (eq/neq/lt/lte/gt/gte/between/contains): a JS guard on
 *    bound terms.
 *
 * OUT OF SCOPE v1 (fail-visibly, never emit plausible-but-wrong code):
 *  - ARITHMETIC built-ins (add/sub/.../concat) — they are value PRODUCERS that
 *    bind a `result` variable, not guards. The derivation spine deriveStages
 *    schedules is pure relational + guards; arithmetic producers belong to a
 *    later evaluator. A rule using one throws naming the rule + the builtin.
 *  - derive_aggregate recipes (sum/count/... over a fact set) — the aggregation
 *    layer; not part of the per-tuple join spine. (These never appear as raw
 *    rules anyway: they expand to chained-arithmetic rules, which we also reject
 *    above.)
 *  - ACTIONS / effects (the imperative consume/produce layer) — deriveStages
 *    explicitly does NOT schedule those, so they are simply not in the rule set
 *    handed here.
 *  - RECURSIVE rules (a head predicate appears in a strictly-higher-or-equal
 *    layer's own body, i.e. a dependency cycle) — deriveStages clamps a cycle to
 *    one layer; we cannot compile a fixpoint to a straight-line function, so a
 *    self/mutually recursive predicate throws.
 *  - strong negation in HEADS / negated body literals — out of the v1 relational
 *    fragment we prove; reject rather than half-handle.
 *
 * FAITHFULNESS PROOF
 * ------------------
 * compileBoardAndCheck (and the selftest in examples/compile-board-agent.ts)
 * runs both paths on the same base facts and asserts the derived-fact SETS are
 * EQUAL (same predicate+args tuples, compared by atomKey). The equality is the
 * headline — if it ever fails, the compilation has a bug.
 */

export type Fact = { predicate: string; args: Record<string, SemanticScalar> }

/**
 * An AGGREGATE spec — the compile-level mirror of a `derive_aggregate`
 * working-memory operation. The board op EXPANDS into a chained-arithmetic rule
 * (one `add`/`min`/`max` literal per source fact, grounding every value as a
 * constant) which the relational+guard compiler rejects. So we DO NOT compile
 * the expanded rules; we intercept at the SPEC level and compile the aggregate
 * as a first-class layer node (a GROUP BY in SQL, a reduce in JS).
 *
 * Field-for-field correspondence to the derive_aggregate op:
 *   target  ← into.predicate         (the head predicate the result lands on)
 *   outArg  ← into.valueArg          (the head arg carrying the aggregate value)
 *   source  ← source.predicate       (the fact set to fold over)
 *   valueArg← source.valueArg        (the numeric arg; omit for kind:'count')
 *   kind    ← kind                   ('sum'|'count'|'min'|'max'; default 'sum')
 *   where   ← where {arg, equals}    (optional equality pre-filter)
 *   groupBy ← group_by               (optional single group key)
 */
export type AggregateSpec = {
  /** into.predicate — the result fact's predicate. */
  target: string
  /** into.valueArg — the result fact's arg carrying the aggregate value. */
  outArg: string
  /** source.predicate — the fact set to aggregate over. */
  source: string
  /** source.valueArg — the numeric arg to fold. Required except for 'count'. */
  valueArg?: string
  /** default 'sum'. */
  kind?: 'sum' | 'count' | 'min' | 'max'
  /** optional equality pre-filter: only source facts with args[arg]===equals. */
  where?: { arg: string; equals: SemanticScalar }
  /** optional single group key: one result fact per distinct source[groupBy]. */
  groupBy?: string
}

export type CompileBoardOptions = {
  /**
   * Goal predicate to compile toward. Default: derive the whole program — we
   * pass a synthetic sink that depends on every head predicate so deriveStages
   * returns ALL layers. (deriveStages only returns layers the goal depends on.)
   */
  goalPredicate?: string
  /**
   * Aggregate specs to compile alongside the relational rules. Each is scheduled
   * AFTER its source predicate's layer (it depends on the fully-materialised
   * source like any derived predicate). The target predicate may then feed
   * downstream rules — its layer is computed from the source's layer + 1.
   */
  aggregates?: AggregateSpec[]
}

export type CompiledBoard = {
  /** Human-readable compiled source: one pure function per layer, dependencies
   *  threaded as inputs. This is the "board → code" artifact. */
  source: string
  /** Executable pipeline: base facts in, all derived facts out (NOT including
   *  the base facts). Built as composed JS closures (no eval) so it is safe to
   *  run; `source` is the readable rendering of the same schedule. */
  pipeline: (baseFacts: Fact[]) => Fact[]
  /** The deriveStages note + the layer order actually compiled. */
  note: string
  /** Per-layer predicate grouping actually compiled, ascending. */
  layers: { layer: number; predicates: string[] }[]
}

export class CompileBoardError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CompileBoardError'
  }
}

const SYNTHETIC_GOAL = '__compile_all__'

const AGGREGATE_KINDS = new Set(['sum', 'count', 'min', 'max'])

/** Validate one aggregate spec, fail-visibly mirroring the engine's expandAggregate
 *  checks. Shared by both backends so the SAME fragment boundary holds. avg is the
 *  honest gap: the board's avg goes through IEEE `div` (a float that does not
 *  round-trip cleanly through atomKey), so the compiler refuses it rather than
 *  emit a value that could silently disagree with the board. */
export function assertAggregatable(spec: AggregateSpec): void {
  const kind = spec.kind ?? 'sum'
  if (kind === ('avg' as string)) {
    throw new CompileBoardError(
      `aggregate into "${spec.target}": kind "avg" is not compiled. avg folds through ` +
        `IEEE div (a float result), which the exact-integer compile fragment does not ` +
        `reproduce faithfully. Compile sum + count and divide downstream, or keep avg on the board.`,
    )
  }
  if (!AGGREGATE_KINDS.has(kind)) {
    throw new CompileBoardError(
      `aggregate into "${spec.target}": unknown kind "${kind}" — supports sum/count/min/max.`,
    )
  }
  if (typeof spec.source !== 'string' || spec.source.length === 0) {
    throw new CompileBoardError(`aggregate into "${spec.target}": source predicate is required.`)
  }
  if (typeof spec.target !== 'string' || spec.target.length === 0) {
    throw new CompileBoardError(`aggregate spec: target predicate is required.`)
  }
  if (typeof spec.outArg !== 'string' || spec.outArg.length === 0) {
    throw new CompileBoardError(`aggregate into "${spec.target}": outArg (into.valueArg) is required.`)
  }
  const needsValue = kind === 'sum' || kind === 'min' || kind === 'max'
  if (needsValue && (typeof spec.valueArg !== 'string' || spec.valueArg.length === 0)) {
    throw new CompileBoardError(
      `aggregate into "${spec.target}": kind "${kind}" needs valueArg (source.valueArg) — the numeric arg to fold.`,
    )
  }
  if (spec.where != null && (typeof spec.where.arg !== 'string' || spec.where.arg.length === 0)) {
    throw new CompileBoardError(`aggregate into "${spec.target}": where.arg must be a non-empty string.`)
  }
  if (spec.groupBy != null && (typeof spec.groupBy !== 'string' || spec.groupBy.length === 0)) {
    throw new CompileBoardError(`aggregate into "${spec.target}": groupBy must be a non-empty string.`)
  }
}

/** Reject if two aggregate specs target the same predicate (ambiguous merge) or
 *  an aggregate target collides with a rule head (the relational and aggregate
 *  layers cannot both produce one predicate in v1). */
export function assertAggregateSchedule(rules: RuleDefinition[], aggregates: AggregateSpec[]): void {
  const ruleHeads = new Set<string>()
  for (const r of rules) for (const h of r.then ?? []) ruleHeads.add(h.predicate)
  const seen = new Set<string>()
  for (const spec of aggregates) {
    assertAggregatable(spec)
    if (ruleHeads.has(spec.target)) {
      throw new CompileBoardError(
        `aggregate target "${spec.target}" is also a rule head — a predicate cannot be ` +
          `produced by BOTH a relational rule and an aggregate in v1. Use a distinct target.`,
      )
    }
    if (seen.has(spec.target)) {
      throw new CompileBoardError(
        `two aggregates both target "${spec.target}" — ambiguous. Give each a distinct target predicate.`,
      )
    }
    seen.add(spec.target)
  }
}

/** Exact-integer fold guard — mirrors kernel/builtins.guardResult: a non-finite
 *  result fails; an integer result that left the safe range fails (it has lost
 *  precision). Returns undefined on failure (the aggregate fact is not emitted,
 *  same as a rule whose arithmetic literal fails). */
function guardExact(result: number, inputs: number[]): number | undefined {
  if (!Number.isFinite(result)) return undefined
  if (inputs.every((v) => Number.isInteger(v)) && Number.isInteger(result) && !Number.isSafeInteger(result)) {
    return undefined
  }
  return result
}

/**
 * Compute one aggregate spec over a set of (already-materialised) facts, returning
 * the result facts. This is the SINGLE source of truth for the aggregate value used
 * by the JS pipeline; the SQL backend reproduces it via GROUP BY and node:sqlite
 * proves they agree. Faithful to the engine's expandAggregate fold: sort each
 * bucket by atomKey, fold left-to-right (sum/count via add, min/max via the binary
 * op), exact-or-fail. A non-numeric value in a sum/min/max bucket fails-visibly.
 */
export function computeAggregate(spec: AggregateSpec, facts: Fact[]): Fact[] {
  const kind = spec.kind ?? 'sum'
  const needsValue = kind === 'sum' || kind === 'min' || kind === 'max'
  // Filter to source predicate, then apply where.
  let pool = facts.filter((f) => f.predicate === spec.source)
  if (spec.where != null) {
    pool = pool.filter((f) => f.args[spec.where!.arg] === spec.where!.equals)
  }
  if (pool.length === 0) return [] // empty bucket(s) → no aggregate fact (matches an empty GROUP BY result set)

  // Bucket by groupBy (or one global bucket).
  const buckets = new Map<string, { groupValue?: SemanticScalar; rows: Fact[] }>()
  for (const f of pool) {
    if (spec.groupBy != null) {
      if (!Object.prototype.hasOwnProperty.call(f.args, spec.groupBy)) {
        throw new CompileBoardError(
          `aggregate into "${spec.target}": fact ${JSON.stringify(f)} lacks groupBy arg "${spec.groupBy}" — ` +
            `every source fact must carry the group key (silent undercount otherwise).`,
        )
      }
      const gv = f.args[spec.groupBy]
      const key = JSON.stringify(gv)
      const b = buckets.get(key) ?? { groupValue: gv, rows: [] }
      b.rows.push(f)
      buckets.set(key, b)
    } else {
      const b = buckets.get('') ?? { rows: [] }
      b.rows.push(f)
      buckets.set('', b)
    }
  }

  const out: Fact[] = []
  for (const key of [...buckets.keys()].sort()) {
    const { groupValue, rows } = buckets.get(key)!
    const sorted = [...rows].sort((a, b) =>
      atomKey(factToAtom(a)) < atomKey(factToAtom(b)) ? -1 : 1,
    )
    let acc: number
    const inputs: number[] = []
    if (kind === 'count') {
      acc = sorted.length
      // count is exact-by-construction (cardinality); safe-int guard still applies.
      const guarded = guardExact(acc, [acc])
      if (guarded === undefined) continue
      acc = guarded
    } else {
      const values: number[] = []
      for (const f of sorted) {
        const v = f.args[spec.valueArg!]
        if (typeof v !== 'number') {
          throw new CompileBoardError(
            `aggregate into "${spec.target}": source fact ${JSON.stringify(f)} has non-numeric ` +
              `"${spec.valueArg}" (${JSON.stringify(v)}). ${kind} needs a number on every source fact.`,
          )
        }
        values.push(v)
      }
      acc = values[0]
      inputs.push(values[0])
      let failed = false
      for (let i = 1; i < values.length; i++) {
        const next = values[i]
        inputs.push(next)
        const folded =
          kind === 'sum' ? acc + next : kind === 'min' ? Math.min(acc, next) : Math.max(acc, next)
        const guarded = guardExact(folded, inputs)
        if (guarded === undefined) {
          failed = true
          break
        }
        acc = guarded
      }
      if (failed) continue
    }
    const args: Record<string, SemanticScalar> = { [spec.outArg]: acc }
    if (spec.groupBy != null) args[spec.groupBy] = groupValue as SemanticScalar
    out.push({ predicate: spec.target, args })
  }
  return out
}

/** A working tuple during evaluation: a fact identified by its atomKey. */
type WorkFact = { atom: PredicateAtom; key: string }

function factToAtom(fact: Fact): PredicateAtom {
  return { predicate: fact.predicate, args: fact.args }
}

function atomToFact(atom: PredicateAtom): Fact {
  return { predicate: atom.predicate, args: { ...(atom.args ?? {}) } }
}

/** Reject any construct the v1 relational+guard compiler cannot faithfully
 *  emit. Fail visibly: name the rule and the exact offending construct.
 *  Exported so the SQL backend (compile-board-sql.ts) enforces the SAME
 *  fragment boundaries — additive, behaviour unchanged. */
export function assertCompilable(rules: RuleDefinition[]): void {
  const heads = new Set<string>()
  for (const rule of rules) for (const h of rule.then ?? []) heads.add(h.predicate)

  for (const rule of rules) {
    for (const head of rule.then ?? []) {
      if (head.naf === true) {
        throw new CompileBoardError(
          `rule "${rule.id}": a rule HEAD uses naf — negation-as-failure is a body-only ` +
            `construct and cannot be a conclusion. Not compilable.`,
        )
      }
      if (head.negated === true) {
        throw new CompileBoardError(
          `rule "${rule.id}": head predicate "${head.predicate}" uses strong negation ` +
            `(negated head). v1 compiles the positive relational fragment only — ` +
            `negative knowledge derivation is out of scope.`,
        )
      }
    }
    for (const literal of rule.when ?? []) {
      const pred = literal.predicate
      // Arithmetic producers and any non-comparison builtin: out of scope.
      if (isBuiltinPredicate(pred) && !COMPARISON_BUILTINS.has(pred)) {
        throw new CompileBoardError(
          `rule "${rule.id}": body literal uses arithmetic/producing built-in "${pred}". ` +
            `v1 compiles comparison guards (${[...COMPARISON_BUILTINS].join('/')}) and joins only — ` +
            `value-producing built-ins (they bind a "result") are out of scope. Not compilable.`,
        )
      }
      if (literal.negated === true) {
        throw new CompileBoardError(
          `rule "${rule.id}": body literal "${pred}" uses strong negation (negated). ` +
            `v1 compiles positive literals + naf anti-joins + comparison guards; ` +
            `strong-negative body literals are out of scope. Not compilable.`,
        )
      }
      // A naf literal over a builtin is meaningless here; builtins are guards.
      if (literal.naf === true && isBuiltinPredicate(pred)) {
        throw new CompileBoardError(
          `rule "${rule.id}": naf applied to built-in "${pred}". naf is an anti-join over ` +
            `facts, not over a guard. Not compilable.`,
        )
      }
    }
  }

  // Recursion / cycle detection: a head predicate that (transitively) depends
  // on itself cannot be compiled to straight-line code (it is a fixpoint, not a
  // pipeline). deriveStages would clamp it to one layer; we reject instead.
  const parents = new Map<string, Set<string>>()
  for (const rule of rules) {
    const body = (rule.when ?? []).map((a) => a.predicate)
    for (const h of rule.then ?? []) {
      const set = parents.get(h.predicate) ?? new Set<string>()
      for (const b of body) set.add(b)
      parents.set(h.predicate, set)
    }
  }
  for (const start of heads) {
    const seen = new Set<string>()
    const stack = [...(parents.get(start) ?? [])]
    while (stack.length > 0) {
      const p = stack.pop()!
      if (p === start) {
        throw new CompileBoardError(
          `predicate "${start}" is recursive (it transitively depends on itself). ` +
            `A recursive predicate is a fixpoint, not a straight-line layer — v1 compiles ` +
            `the non-recursive derivation spine only. Not compilable.`,
        )
      }
      if (seen.has(p)) continue
      seen.add(p)
      for (const parent of parents.get(p) ?? []) stack.push(parent)
    }
  }
}

/** Match one positive literal against a fact set under existing bindings,
 *  extending bindings. Mirrors kernel/predicate.matchAtom semantics. */
function matchPositive(
  literal: PredicateAtom,
  facts: WorkFact[],
  bindings: Record<string, SemanticScalar>,
): Record<string, SemanticScalar>[] {
  const out: Record<string, SemanticScalar>[] = []
  for (const fact of facts) {
    if (fact.atom.predicate !== literal.predicate) continue
    const next = unify(literal.args ?? {}, fact.atom.args ?? {}, bindings)
    if (next) out.push(next)
  }
  return out
}

/** Try to unify a pattern arg-map against a fact arg-map, extending bindings.
 *  Returns the extended bindings or undefined. Matches kernel/predicate.matchAtom:
 *  every pattern key must be present in the fact; constants must equal; a
 *  variable binds (and must stay consistent). */
function unify(
  patternArgs: Record<string, SemanticScalar>,
  factArgs: Record<string, SemanticScalar>,
  existing: Record<string, SemanticScalar>,
): Record<string, SemanticScalar> | undefined {
  const bindings = { ...existing }
  for (const [key, patternValue] of Object.entries(patternArgs)) {
    const factValue = factArgs[key]
    if (factValue === undefined) return undefined
    if (isVariable(patternValue)) {
      const name = patternValue.slice(1)
      const bound = bindings[name]
      if (bound !== undefined && bound !== factValue) return undefined
      bindings[name] = factValue
    } else if (patternValue !== factValue) {
      return undefined
    }
  }
  return bindings
}

/** Resolve a term (variable or constant) against bindings. */
function resolve(
  term: SemanticScalar,
  bindings: Record<string, SemanticScalar>,
): SemanticScalar | undefined {
  if (isVariable(term)) return bindings[term.slice(1)]
  return term
}

/** Evaluate a comparison guard under bindings. Mirrors builtins.evaluateBuiltin
 *  but reads from bindings instead of a ground atom. */
function evalGuard(
  literal: PredicateAtom,
  bindings: Record<string, SemanticScalar>,
): boolean {
  const pred = literal.predicate
  const args = literal.args ?? {}
  if (pred === 'between') {
    const value = resolve(args.value, bindings)
    const low = resolve(args.low, bindings)
    const high = resolve(args.high, bindings)
    if (value === undefined || low === undefined || high === undefined) return false
    return (
      typeof value === 'number' &&
      typeof low === 'number' &&
      typeof high === 'number' &&
      low <= value &&
      value <= high
    )
  }
  const left = resolve(args.left, bindings)
  const right = resolve(args.right, bindings)
  if (left === undefined || right === undefined) return false
  switch (pred) {
    case 'contains':
      return typeof left === 'string' && typeof right === 'string' && left.includes(right)
    case 'eq':
      return left === right
    case 'neq':
      return left !== right
    case 'lt':
      return typeof left === 'number' && typeof right === 'number' && left < right
    case 'lte':
      return typeof left === 'number' && typeof right === 'number' && left <= right
    case 'gt':
      return typeof left === 'number' && typeof right === 'number' && left > right
    case 'gte':
      return typeof left === 'number' && typeof right === 'number' && left >= right
    default:
      return false
  }
}

/** Does any fact match this (naf) literal under the bindings? Anti-join input. */
function existsMatch(
  literal: PredicateAtom,
  facts: WorkFact[],
  bindings: Record<string, SemanticScalar>,
): boolean {
  return facts.some(
    (fact) =>
      fact.atom.predicate === literal.predicate &&
      unify(literal.args ?? {}, fact.atom.args ?? {}, bindings) !== undefined,
  )
}

/**
 * Order a rule body the same way the kernel matcher does: positive (binding)
 * literals first, then guards/naf. (For the relational+guard fragment we
 * support, comparison guards and naf only need their variables bound, which the
 * positive literals provide; we keep a stable order otherwise.)
 */
function orderBody(body: PredicateAtom[]): PredicateAtom[] {
  const isBinder = (l: PredicateAtom): boolean =>
    l.naf !== true && !isBuiltinPredicate(l.predicate)
  const binders = body.filter(isBinder)
  const rest = body.filter((l) => !isBinder(l))
  return [...binders, ...rest]
}

/**
 * Solve a rule/action body against a fact set: return EVERY satisfying binding
 * (variable name → value) in matcher order. Positive literals join and bind, naf
 * literals anti-join, comparison built-ins guard. This is the SHARED binding
 * engine for both rule firing (all bindings → head facts) and action application
 * (the FIRST binding → effects). Pure: does not mutate `facts`.
 */
function solveBody(
  body: PredicateAtom[],
  facts: WorkFact[],
): Record<string, SemanticScalar>[] {
  const ordered = orderBody(body)
  let bindingSets: Record<string, SemanticScalar>[] = [{}]
  for (const literal of ordered) {
    if (literal.naf === true) {
      bindingSets = bindingSets.filter((b) => !existsMatch(literal, facts, b))
    } else if (isBuiltinPredicate(literal.predicate)) {
      bindingSets = bindingSets.filter((b) => evalGuard(literal, b))
    } else {
      const next: Record<string, SemanticScalar>[] = []
      for (const b of bindingSets) next.push(...matchPositive(literal, facts, b))
      bindingSets = next
    }
    if (bindingSets.length === 0) break
  }
  return bindingSets
}

/** Run one rule over a fact set, returning the head facts it derives (as atoms,
 *  for every satisfying binding). Pure: does not mutate `facts`. */
function fireRule(rule: RuleDefinition, facts: WorkFact[]): PredicateAtom[] {
  const bindingSets = solveBody(rule.when ?? [], facts)
  const derived: PredicateAtom[] = []
  for (const head of rule.then ?? []) {
    for (const b of bindingSets) {
      const args: Record<string, SemanticScalar> = {}
      let ok = true
      for (const [key, value] of Object.entries(head.args ?? {})) {
        const resolved = isVariable(value) ? b[value.slice(1)] : value
        if (resolved === undefined) {
          ok = false
          break
        }
        args[key] = resolved
      }
      if (ok) derived.push({ predicate: head.predicate, args })
    }
  }
  return derived
}

/**
 * Compile the rule program to a pure-function pipeline ordered by deriveStages.
 * Each layer is evaluated to fixpoint over (base + lower layers' derived)
 * facts — this matches the kernel's stratified closure for the relational +
 * guard + naf fragment, where a NAF literal is only sound once its target
 * predicate is fully derived in a strictly lower layer (deriveStages/strata
 * guarantee that ordering).
 */
/**
 * Compute the deriveStages-ordered layer schedule for a rule program: the
 * derivation layers (ascending), the rules grouped by their head predicate's
 * layer, and the set of head (derivable) predicates. Shared between the JS
 * pipeline backend and the SQL view backend so BOTH emit the SAME ordering from
 * the SAME source of truth. Additive — compileBoard's behaviour is unchanged.
 */
export function layerSchedule(
  rules: RuleDefinition[],
  goalPredicate?: string,
  aggregates: AggregateSpec[] = [],
): {
  layers: { layer: number; predicates: string[] }[]
  rulesByLayer: Map<number, RuleDefinition[]>
  /** Aggregate specs grouped by the layer their target predicate landed in. */
  aggregatesByLayer: Map<number, AggregateSpec[]>
  heads: Set<string>
  note: string
} {
  const heads = new Set<string>()
  for (const rule of rules) for (const h of rule.then ?? []) heads.add(h.predicate)
  // Aggregate targets are derived predicates too.
  const aggHeads = new Set(aggregates.map((a) => a.target))
  for (const t of aggHeads) heads.add(t)

  // Synthesize a dependency rule per aggregate: target :- source. This carries
  // the dependency edge (target depends on source) into deriveStages so the
  // target lands in a layer strictly after its source — exactly like any derived
  // predicate. The synthetic rule is NEVER fired relationally; the aggregate fold
  // replaces it. (It only shapes the schedule.)
  const aggDepRules: RuleDefinition[] = aggregates.map((a, i) => ({
    id: `__agg_dep_${i}__`,
    when: [{ predicate: a.source, args: {} }],
    then: [{ predicate: a.target, args: {} }],
  }))

  const scheduleRules = [...rules, ...aggDepRules]

  let stagesRules = scheduleRules
  const goal = goalPredicate ?? SYNTHETIC_GOAL
  if (goalPredicate === undefined) {
    stagesRules = [
      ...scheduleRules,
      {
        id: '__compile_sink__',
        when: [...heads].map((p) => ({ predicate: p, args: {} })),
        then: [{ predicate: SYNTHETIC_GOAL, args: {} }],
      },
    ]
  }

  const { groups, note } = deriveStageLayers(stagesRules, goal)
  // Drop the synthetic sink's layer (the topmost) if present.
  const layers = groups
    .filter((g) => !g.predicates.includes(SYNTHETIC_GOAL))
    .map((g) => ({ layer: g.layer, predicates: g.predicates.slice() }))

  // Group the real rules by the layer of their head predicate.
  const layerOf = new Map<string, number>()
  for (const g of layers) for (const p of g.predicates) layerOf.set(p, g.layer)
  const rulesByLayer = new Map<number, RuleDefinition[]>()
  for (const rule of rules) {
    const headPreds = (rule.then ?? []).map((h) => h.predicate)
    const layer = Math.max(...headPreds.map((p) => layerOf.get(p) ?? 1))
    const list = rulesByLayer.get(layer) ?? []
    list.push(rule)
    rulesByLayer.set(layer, list)
  }

  // Group aggregate specs by the layer their target predicate landed in.
  const aggregatesByLayer = new Map<number, AggregateSpec[]>()
  for (const spec of aggregates) {
    const layer = layerOf.get(spec.target) ?? 1
    const list = aggregatesByLayer.get(layer) ?? []
    list.push(spec)
    aggregatesByLayer.set(layer, list)
  }

  return { layers, rulesByLayer, aggregatesByLayer, heads, note }
}

export function compileBoard(
  rules: RuleDefinition[],
  options: CompileBoardOptions = {},
): CompiledBoard {
  assertCompilable(rules)
  const aggregates = options.aggregates ?? []
  assertAggregateSchedule(rules, aggregates)

  const { layers, rulesByLayer, aggregatesByLayer, heads, note } = layerSchedule(
    rules,
    options.goalPredicate,
    aggregates,
  )
  const orderedLayers = layers.map((g) => g.layer)

  const pipeline = (baseFacts: Fact[]): Fact[] => {
    const known = new Map<string, WorkFact>()
    for (const f of baseFacts) {
      const atom = factToAtom(f)
      const key = atomKey(atom)
      if (!known.has(key)) known.set(key, { atom, key })
    }
    const baseKeys = new Set(known.keys())

    for (const layer of orderedLayers) {
      const layerRules = rulesByLayer.get(layer) ?? []
      // Fixpoint within the layer (positive recursion inside one layer is not
      // present for our scope, but the loop is the faithful closure operator).
      let changed = true
      while (changed) {
        changed = false
        const facts = [...known.values()]
        for (const rule of layerRules) {
          for (const atom of fireRule(rule, facts)) {
            const key = atomKey(atom)
            if (!known.has(key)) {
              known.set(key, { atom, key })
              changed = true
            }
          }
        }
      }
      // Aggregates scheduled in this layer fold over the now-materialised facts
      // (base + everything lower layers, and this layer's relational rules,
      // derived). Their target predicates may feed higher layers.
      const layerAggs = aggregatesByLayer.get(layer) ?? []
      if (layerAggs.length > 0) {
        const facts = [...known.values()].map((wf) => atomToFact(wf.atom))
        for (const spec of layerAggs) {
          for (const resultFact of computeAggregate(spec, facts)) {
            const atom = factToAtom(resultFact)
            const key = atomKey(atom)
            if (!known.has(key)) known.set(key, { atom, key })
          }
        }
      }
    }

    // Return only the DERIVED facts (everything not in the base set).
    const derived: Fact[] = []
    for (const [key, wf] of known) {
      if (!baseKeys.has(key)) derived.push(atomToFact(wf.atom))
    }
    return derived
  }

  const source = renderSource(layers, rulesByLayer, aggregatesByLayer, heads, rules)
  return { source, pipeline, note, layers }
}

/** Render the readable compiled source: one function per layer. */
function renderSource(
  layers: { layer: number; predicates: string[] }[],
  rulesByLayer: Map<number, RuleDefinition[]>,
  aggregatesByLayer: Map<number, AggregateSpec[]>,
  heads: Set<string>,
  allRules: RuleDefinition[],
): string {
  const baseInputs = new Set<string>()
  for (const rule of allRules) {
    for (const l of rule.when ?? []) {
      if (!heads.has(l.predicate) && !isBuiltinPredicate(l.predicate)) baseInputs.add(l.predicate)
    }
  }

  const lines: string[] = []
  lines.push('// Compiled board → pure-function pipeline (one function per dependency layer).')
  lines.push('// deriveStages ordered the layers; each layer reads base facts +')
  lines.push('// everything lower layers derived, scans/joins, and appends head facts.')
  lines.push('// Fact = { predicate, args }; facts flow downstream, never mutated upstream.')
  lines.push('')
  lines.push(`// base inputs (ingested, depth 0): ${[...baseInputs].sort().join(', ') || '(none)'}`)
  lines.push('')

  for (const g of layers) {
    const layerRules = rulesByLayer.get(g.layer) ?? []
    const layerAggs = aggregatesByLayer.get(g.layer) ?? []
    lines.push(`// ── layer ${g.layer}: derive ${g.predicates.join(', ')} ──`)
    lines.push(`function deriveLayer${g.layer}(facts) {`)
    lines.push('  const out = []')
    for (const rule of layerRules) {
      lines.push(...renderRule(rule).map((l) => '  ' + l))
    }
    for (const spec of layerAggs) {
      lines.push(...renderAggregate(spec).map((l) => '  ' + l))
    }
    lines.push('  return out // append to facts, continue to next layer')
    lines.push('}')
    lines.push('')
  }

  // The driver.
  lines.push('function pipeline(baseFacts) {')
  lines.push('  let facts = [...baseFacts]')
  for (const g of layers) {
    lines.push(`  facts = facts.concat(deriveLayer${g.layer}(facts)) // layer ${g.layer}`)
  }
  lines.push('  return facts.filter(f => !isBase(f)) // the derived facts')
  lines.push('}')
  return lines.join('\n')
}

/** Render one aggregate spec as readable reduce pseudocode. The runtime fold is
 *  computeAggregate; this is its prose. */
function renderAggregate(spec: AggregateSpec): string[] {
  const kind = spec.kind ?? 'sum'
  const lines: string[] = []
  const whereNote = spec.where != null ? ` where ${spec.where.arg}===${JSON.stringify(spec.where.equals)}` : ''
  const groupNote = spec.groupBy != null ? ` GROUP BY ${spec.groupBy}` : ''
  lines.push(`// aggregate: ${kind} of ${spec.source}${spec.valueArg ? '.' + spec.valueArg : ''}${whereNote}${groupNote} -> ${spec.target}.${spec.outArg}`)
  lines.push(`{`)
  lines.push(`  let rows = facts.filter(f => f.predicate === '${spec.source}')`)
  if (spec.where != null) {
    lines.push(`  rows = rows.filter(f => f.args.${spec.where.arg} === ${JSON.stringify(spec.where.equals)})`)
  }
  if (spec.groupBy != null) {
    lines.push(`  const groups = new Map() // key = f.args.${spec.groupBy}`)
    lines.push(`  for (const f of rows) groups.get(f.args.${spec.groupBy})?.push(f) ?? groups.set(f.args.${spec.groupBy}, [f])`)
    lines.push(`  for (const [g, bucket] of groups) {`)
    lines.push(`    const v = reduce${kind}(bucket${spec.valueArg ? `, '${spec.valueArg}'` : ''}) // exact-or-fail`)
    lines.push(`    if (v !== undefined) out.push({ predicate: '${spec.target}', args: { ${spec.outArg}: v, ${spec.groupBy}: g } })`)
    lines.push(`  }`)
  } else {
    lines.push(`  if (rows.length > 0) {`)
    lines.push(`    const v = reduce${kind}(rows${spec.valueArg ? `, '${spec.valueArg}'` : ''}) // exact-or-fail`)
    lines.push(`    if (v !== undefined) out.push({ predicate: '${spec.target}', args: { ${spec.outArg}: v } })`)
    lines.push(`  }`)
  }
  lines.push(`}`)
  return lines
}

/**
 * Emit the nested-scan OPENING for a body: each positive literal becomes a scan
 * binding a fresh row alias; equality between a scan's arg and an already-bound
 * variable becomes an explicit join guard; comparison built-ins become if-guards;
 * naf literals become anti-join if-guards. Leaves the cursor at the innermost
 * indent with every variable mapped in `varExpr`, so the caller can emit its
 * payload (head pushes for a rule, effects for an action) and then close the
 * braces. Shared by renderRule and renderActionSource — identical scan rendering
 * for both backends. */
function openBodyScans(
  body: PredicateAtom[],
  baseIndent: string,
): { lines: string[]; indent: string; varExpr: Map<string, string> } {
  const ordered = orderBody(body)
  const positives = ordered.filter((l) => l.naf !== true && !isBuiltinPredicate(l.predicate))
  const guards = ordered.filter((l) => isBuiltinPredicate(l.predicate))
  const nafs = ordered.filter((l) => l.naf === true)

  // Map each variable to the JS expression that holds it (first scan that binds
  // it). Subsequent occurrences become join-equality guards.
  const varExpr = new Map<string, string>()
  const lines: string[] = []
  let indent = baseIndent

  positives.forEach((lit, i) => {
    const alias = `r${i}` // row alias for this scan
    lines.push(
      `${indent}for (const ${alias} of facts.filter(f => f.predicate === '${lit.predicate}')) {`,
    )
    indent += '  '
    for (const [key, value] of Object.entries(lit.args ?? {})) {
      const expr = `${alias}.args.${key}`
      if (isVariable(value)) {
        const name = value.slice(1)
        const bound = varExpr.get(name)
        if (bound === undefined) {
          varExpr.set(name, expr) // binds the variable
        } else {
          lines.push(`${indent}if (${expr} === ${bound}) { // join on ?${name}`)
          indent += '  '
        }
      } else {
        lines.push(`${indent}if (${expr} === ${JSON.stringify(value)}) { // const match`)
        indent += '  '
      }
    }
  })
  for (const g of guards) {
    lines.push(`${indent}if (${formatGuard(g, varExpr)}) {`)
    indent += '  '
  }
  for (const n of nafs) {
    lines.push(
      `${indent}if (!facts.some(f => f.predicate === '${n.predicate}' && ${formatAntiJoin(n, varExpr)})) { // naf anti-join`,
    )
    indent += '  '
  }
  return { lines, indent, varExpr }
}

/** Render one rule as a readable nested-scan pseudocode block. Each positive
 *  literal becomes a scan binding a fresh row alias; equality between a scan's
 *  arg and an already-bound variable becomes an explicit join guard. naf =
 *  anti-join, comparison builtins = guards. This is the readable rendering of
 *  what fireRule does at runtime. */
function renderRule(rule: RuleDefinition): string[] {
  const { lines, indent, varExpr } = openBodyScans(rule.when ?? [], '')
  const out: string[] = [`// rule ${rule.id}`, ...lines]
  let ind = indent
  for (const head of rule.then ?? []) {
    out.push(`${ind}out.push({ predicate: '${head.predicate}', args: ${formatHeadArgs(head, varExpr)} })`)
  }
  // close every open brace (scans + per-arg join/const guards + guards + nafs).
  while (ind.length > 0) {
    ind = ind.slice(2)
    out.push(`${ind}}`)
  }
  return out
}

/** A scan's bound variables / constants → an object literal for a head atom. */
function formatHeadArgs(atom: PredicateAtom, varExpr: Map<string, string>): string {
  const parts = Object.entries(atom.args ?? {}).map(([k, v]) => {
    if (isVariable(v)) return `${k}: ${varExpr.get(v.slice(1)) ?? `/*unbound ?${v.slice(1)}*/`}`
    return `${k}: ${JSON.stringify(v)}`
  })
  return `{ ${parts.join(', ')} }`
}

/** An anti-join predicate over a naf literal: match its args against f. */
function formatAntiJoin(atom: PredicateAtom, varExpr: Map<string, string>): string {
  const conds = Object.entries(atom.args ?? {}).map(([k, v]) => {
    const rhs = isVariable(v) ? varExpr.get(v.slice(1)) ?? `/*unbound*/` : JSON.stringify(v)
    return `f.args.${k} === ${rhs}`
  })
  return conds.length > 0 ? conds.join(' && ') : 'true'
}

function formatGuard(atom: PredicateAtom, varExpr: Map<string, string>): string {
  const a = atom.args ?? {}
  const term = (v: SemanticScalar | undefined): string =>
    v === undefined
      ? 'undefined'
      : isVariable(v)
        ? varExpr.get(v.slice(1)) ?? `/*unbound ?${v.slice(1)}*/`
        : JSON.stringify(v)
  switch (atom.predicate) {
    case 'eq':
      return `${term(a.left)} === ${term(a.right)}`
    case 'neq':
      return `${term(a.left)} !== ${term(a.right)}`
    case 'lt':
      return `${term(a.left)} < ${term(a.right)}`
    case 'lte':
      return `${term(a.left)} <= ${term(a.right)}`
    case 'gt':
      return `${term(a.left)} > ${term(a.right)}`
    case 'gte':
      return `${term(a.left)} >= ${term(a.right)}`
    case 'between':
      return `${term(a.low)} <= ${term(a.value)} && ${term(a.value)} <= ${term(a.high)}`
    case 'contains':
      return `${term(a.left)}.includes(${term(a.right)})`
    default:
      return `/* ${atom.predicate} */ false`
  }
}

// ───────────────────────────────────────────────────────────────────────────
// ACTION-LAYER COMPILATION (v1)
//
// The rule backend above compiles the DERIVATION spine (rules → straight-line
// layers). Actions are the other half of the board: the imperative consume /
// produce layer. They compile too, but to a DIFFERENT shape, because their
// runtime semantics are different:
//
//   - A RULE fires for EVERY satisfying binding, to fixpoint (set semantics).
//     deriveStages schedules it; it becomes a layer function.
//   - An ACTION applies ONCE, at the FIRST satisfying precondition binding
//     (single-shot). Its positive effects assert facts; its negated effects
//     consume (delete) the matching facts. This is exactly deriveActionEffects /
//     simulateActionEffects in the engine.
//
// So an action compiles to a GUARDED TRANSFORMATION: bind the preconditions
// (the same relational join + comparison-guard + naf anti-join fragment the rule
// backend compiles), take the first binding, and apply the effects once. The
// compiled `apply` reproduces deriveActionEffects's added/removed atom sets on
// the same facts — that equality is the faithfulness headline (proven by the
// roundtrip tests), the action-layer mirror of compileBoardAndCheck.
//
// WHY NO SQL BACKEND FOR ACTIONS (the honest boundary, mirroring avg):
//   The single-shot "first binding wins" semantics has no faithful SQL image:
//   an INSERT…SELECT / DELETE…WHERE applies to the WHOLE matching set, not one
//   chosen tuple, so it would DISAGREE with the board whenever the preconditions
//   bind more than once. Rather than emit set-semantics SQL that silently
//   diverges from the board (the exact failure mode this project refuses), the
//   SQL action backend is deliberately out of v1 scope. (When the preconditions
//   are provably unambiguous the two coincide — a later, narrower SQL path.)
// ───────────────────────────────────────────────────────────────────────────

/** Compile-level mirror of a `define_action` working-memory op: an id, its
 *  preconditions (a rule-body-shaped pattern), and its effects (positive =
 *  assert, negated = consume). Field-for-field the action node's `semantic`. */
export type ActionForCompile = {
  id: string
  preconditions?: PredicateAtom[]
  effects?: PredicateAtom[]
}

/** The result of running a compiled action over a fact set. Faithful to
 *  deriveActionEffects/simulateActionEffects: `added`/`removed` are the ground
 *  atoms the action asserted/consumed; `facts` is the resulting set. */
export type ActionApplication = {
  /** Did the preconditions bind against the facts? */
  applied: boolean
  /** The fact set after the effects (consumed removed, asserted appended). When
   *  preconditions do not bind, the input set is returned unchanged. */
  facts: Fact[]
  /** Ground atoms newly asserted (were not already present). */
  added: Fact[]
  /** Ground atoms consumed (were present and got deleted). */
  removed: Fact[]
  /** The first precondition binding the action ran under (variable → value). */
  binding: Record<string, SemanticScalar>
  /** Number of distinct precondition bindings; >1 means the choice was ambiguous
   *  (the first is used, exactly like the engine, which surfaces a warning). */
  candidates: number
}

/** The "board → code" artifact for one action, plus the executable transform. */
export type CompiledAction = {
  /** Human-readable compiled source: a single-shot guarded transformation
   *  function. The "action → code" artifact. */
  source: string
  /** Executable transform (composed closures, no eval): facts in → application
   *  out. Faithful to deriveActionEffects on the same facts. */
  apply: (facts: Fact[]) => ActionApplication
  /** What was compiled and the fragment boundary. */
  note: string
}

/** Reject any action the v1 transformation compiler cannot faithfully emit.
 *  Two layers of checks, both fail-visible:
 *   1. assertActionSafety (the kernel's define_action validator): every effect
 *      variable must be bound by a precondition, effects carry no built-ins / no
 *      naf, precondition built-in inputs are range-restricted. SHARED with the
 *      engine so the compiler's boundary is exactly the board's.
 *   2. The relational+guard fragment on preconditions, mirroring assertCompilable
 *      for rule bodies: no arithmetic/producing built-ins (they bind a value —
 *      "counted transformation" — which belongs to a later evaluator, same as
 *      rules), no strong-negated body literals, no naf over a built-in. */
export function assertCompilableAction(action: ActionForCompile): void {
  // (1) engine-shared safety: effect vars bound, effects clean, ranges restricted.
  assertActionSafety(action)

  // (2) the relational+comparison-guard precondition fragment.
  for (const literal of action.preconditions ?? []) {
    const pred = literal.predicate
    if (isBuiltinPredicate(pred) && !COMPARISON_BUILTINS.has(pred)) {
      throw new CompileBoardError(
        `action "${action.id}": precondition uses arithmetic/producing built-in "${pred}". ` +
          `v1 compiles comparison guards (${[...COMPARISON_BUILTINS].join('/')}) + joins + naf ` +
          `anti-joins only — value-producing built-ins (they bind a "result", e.g. counted ` +
          `transformation) are out of the compile fragment. Not compilable.`,
      )
    }
    if (literal.negated === true) {
      throw new CompileBoardError(
        `action "${action.id}": precondition "${pred}" uses strong negation (negated). v1 compiles ` +
          `positive literals + naf anti-joins + comparison guards; strong-negative precondition ` +
          `literals are out of scope. Not compilable.`,
      )
    }
    if (literal.naf === true && isBuiltinPredicate(pred)) {
      throw new CompileBoardError(
        `action "${action.id}": naf applied to built-in "${pred}". naf is an anti-join over facts, ` +
          `not over a guard. Not compilable.`,
      )
    }
  }
}

/** Ground an effect atom with a precondition binding, mirroring the engine's
 *  instantiateEffect but in this module's own binding convention (var name →
 *  value, the same `solveBody` produces). assertCompilableAction has already
 *  guaranteed every effect variable is bound; the throw is defense in depth. */
function groundEffect(
  effect: PredicateAtom,
  binding: Record<string, SemanticScalar>,
  actionId: string,
): Fact {
  const args: Record<string, SemanticScalar> = {}
  for (const [key, value] of Object.entries(effect.args ?? {})) {
    const resolved = isVariable(value) ? binding[value.slice(1)] : value
    if (resolved === undefined) {
      throw new CompileBoardError(
        `action "${actionId}": effect "${effect.predicate}" uses variable ${String(value)} not bound ` +
          `by any precondition. (define_action safety should have caught this.)`,
      )
    }
    args[key] = resolved
  }
  return { predicate: effect.predicate, args }
}

/**
 * Compile one action to a single-shot guarded transformation. The returned
 * `apply` binds the preconditions to their FIRST satisfying instance, then
 * applies the effects in order — a positive effect asserts a ground fact, a
 * negated effect consumes the matching one — reproducing deriveActionEffects's
 * added/removed sets on the same facts. Effects are applied sequentially against
 * a mutable key set, so an add-then-consume of the same atom nets out exactly as
 * the engine's simulate does.
 */
export function compileAction(action: ActionForCompile): CompiledAction {
  assertCompilableAction(action)
  const preconditions = action.preconditions ?? []
  const effects = action.effects ?? []

  const apply = (inputFacts: Fact[]): ActionApplication => {
    // Dedup the input into the matcher's working set (facts are unique by
    // atomKey on a board); keep the original list for output ordering.
    const work: WorkFact[] = []
    const present = new Set<string>()
    for (const f of inputFacts) {
      const atom = factToAtom(f)
      const key = atomKey(atom)
      if (!present.has(key)) {
        present.add(key)
        work.push({ atom, key })
      }
    }

    const bindingSets = solveBody(preconditions, work)
    if (bindingSets.length === 0) {
      return { applied: false, facts: inputFacts.slice(), added: [], removed: [], binding: {}, candidates: 0 }
    }
    const binding = bindingSets[0]!

    const added: Fact[] = []
    const removed: Fact[] = []
    for (const effect of effects) {
      const ground = groundEffect(effect, binding, action.id)
      const key = atomKey(factToAtom(ground))
      if (effect.negated === true) {
        if (present.has(key)) {
          present.delete(key)
          removed.push(ground)
        }
      } else if (!present.has(key)) {
        present.add(key)
        added.push(ground)
      }
    }

    // Rebuild the surviving set: originals still present (input order), then the
    // asserted atoms still present (effect order). `present` reflects the net of
    // every add/consume, so add-then-consume of one atom leaves it in neither.
    const facts: Fact[] = []
    const emitted = new Set<string>()
    for (const f of inputFacts) {
      const key = atomKey(factToAtom(f))
      if (present.has(key) && !emitted.has(key)) {
        emitted.add(key)
        facts.push(f)
      }
    }
    for (const a of added) {
      const key = atomKey(factToAtom(a))
      if (present.has(key) && !emitted.has(key)) {
        emitted.add(key)
        facts.push(a)
      }
    }

    return { applied: true, facts, added, removed, binding, candidates: bindingSets.length }
  }

  const source = renderActionSource(action).join('\n')
  const note =
    `action "${action.id}" → single-shot guarded transformation: bind the first precondition ` +
    `instance, then assert positive effects and consume negated ones. Faithful to ` +
    `deriveActionEffects on the same facts (added/removed sets equal).`
  return { source, apply, note }
}

/** A JS-identifier-safe rendering of an action id for the function name. */
function sanitizeId(id: string): string {
  const safe = id.replace(/[^A-Za-z0-9_]/g, '_')
  return /^[A-Za-z_]/.test(safe) ? safe : `_${safe}`
}

/** Render one action as readable single-shot-transformation pseudocode: the
 *  precondition scans (shared with renderRule via openBodyScans), then the
 *  effects at the first binding, then the return. This is the readable rendering
 *  of what compileAction's `apply` does at runtime. */
function renderActionSource(action: ActionForCompile): string[] {
  const preconditions = action.preconditions ?? []
  const effects = action.effects ?? []
  const fnName = `applyAction_${sanitizeId(action.id)}`

  const lines: string[] = []
  lines.push(`// Compiled action "${action.id}" → single-shot guarded transformation.`)
  lines.push('// Bind the FIRST precondition match (joins + comparison guards + naf anti-joins),')
  lines.push('// then apply effects once: a positive effect asserts a fact, a negated effect')
  lines.push("// consumes the matching one. Mirrors deriveActionEffects (first match, ground effects).")
  lines.push('// sameFact(a, b): a.predicate === b.predicate && deepEqual(a.args, b.args).')
  lines.push(`function ${fnName}(facts) {`)

  const { lines: scanLines, indent, varExpr } = openBodyScans(preconditions, '  ')
  lines.push(...scanLines)

  lines.push(`${indent}// first satisfying binding → apply effects once, then return`)
  lines.push(`${indent}const added = [], removed = []`)
  for (const eff of effects) {
    const args = formatHeadArgs(eff, varExpr)
    if (eff.negated === true) {
      lines.push(`${indent}removed.push({ predicate: '${eff.predicate}', args: ${args} }) // consume`)
    } else {
      lines.push(`${indent}added.push({ predicate: '${eff.predicate}', args: ${args} }) // assert`)
    }
  }
  lines.push(
    `${indent}const facts2 = facts.filter(f => !removed.some(r => sameFact(f, r)))` +
      `.concat(added.filter(a => !facts.some(f => sameFact(f, a))))`,
  )
  lines.push(`${indent}return { applied: true, facts: facts2, added, removed } // first binding wins`)

  // Close the scan braces back down to the function-body indent.
  let ind = indent
  while (ind.length > 2) {
    ind = ind.slice(2)
    lines.push(`${ind}}`)
  }
  if (scanLines.length > 0) {
    lines.push('  return { applied: false, facts, added: [], removed: [] } // preconditions unsatisfied')
  }
  lines.push('}')
  return lines
}
