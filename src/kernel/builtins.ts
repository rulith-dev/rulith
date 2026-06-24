import type { PredicateAtom, SemanticScalar } from '../model/types.js'

/**
 * Guarded comparison built-ins. They are evaluated against bindings, not
 * against facts, and they never bind variables — rule safety requires
 * every variable in a built-in literal to be bound by a positive body
 * literal first. Because they cannot appear in rule heads and never
 * create new constants, they preserve termination of the closure.
 */
// Boolean GUARD built-ins: evaluated against bound args, never bind a variable,
// range-restricted (every var must be bound by a positive body literal first).
// Includes string membership (contains) alongside the numeric comparisons.
export const COMPARISON_BUILTINS = new Set(['eq', 'neq', 'lt', 'lte', 'gt', 'gte', 'between', 'contains'])

/**
 * Arithmetic built-ins — value PRODUCERS, not just guards. They evaluate
 * `left ∘ right` (or `left` for unary) against bound inputs and bind the
 * `result` arg. This is function evaluation, categorically different from
 * predicate derivation: the board computes an exact value the model would
 * mis-compute mentally — an external evaluator, the same role the compiler
 * plays for the repair agent.
 *
 * Discipline that keeps the kernel's guarantees: inputs must be bound by
 * positive body literals (range restriction), the literal cannot appear in
 * a rule head or be negated, and a single firing produces one value per
 * binding (the function is single-valued). The one footgun is RECURSIVE
 * arithmetic (feeding a result back as an input across iterations), which
 * can be unbounded — the closure evaluator caps iterations and fails loudly
 * rather than diverging. Non-recursive arithmetic always terminates.
 *
 * `concat` is a STRING producer in the same family: it joins two parts
 * (strings or safe integers) into a result string. Safe integers are
 * stringified canonically (5 → "5"). Floats, booleans, and out-of-range
 * integers are rejected with a teaching error.
 */
export const ARITHMETIC_BUILTINS = new Set([
  'add', 'sub', 'mul', 'div', 'mod', 'pow', // binary: left, right -> result
  'idiv', 'imod',                            // binary: integer floor-div & mathematical modulo
  'min', 'max',                              // binary
  'neg', 'abs', 'sign',                      // unary: left -> result
  'sqrt', 'ln', 'exp',                       // unary transcendental: finite-or-fail, IEEE by declaration
  'concat',                                  // binary string producer: left, right -> result (string)
])

const UNARY_ARITHMETIC = new Set(['neg', 'abs', 'sign', 'sqrt', 'ln', 'exp'])

export const BUILTIN_PREDICATES = new Set([...COMPARISON_BUILTINS, ...ARITHMETIC_BUILTINS])

export function isBuiltinPredicate(predicate: string): boolean {
  return BUILTIN_PREDICATES.has(predicate)
}

export function isComparisonBuiltin(predicate: string): boolean {
  return COMPARISON_BUILTINS.has(predicate)
}

export function isArithmeticBuiltin(predicate: string): boolean {
  return ARITHMETIC_BUILTINS.has(predicate)
}

/** The arg key an arithmetic built-in binds (its output). */
export const ARITHMETIC_RESULT_KEY = 'result'

/**
 * Evaluate a fully-ground comparison built-in. Expects `left`/`right`.
 * eq/neq use strict scalar equality; order comparisons require numbers.
 */
export function evaluateBuiltin(atom: PredicateAtom): boolean {
  // between is ternary {value, low, high} - dispatch before the binary
  // left/right guard (which would reject it for missing left/right). Closed
  // interval value in [low, high]; all three must be numbers or it is false
  // (conservative, like the binary order comparisons).
  if (atom.predicate === 'between') {
    const value = atom.args?.value
    const low = atom.args?.low
    const high = atom.args?.high
    if (value === undefined || low === undefined || high === undefined) return false
    return allNumbers(value, low, high) && low <= value && value <= high
  }

  const left = atom.args?.left
  const right = atom.args?.right
  if (left === undefined || right === undefined) return false

  switch (atom.predicate) {
    case 'contains':
      // string membership: left contains right as a substring. Both must be
      // strings (no number coercion) - else false, like the order comparisons.
      return typeof left === 'string' && typeof right === 'string' && left.includes(right)
    case 'eq':
      return left === right
    case 'neq':
      return left !== right
    case 'lt':
      return bothNumbers(left, right) && left < right
    case 'lte':
      return bothNumbers(left, right) && left <= right
    case 'gt':
      return bothNumbers(left, right) && left > right
    case 'gte':
      return bothNumbers(left, right) && left >= right
    default:
      return false
  }
}

/**
 * Compute an arithmetic or string-producing built-in from its (already-bound)
 * inputs. Returns the result value, or undefined when inputs are invalid /
 * the operation is undefined (e.g. divide by zero) — an undefined result makes
 * the literal fail, just like a false guard.
 *
 * Numeric exactness contract: integer arithmetic is EXACT within ±2^53
 * (Number.MAX_SAFE_INTEGER). An integer result beyond that range would be
 * silently rounded by IEEE-754 — a wrong number presented as exact — so it
 * fails instead. Non-finite results (overflow to Infinity, NaN from e.g.
 * pow(-1, 0.5)) also fail: they must never become fact arguments. Float
 * arithmetic is IEEE best-effort by declaration (0.1+0.2 has the usual
 * representation error) and is allowed.
 *
 * `concat` is a string producer in this family: it throws a teaching error
 * rather than returning undefined for clearly-wrong input types (boolean,
 * float with lossy representation, out-of-range integer) so the model gets
 * an actionable message instead of a silent rule non-firing.
 */
export function computeArithmetic(predicate: string, left: SemanticScalar, right?: SemanticScalar): SemanticScalar | undefined {
  // concat is a STRING producer — dispatch before the number guard below.
  if (predicate === 'concat') {
    return computeConcat(left, right)
  }

  // Inputs must BE numbers - no coercion. Number('')===0, Number(null)===0,
  // Number(true)===1 all slip past an isNaN-only check and come back
  // wearing the exactness contract they never earned (open-ended
  // local-model review, #29 open round). Canonical numeric strings were
  // already normalized to numbers at the working-memory boundary; whatever
  // is still a string here is not a number, and the literal fails.
  if (typeof left !== 'number' || Number.isNaN(left)) return undefined
  const a = left
  if (UNARY_ARITHMETIC.has(predicate)) {
    switch (predicate) {
      case 'neg': return guardResult(-a, a)
      case 'abs': return guardResult(Math.abs(a), a)
      case 'sign': return guardResult(Math.sign(a), a)
      // Transcendentals: IEEE best-effort by declaration (like non-integer div), with the SAME or-fail
      // guard — guardResult fails the non-finite domain/overflow cases so they never become fact args:
      // sqrt(-1)=NaN, ln(0)=-Infinity, ln(-1)=NaN, exp(710)=Infinity all return undefined (the literal
      // fails, loudly). Exact when the result lands clean (sqrt(4)=2, ln(1)=0, exp(0)=1); irrational
      // results (sqrt(2), ln(2), exp(1)) carry the usual IEEE representation, same contract as div(1,3).
      case 'sqrt': return guardResult(Math.sqrt(a), a)
      case 'ln': return guardResult(Math.log(a), a)
      case 'exp': return guardResult(Math.exp(a), a)
      default: return undefined
    }
  }
  if (typeof right !== 'number' || Number.isNaN(right)) return undefined
  const b = right
  switch (predicate) {
    case 'add': return guardResult(a + b, a, b)
    case 'sub': return guardResult(a - b, a, b)
    case 'mul': return guardResult(a * b, a, b)
    case 'div': return b === 0 ? undefined : guardResult(a / b, a, b)
    case 'mod': return b === 0 ? undefined : guardResult(a % b, a, b)
    // idiv = floor division (floors toward -Infinity, unlike JS truncation);
    // imod = mathematical modulo ((a%b)+b)%b, in [0,b) for any positive
    // divisor (JS `%` is a remainder that keeps the dividend's sign). The
    // identity a === idiv(a,b)*b + imod(a,b) holds for every sign combo.
    // Both fail loudly on divide-by-zero instead of yielding Infinity/NaN.
    // idiv/imod are INTEGER operations: an unsafe-integer input has already
    // lost precision (2^53 and 2^53+1 are the same float), so the result -
    // small though it may be - is computed from a corrupted operand. Reject
    // it (guardResult only catches unsafe RESULTS; these inputs need the
    // explicit check). Fractional inputs are fine: floor/modulo define them.
    case 'idiv':
    case 'imod': {
      if (b === 0) return undefined
      if ((Number.isInteger(a) && !Number.isSafeInteger(a)) || (Number.isInteger(b) && !Number.isSafeInteger(b))) {
        return undefined
      }
      const value = predicate === 'idiv' ? Math.floor(a / b) : ((a % b) + b) % b
      return guardResult(value, a, b)
    }
    case 'pow': return guardResult(a ** b, a, b)
    case 'min': return guardResult(Math.min(a, b), a, b)
    case 'max': return guardResult(Math.max(a, b), a, b)
    default: return undefined
  }
}

/**
 * Stringify a single concat part: strings pass through, safe integers are
 * converted to their canonical decimal form (5 → "5"). Anything else is
 * rejected — boolean/float-lossy/out-of-range — with a teaching error so
 * the model gets an actionable message rather than a silent rule non-firing.
 *
 * "Canonical integer string": same rule as working-memory normalisation —
 * a safe integer round-trips through String() without loss, so String(n)
 * is the canonical representation. Non-integer floats (0.5, 1.1) are
 * rejected because they are not part of the concat contract (use string
 * facts for formatted numbers).
 */
function concatPart(value: SemanticScalar, position: 'left' | 'right'): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number') {
    if (Number.isNaN(value) || !Number.isFinite(value)) {
      throw new Error(
        `concat ${position} must be a string or safe integer, got non-finite number ${value}. ` +
          `Pass a string fact or an exact integer (e.g. {"left": "?myStr", "right": "?myInt"}).`,
      )
    }
    if (!Number.isInteger(value)) {
      throw new Error(
        `concat ${position} must be a string or safe integer, got float ${value}. ` +
          `Floats cannot be canonically stringified — convert to a string fact first ` +
          `(e.g. assert_fact {predicate:"label", args:{text:"${value}"}}) and bind that string.`,
      )
    }
    if (!Number.isSafeInteger(value)) {
      throw new Error(
        `concat ${position} must be a safe integer (|n| ≤ 2^53-1), got ${value}. ` +
          `Integers outside ±${Number.MAX_SAFE_INTEGER} have lost precision and cannot be ` +
          `canonically represented — store large numbers as string facts instead.`,
      )
    }
    return String(value)
  }
  // boolean or any other type
  throw new Error(
    `concat ${position} must be a string or safe integer, got ${typeof value} ${JSON.stringify(value)}. ` +
      `Booleans and other non-scalar types are not valid concat parts. ` +
      `Use a string or integer fact argument (e.g. {"left": "?name", "right": "?suffix"}).`,
  )
}

function computeConcat(left: SemanticScalar, right: SemanticScalar | undefined): string | undefined {
  if (right === undefined) return undefined
  const l = concatPart(left, 'left')
  const r = concatPart(right, 'right')
  return l + r
}

/** Reject non-finite results, and integer results that left exact range. */
function guardResult(result: number, ...inputs: number[]): number | undefined {
  if (!Number.isFinite(result)) return undefined
  if (
    inputs.every((value) => Number.isInteger(value)) &&
    Number.isInteger(result) &&
    !Number.isSafeInteger(result)
  ) {
    return undefined
  }
  return result
}

function allNumbers(...values: SemanticScalar[]): boolean {
  return values.every((value) => typeof value === 'number')
}

function bothNumbers(left: SemanticScalar, right: SemanticScalar): boolean {
  return typeof left === 'number' && typeof right === 'number'
}
