import type { PredicateAtom, SemanticScalar } from '../model/types.js'
import {
  isBuiltinPredicate,
  isComparisonBuiltin,
  isArithmeticBuiltin,
  evaluateBuiltin,
  computeArithmetic,
  ARITHMETIC_RESULT_KEY,
} from './builtins.js'
import { isVariable, type RuleDefinition } from './predicate.js'

/**
 * Rule-quality lint (warnings, not errors): detect rules that have the
 * shape of a derivation but no deductive power — the body merely renames
 * the conclusion, so "deriving" the finding adds nothing a direct
 * assertion would not. The canonical offender, seen in real model runs,
 * is `IF suspected_resource_leak(file=?f) THEN finding(type=resource_leak, file=?f)`:
 * the body predicate echoes the head's discriminating constant.
 *
 * A rule has real filtering power when its body literals are MORE
 * primitive and independently checkable than the conclusion (e.g.
 * `empty_catch(file, line)`, `no_close_in_finally(file)`). These warnings
 * teach that distinction without blocking — false positives are harmless
 * noise, never a hard stop.
 */
export function detectVacuousRule(rule: RuleDefinition): string | undefined {
  const positiveBody = (rule.when ?? []).filter(
    (literal) => literal.naf !== true && !isBuiltinPredicate(literal.predicate),
  )
  const heads = rule.then ?? []
  if (positiveBody.length !== 1 || heads.length === 0) return undefined
  const body = positiveBody[0]
  if (!body) return undefined

  for (const head of heads) {
    // Signal 1 — identity rule: body and head are the same atom shape.
    if (body.predicate === head.predicate && sameArgShape(body, head)) {
      return (
        `rule "${rule.id}" looks vacuous: body and head are the same predicate ` +
        `"${body.predicate}" with the same arguments, so it derives nothing new. ` +
        `Use a more primitive, independently checkable observation in the body.`
      )
    }

    // Signal 2 — name echo: the head's discriminating constant value
    // appears inside the single body predicate's name (suspected_X -> finding(type=X)).
    // ALL meaningful tokens of the constant must occur in the body predicate:
    // suspected_resource_leak echoes finding(type=resource_leak), but
    // mutable_static_field_unsynchronized does NOT echo
    // finding(type=unsynchronized_access) - sharing one token is how
    // genuine observations naturally relate to their conclusion
    // (false positive seen in a real run, verification #8b).
    const bodyTokens = tokensOf(body.predicate)
    const echoed = headConstantValues(head).find((value) => {
      const valueTokens = tokensOf(value).filter((token) => token.length >= 4)
      return valueTokens.length > 0 && valueTokens.every((token) => bodyTokens.includes(token))
    })
    if (echoed !== undefined) {
      return (
        `rule "${rule.id}" looks vacuous: its only body literal "${body.predicate}(...)" ` +
        `just renames the conclusion "${head.predicate}(...=${echoed})", so the derivation ` +
        `adds no filtering power. Replace the body with a concrete code observation that is ` +
        `more primitive than the finding (e.g. empty_catch, no_close_in_finally, ` +
        `mutable_static_field_unsynchronized) — something you could be wrong about and check.`
      )
    }
  }

  return undefined
}

/**
 * Unfirable-rule lint (warning, not error): detect a rule whose body can
 * PROVABLY never be satisfied, so the rule is dead — it sits in the program
 * firing nothing, almost always a typo (a guard like eq(1,2), a wrong constant
 * arithmetic result, or a literal required both present and absent). ONLY
 * decidable, zero-false-positive signals: every flag rests on evaluating
 * constant operands, never a guess about what facts might exist. A literal with
 * any variable operand is left alone (its satisfiability is not statically
 * decidable). Like detectVacuousRule, this teaches without blocking.
 */
export function detectUnfirableRule(rule: RuleDefinition): string | undefined {
  const body = rule.when ?? []

  // Signal A: a positive builtin guard with ALL-constant operands that cannot hold.
  for (const lit of body) {
    if (lit.naf === true || !isBuiltinPredicate(lit.predicate)) continue
    const args = lit.args ?? {}
    const allConstant = Object.values(args).every((v) => !(typeof v === 'string' && isVariable(v)))
    if (!allConstant) continue

    if (isComparisonBuiltin(lit.predicate)) {
      if (!evaluateBuiltin(lit)) {
        return (
          `rule "${rule.id}" can never fire: body guard ${formatBuiltin(lit)} is always false ` +
          `(constant operands), so the closure can never satisfy it. Fix the constants or use variables.`
        )
      }
    } else if (isArithmeticBuiltin(lit.predicate)) {
      const result = args[ARITHMETIC_RESULT_KEY]
      const left = args.left
      // Only a fully-constant CHECK form is decidable; a variable result is a
      // producer (binds the result) and is always potentially firable.
      if (result === undefined || left === undefined) continue
      let computed: SemanticScalar | undefined
      try {
        computed = computeArithmetic(lit.predicate, left, args.right)
      } catch {
        continue // e.g. concat teaching-throw — a different diagnostic, not "unfirable"
      }
      if (computed === undefined || computed !== result) {
        return (
          `rule "${rule.id}" can never fire: body arithmetic ${formatBuiltin(lit)} cannot hold — ` +
          `${lit.predicate} of the given constants is ` +
          `${computed === undefined ? 'undefined/out-of-range' : String(computed)}, not ${String(result)}.`
        )
      }
    }
  }

  // Signal B: the same literal is required to BOTH hold and be absent (naf).
  const positives = body.filter((literal) => literal.naf !== true)
  const nafs = body.filter((literal) => literal.naf === true)
  for (const positive of positives) {
    for (const negative of nafs) {
      if (positive.predicate === negative.predicate && sameArgsExact(positive, negative)) {
        return (
          `rule "${rule.id}" can never fire: it requires ${positive.predicate}(...) to BOTH hold and ` +
          `be absent (naf) with identical arguments — a contradiction.`
        )
      }
    }
  }

  return undefined
}

/** Exact argument equality: same keys AND same values (variables compared by name). */
function sameArgsExact(left: PredicateAtom, right: PredicateAtom): boolean {
  const leftArgs = left.args ?? {}
  const rightArgs = right.args ?? {}
  const leftKeys = Object.keys(leftArgs).sort()
  const rightKeys = Object.keys(rightArgs).sort()
  if (leftKeys.length !== rightKeys.length) return false
  return leftKeys.every((key, index) => key === rightKeys[index] && leftArgs[key] === rightArgs[key])
}

function formatBuiltin(atom: PredicateAtom): string {
  const args = Object.entries(atom.args ?? {})
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(',')
  return `${atom.predicate}(${args})`
}

function sameArgShape(left: PredicateAtom, right: PredicateAtom): boolean {
  const leftKeys = Object.keys(left.args ?? {}).sort()
  const rightKeys = Object.keys(right.args ?? {}).sort()
  if (leftKeys.length !== rightKeys.length) return false
  return leftKeys.every((key, index) => key === rightKeys[index])
}

function headConstantValues(head: PredicateAtom): string[] {
  return Object.values(head.args ?? {})
    .filter((value): value is SemanticScalar & string => typeof value === 'string' && !isVariable(value))
}

/** Split a predicate or value into lowercase word tokens (snake/camel/kebab). */
function tokensOf(text: string): string[] {
  return text
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0)
}
