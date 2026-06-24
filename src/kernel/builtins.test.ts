import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { evaluateStratifiedClosure } from './stratify.js'
import { computeArithmetic, evaluateBuiltin, isArithmeticBuiltin, isBuiltinPredicate } from './builtins.js'
import { validateRuleSafety } from './safety.js'

describe('exact-or-fail hardening: overflow / unsafe inputs fail rather than silently round', () => {
  it('pow that overflows or goes non-finite returns undefined (no silent Infinity/NaN)', () => {
    assert.equal(computeArithmetic('pow', 2, 1024), undefined) // overflows Number.MAX_VALUE -> Infinity
    assert.equal(computeArithmetic('pow', 10, 400), undefined)
    assert.equal(computeArithmetic('pow', -1, 0.5), undefined) // NaN
  })

  it('mul of two safe integers whose PRODUCT exceeds 2^53 fails (no silent precision loss)', () => {
    assert.equal(computeArithmetic('mul', Number.MAX_SAFE_INTEGER, 2), undefined)
    assert.equal(computeArithmetic('mul', 2 ** 52, 4), undefined) // 2^54, unsafe
    assert.equal(computeArithmetic('mul', 9381274, 6473), 60724986602) // safe result still works
  })

  it('add whose result exceeds 2^53 fails', () => {
    assert.equal(computeArithmetic('add', Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER), undefined)
    assert.equal(computeArithmetic('add', 2, 3), 5)
  })

  it('idiv/imod reject unsafe-integer INPUTS (corrupted operand, even if result would be small)', () => {
    assert.equal(computeArithmetic('idiv', 2 ** 53, 2), undefined) // 2^53 is not a safe integer
    assert.equal(computeArithmetic('imod', 2 ** 53 + 100, 7), undefined)
    assert.equal(computeArithmetic('idiv', 17, 5), 3) // safe inputs still work
    assert.equal(computeArithmetic('imod', -7, 3), 2) // mathematical modulo, non-negative
  })

  it('non-numeric / NaN / Infinity inputs do not produce a number', () => {
    assert.equal(computeArithmetic('add', Infinity, 1), undefined)
    assert.equal(computeArithmetic('add', NaN, 1), undefined)
  })
})

describe('sqrt / ln / exp (unary transcendental producers — finite-or-fail, IEEE by declaration)', () => {
  it('sqrt: exact on perfect squares, IEEE float otherwise, domain error fails', () => {
    assert.equal(computeArithmetic('sqrt', 4), 2)
    assert.equal(computeArithmetic('sqrt', 6.25), 2.5)
    assert.equal(computeArithmetic('sqrt', 0), 0)
    assert.equal(computeArithmetic('sqrt', 2), Math.sqrt(2)) // irrational → IEEE best-effort, by declaration
    assert.equal(computeArithmetic('sqrt', -1), undefined) // NaN domain error → fail (never a fact arg)
  })

  it('ln: 0 at 1, IEEE otherwise, non-positive domain errors fail', () => {
    assert.equal(computeArithmetic('ln', 1), 0)
    assert.equal(computeArithmetic('ln', 2), Math.log(2))
    assert.equal(computeArithmetic('ln', 0), undefined) // -Infinity → fail
    assert.equal(computeArithmetic('ln', -1), undefined) // NaN → fail
  })

  it('exp: 1 at 0, IEEE otherwise, overflow fails', () => {
    assert.equal(computeArithmetic('exp', 0), 1)
    assert.equal(computeArithmetic('exp', 1), Math.exp(1))
    assert.equal(computeArithmetic('exp', 2), Math.exp(2))
    assert.equal(computeArithmetic('exp', 710), undefined) // e^710 overflows to Infinity → fail
  })

  it('reject non-number inputs (no coercion) — same exactness contract as the other producers', () => {
    assert.equal(computeArithmetic('sqrt', '4'), undefined) // an un-normalized string is not a number here
    assert.equal(computeArithmetic('ln', true), undefined)
    assert.equal(computeArithmetic('exp', NaN), undefined)
  })

  it('registered as unary arithmetic producers', () => {
    for (const p of ['sqrt', 'ln', 'exp']) {
      assert.equal(isArithmeticBuiltin(p), true, `${p} is an arithmetic builtin`)
      assert.equal(isBuiltinPredicate(p), true, `${p} is a builtin predicate`)
    }
  })

  it('end-to-end in the closure: a rule binds sqrt(result) and derives a fact', () => {
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_ROOT',
          when: [
            { predicate: 'square', args: { n: '?n' } },
            { predicate: 'sqrt', args: { left: '?n', result: '?r' } },
          ],
          then: [{ predicate: 'root', args: { n: '?n', val: '?r' } }],
        },
      ],
      facts: [{ id: 'F_SQ', atom: { predicate: 'square', args: { n: 9 } } }],
    })
    assert.equal(result.derivations.length, 1)
    assert.deepEqual(result.derivations[0]?.atom, { predicate: 'root', args: { n: 9, val: 3 }, negated: undefined })
  })

  it('cannot appear in a rule head (reserved built-in producer)', () => {
    assert.match(
      validateRuleSafety({
        id: 'R_SQRT_HEAD',
        when: [{ predicate: 'a', args: { x: '?x' } }],
        then: [{ predicate: 'sqrt', args: { left: '?x', result: '?r' } }],
      }).join('; '),
      /reserved built-in/,
    )
  })
})

describe('guarded comparison built-ins', () => {
  it('evaluates comparisons against bound values during rule matching', () => {
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_WALKABLE',
          when: [
            { predicate: 'distance', args: { from: '?f', to: '?t', meters: '?m' } },
            { predicate: 'lte', args: { left: '?m', right: 500 } },
          ],
          then: [{ predicate: 'walkable', args: { from: '?f', to: '?t' } }],
        },
      ],
      facts: [
        { id: 'F1', atom: { predicate: 'distance', args: { from: 'home', to: 'car_wash', meters: 100 } } },
        { id: 'F2', atom: { predicate: 'distance', args: { from: 'home', to: 'airport', meters: 12000 } } },
      ],
    })

    assert.equal(result.derivations.length, 1)
    assert.deepEqual(result.derivations[0]?.atom, {
      predicate: 'walkable',
      args: { from: 'home', to: 'car_wash' },
      negated: undefined,
    })
  })

  it('rejects built-ins in rule heads and unbound built-in variables', () => {
    assert.match(
      validateRuleSafety({
        id: 'R_BAD_HEAD',
        when: [{ predicate: 'a', args: { item: '?x' } }],
        then: [{ predicate: 'lt', args: { left: '?x', right: 5 } }],
      }).join('; '),
      /reserved built-in/,
    )
    assert.match(
      validateRuleSafety({
        id: 'R_UNBOUND',
        when: [{ predicate: 'lt', args: { left: '?m', right: 5 } }],
        then: [{ predicate: 'small', args: { item: 'thing' } }],
      }).join('; '),
      /not bound/,
    )
    assert.match(
      validateRuleSafety({
        id: 'R_NEGATED_BUILTIN',
        when: [
          { predicate: 'a', args: { item: '?x' } },
          { predicate: 'lt', args: { left: '?x', right: 5 }, naf: true },
        ],
        then: [{ predicate: 'b', args: { item: '?x' } }],
      }).join('; '),
      /inverse comparison/,
    )
  })
})

describe('arithmetic input strictness (open-review #29 finding)', () => {
  it('refuses non-number inputs instead of coercing them to 0/1', () => {
    // Number('')===0, Number(null)===0, Number(true)===1, Number('  ')===0:
    // all slip past an isNaN-only check and come back looking exact -
    // found by the open-ended local-model review round.
    assert.equal(computeArithmetic('mul', '' as never, 5), undefined)
    assert.equal(computeArithmetic('add', true as never, 5), undefined)
    assert.equal(computeArithmetic('mul', '007' as never, 5), undefined)
    assert.equal(computeArithmetic('add', 2, '3' as never), undefined)
    assert.equal(computeArithmetic('neg', '' as never), undefined)
    assert.equal(computeArithmetic('min', 2, true as never), undefined)
    // and plain numbers still work exactly
    assert.equal(computeArithmetic('mul', 9381274, 6473), 60724986602)
  })
})


describe('integer division and mathematical modulo (idiv / imod)', () => {
  it('idiv floors toward -Infinity for positive and negative operands', () => {
    assert.equal(computeArithmetic('idiv', 7, 2), 3)
    assert.equal(computeArithmetic('idiv', 6, 2), 3)
    assert.equal(computeArithmetic('idiv', -7, 2), -4)
    assert.equal(computeArithmetic('idiv', 7, -2), -4)
    assert.equal(computeArithmetic('idiv', -7, -2), 3)
    assert.equal(computeArithmetic('idiv', 0, 5), 0)
  })

  it('imod is the mathematical modulo: non-negative for any positive divisor', () => {
    assert.equal(computeArithmetic('imod', 7, 3), 1)
    assert.equal(computeArithmetic('imod', -7, 3), 2)
    assert.equal(computeArithmetic('imod', -1, 3), 2)
    assert.equal(computeArithmetic('imod', 0, 5), 0)
    assert.equal(computeArithmetic('mod', -7, 3), -1)
  })

  it('imod follows the divisor sign for negative divisors (standard modulo)', () => {
    assert.equal(computeArithmetic('imod', 7, -3), -2)
    assert.equal(computeArithmetic('imod', -7, -3), -1)
  })

  it('the idiv/imod identity holds: a === idiv(a,b)*b + imod(a,b)', () => {
    for (const [a, b] of [[7, 3], [-7, 3], [7, -3], [-7, -3], [10, 4], [-1, 5]] as const) {
      const q = computeArithmetic('idiv', a, b) as number
      const r = computeArithmetic('imod', a, b) as number
      assert.equal(q * b + r, a, `failed for a=${a} b=${b}`)
    }
  })

  it('fails on divide-by-zero for both idiv and imod', () => {
    assert.equal(computeArithmetic('idiv', 7, 0), undefined)
    assert.equal(computeArithmetic('imod', 7, 0), undefined)
    assert.equal(computeArithmetic('idiv', 0, 0), undefined)
    assert.equal(computeArithmetic('imod', 0, 0), undefined)
  })

  it('refuses non-number inputs instead of coercing (no 0/1 from ""/null/true)', () => {
    assert.equal(computeArithmetic('idiv', '' as never, 5), undefined)
    assert.equal(computeArithmetic('idiv', 7, null as never), undefined)
    assert.equal(computeArithmetic('imod', true as never, 3), undefined)
    assert.equal(computeArithmetic('imod', 7, '3' as never), undefined)
  })

  it('honours the exact-or-fail boundary at +/-2^53', () => {
    assert.equal(computeArithmetic('idiv', Number.MAX_SAFE_INTEGER, 1), Number.MAX_SAFE_INTEGER)
    assert.equal(Number.isSafeInteger(2 ** 53), false)
    assert.equal(computeArithmetic('idiv', 2 ** 53, 1), undefined)
    assert.equal(computeArithmetic('imod', 2 ** 53, 7), undefined)
  })
})


describe('between (ternary closed-interval guard {value, low, high})', () => {
  it('is true when value is strictly inside the interval', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 5, low: 1, high: 10 } }), true)
  })
  it('is true at both inclusive endpoints', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 1, low: 1, high: 10 } }), true)
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 10, low: 1, high: 10 } }), true)
  })
  it('is false outside the interval on either side', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 0, low: 1, high: 10 } }), false)
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 11, low: 1, high: 10 } }), false)
  })
  it('is false for an inverted interval (low > high)', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 5, low: 10, high: 1 } }), false)
  })
  it('is false when any arg is not a number', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: '5', low: 1, high: 10 } }), false)
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 5, low: 'a', high: 10 } }), false)
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 5, low: 1, high: true } }), false)
  })
  it('is false when an arg is missing', () => {
    assert.equal(evaluateBuiltin({ predicate: 'between', args: { value: 5, low: 1 } }), false)
  })
  it('requires all three of value/low/high to be bound (range restriction)', () => {
    const violations = validateRuleSafety({
      id: 'R_BETWEEN_UNBOUND',
      when: [
        { predicate: 'reading', args: { sensor: '?s', temp: '?t' } },
        { predicate: 'between', args: { value: '?t', low: '?lo', high: '?hi' } },
      ],
      then: [{ predicate: 'in_range', args: { sensor: '?s' } }],
    }).join('; ')
    assert.match(violations, /\?lo .*not bound/)
    assert.match(violations, /\?hi .*not bound/)
  })
  it('collapses two lt/gt guards into one between in a rule body (end-to-end closure)', () => {
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_IN_RANGE',
          when: [
            { predicate: 'reading', args: { sensor: '?s', temp: '?t' } },
            { predicate: 'between', args: { value: '?t', low: 20, high: 25 } },
          ],
          then: [{ predicate: 'comfortable', args: { sensor: '?s' } }],
        },
      ],
      facts: [
        { id: 'F1', atom: { predicate: 'reading', args: { sensor: 'living', temp: 22 } } },
        { id: 'F2', atom: { predicate: 'reading', args: { sensor: 'attic', temp: 31 } } },
        { id: 'F3', atom: { predicate: 'reading', args: { sensor: 'cellar', temp: 20 } } },
      ],
    })
    const derived = result.derivations.map((d) => d.atom.args?.sensor).sort()
    assert.deepEqual(derived, ['cellar', 'living'])
  })
})

describe('concat (binary string producer {left, right, result})', () => {
  it('joins two strings into a result string', () => {
    // Non-vacuous: verifies the concatenation actually happens — a no-op
    // identity impl ("left") would return "hello" not "hello world".
    assert.equal(computeArithmetic('concat', 'hello', ' world'), 'hello world')
    assert.equal(computeArithmetic('concat', 'foo', 'bar'), 'foobar')
  })

  it('joins a string and a safe integer (integer stringified canonically)', () => {
    // Non-vacuous: verifies integer → "5" conversion, not just string passthrough.
    // A wrong impl that returned the number 5 would fail the strict equality.
    assert.equal(computeArithmetic('concat', 'item-', 5), 'item-5')
    assert.equal(computeArithmetic('concat', 42, '!'), '42!')
    assert.equal(computeArithmetic('concat', -7, 'x'), '-7x')
    // MAX_SAFE_INTEGER exactly — the boundary that must pass
    assert.equal(computeArithmetic('concat', Number.MAX_SAFE_INTEGER, ''), String(Number.MAX_SAFE_INTEGER))
  })

  it('joins two empty strings (edge: empty parts)', () => {
    // Non-vacuous: '' + '' = '' is not the same as returning left or right.
    // An impl that returned left would pass "empty left" but fail "empty right" below.
    assert.equal(computeArithmetic('concat', '', ''), '')
    assert.equal(computeArithmetic('concat', '', 'suffix'), 'suffix')
    assert.equal(computeArithmetic('concat', 'prefix', ''), 'prefix')
  })

  it('joins a single non-empty part on either side (edge: single-meaningful part)', () => {
    // Non-vacuous: confirms the binary shape is correct and order matters.
    // An impl that sorted or swapped args would fail this.
    assert.equal(computeArithmetic('concat', 'a', 'b'), 'ab')
    assert.equal(computeArithmetic('concat', 'b', 'a'), 'ba')
    assert.notEqual(
      computeArithmetic('concat', 'a', 'b'),
      computeArithmetic('concat', 'b', 'a'),
    )
  })

  it('throws a teaching error when left is a boolean', () => {
    // Non-vacuous: boolean is SemanticScalar; without the check, true
    // would fall through to the typeof === 'number' branch (false) and
    // be silently rejected (undefined). We want a loud teaching error instead.
    assert.throws(
      () => computeArithmetic('concat', true as never, 'x'),
      /concat left must be a string or safe integer.*boolean/,
    )
  })

  it('throws a teaching error when right is a non-integer float', () => {
    // Non-vacuous: 1.5 IS a finite number so a naive impl might stringify it
    // as "1.5" and silently pass. We specifically reject non-integer floats.
    assert.throws(
      () => computeArithmetic('concat', 'v', 1.5),
      /concat right must be a string or safe integer.*float/,
    )
  })

  it('throws a teaching error when either part is an out-of-range integer', () => {
    // Non-vacuous: 2^53 passes Number.isInteger but not Number.isSafeInteger.
    // Without the safe-integer guard it would stringify to "9007199254740992"
    // which is a value that could silently be wrong (2^53 and 2^53+1 are the
    // same float). We require the caller to pass a string fact for large IDs.
    assert.throws(
      () => computeArithmetic('concat', 2 ** 53, 'x'),
      /concat left must be a safe integer/,
    )
    assert.throws(
      () => computeArithmetic('concat', 'x', -(2 ** 53)),
      /concat right must be a safe integer/,
    )
  })

  it('returns undefined when right is missing (malformed: not an error, just no-match)', () => {
    // Non-vacuous: undefined right must not throw (it is not a wrong-type
    // input — it is the unary call shape, which means the rule body did not
    // provide the right arg). Mirrors how binary arithmetic handles missing
    // right (returns undefined, making the rule literal fail silently).
    assert.equal(computeArithmetic('concat', 'x', undefined), undefined)
  })

  it('result variable is bound in a rule body (safety + closure end-to-end)', () => {
    // Non-vacuous: if concat were not in ARITHMETIC_BUILTINS, safety would
    // reject the result variable as unbound; if computeArithmetic returned
    // undefined for correct inputs, the derivation would never be produced.
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_LABEL',
          when: [
            { predicate: 'item', args: { prefix: '?p', id: '?n' } },
            { predicate: 'concat', args: { left: '?p', right: '?n', result: '?label' } },
          ],
          then: [{ predicate: 'labeled', args: { label: '?label' } }],
        },
      ],
      facts: [
        { id: 'F1', atom: { predicate: 'item', args: { prefix: 'item-', id: 'A' } } },
        { id: 'F2', atom: { predicate: 'item', args: { prefix: 'item-', id: 'B' } } },
      ],
    })
    assert.deepEqual(
      result.derivations.map((d) => d.atom.args?.label).sort(),
      ['item-A', 'item-B'],
    )
  })

  it('result variable is bound in a rule body with integer right operand (end-to-end)', () => {
    // Non-vacuous: verifies the integer→string canonicalisation actually works
    // during closure evaluation (not just in the unit call above).
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_TAG',
          when: [
            { predicate: 'widget', args: { name: '?n', rev: '?r' } },
            { predicate: 'concat', args: { left: '?n', right: '?r', result: '?tag' } },
          ],
          then: [{ predicate: 'tag', args: { value: '?tag' } }],
        },
      ],
      facts: [
        { id: 'F1', atom: { predicate: 'widget', args: { name: 'v', rev: 3 } } },
      ],
    })
    assert.deepEqual(result.derivations.map((d) => d.atom.args?.value), ['v3'])
  })

  it('blocks concat in a rule head (reserved built-in)', () => {
    // Non-vacuous: verifies that adding concat to ARITHMETIC_BUILTINS
    // (which is part of BUILTIN_PREDICATES) causes safety to reject it in a head.
    assert.match(
      validateRuleSafety({
        id: 'R_BAD_HEAD',
        when: [{ predicate: 'item', args: { x: '?x' } }],
        then: [{ predicate: 'concat', args: { left: '?x', right: '?x', result: '?r' } }],
      }).join('; '),
      /reserved built-in/,
    )
  })

  it('unbound result variable is accepted (result is an OUTPUT, not an input)', () => {
    // Non-vacuous: if concat were in COMPARISON_BUILTINS instead of
    // ARITHMETIC_BUILTINS, safety would demand ALL variables — including
    // result — be bound by a positive body literal first and would
    // incorrectly reject a rule that uses concat correctly.
    const violations = validateRuleSafety({
      id: 'R_OK',
      when: [
        { predicate: 'item', args: { prefix: '?p', id: '?i' } },
        { predicate: 'concat', args: { left: '?p', right: '?i', result: '?out' } },
      ],
      then: [{ predicate: 'label', args: { text: '?out' } }],
    })
    assert.deepEqual(violations, [])
  })
})

describe('contains (string membership guard {left, right})', () => {
  it('is true when left contains right as a substring', () => {
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'hello world', right: 'o w' } }), true)
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'abc', right: 'abc' } }), true)
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'abc', right: '' } }), true)
  })
  it('is false when the substring is absent', () => {
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'hello', right: 'xyz' } }), false)
  })
  it('is false (no coercion) when either arg is not a string', () => {
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 123, right: '2' } }), false)
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'a1', right: 1 } }), false)
    assert.equal(evaluateBuiltin({ predicate: 'contains', args: { left: 'x' } }), false)
  })
  it('requires both left and right bound (range restriction)', () => {
    const violations = validateRuleSafety({
      id: 'R_CONTAINS_UNBOUND',
      when: [
        { predicate: 'log', args: { line: '?l' } },
        { predicate: 'contains', args: { left: '?l', right: '?needle' } },
      ],
      then: [{ predicate: 'hit', args: { line: '?l' } }],
    }).join('; ')
    assert.match(violations, /\?needle .*not bound/)
  })
  it('filters in a rule body end-to-end (closure)', () => {
    const result = evaluateStratifiedClosure({
      rules: [
        {
          id: 'R_ERR',
          when: [
            { predicate: 'log', args: { id: '?i', text: '?t' } },
            { predicate: 'contains', args: { left: '?t', right: 'ERROR' } },
          ],
          then: [{ predicate: 'flagged', args: { id: '?i' } }],
        },
      ],
      facts: [
        { id: 'F1', atom: { predicate: 'log', args: { id: 'a', text: 'ERROR: boom' } } },
        { id: 'F2', atom: { predicate: 'log', args: { id: 'b', text: 'all good' } } },
        { id: 'F3', atom: { predicate: 'log', args: { id: 'c', text: 'minor ERROR here' } } },
      ],
    })
    assert.deepEqual(result.derivations.map((d) => d.atom.args?.id).sort(), ['a', 'c'])
  })
})
