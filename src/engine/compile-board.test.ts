import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { RuleDefinition } from '../kernel/predicate.js'
import { atomKey } from '../kernel/predicate.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import { compileBoard, CompileBoardError, type Fact } from './compile-board.js'

/**
 * Roundtrip oracle: derive the spine via the BOARD's own closure (the ground
 * truth) and compare to the COMPILED pipeline. Equality of the derived-fact
 * sets (by atomKey over predicate+args) is the faithfulness proof.
 */
function boardDerived(rules: RuleDefinition[], baseFacts: Fact[], predicates: string[]): Set<string> {
  const store = new MemorySpaceStore()
  const { id: spaceId } = store.createSpace({ title: 'compile-board roundtrip' })
  const ops: WorkingMemoryOperation[] = [
    ...baseFacts.map(
      (f, i) =>
        ({ op: 'assert_fact', id: `base_${i}`, predicate: f.predicate, args: f.args } as WorkingMemoryOperation),
    ),
    ...rules.map(
      (r) => ({ op: 'add_axiom', id: r.id, label: r.id, when: r.when, then: r.then } as WorkingMemoryOperation),
    ),
  ]
  applyWorkingMemoryOperations(store, spaceId, ops)
  const ctx = getLogicContext(store, spaceId)
  const want = new Set(predicates)
  const out = new Set<string>()
  for (const fact of ctx.facts) {
    if (fact.derived && want.has(fact.atom.predicate)) {
      out.add(atomKey({ predicate: fact.atom.predicate, args: fact.atom.args }))
    }
  }
  return out
}

function compiledDerived(rules: RuleDefinition[], baseFacts: Fact[], predicates: string[]): Set<string> {
  const want = new Set(predicates)
  const derived = compileBoard(rules).pipeline(baseFacts)
  const out = new Set<string>()
  for (const f of derived) {
    if (want.has(f.predicate)) out.add(atomKey({ predicate: f.predicate, args: f.args }))
  }
  return out
}

function assertRoundtrip(rules: RuleDefinition[], baseFacts: Fact[], predicates: string[]): void {
  const board = boardDerived(rules, baseFacts, predicates)
  const compiled = compiledDerived(rules, baseFacts, predicates)
  assert.deepEqual(
    [...compiled].sort(),
    [...board].sort(),
    `compiled derived set must EQUAL the board's closure derived set\n` +
      `  board:    ${[...board].sort().join('\n            ') || '(none)'}\n` +
      `  compiled: ${[...compiled].sort().join('\n            ') || '(none)'}`,
  )
  // Non-vacuous: the board must actually derive something.
  assert.ok(board.size > 0, 'board must derive at least one fact (else the roundtrip is vacuous)')
}

test('roundtrip: simple 2-layer chain (join then chain)', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'r_parent',
      when: [
        { predicate: 'parent', args: { p: '?a', c: '?b' } },
        { predicate: 'parent', args: { p: '?b', c: '?d' } },
      ],
      then: [{ predicate: 'grandparent', args: { gp: '?a', gc: '?d' } }],
    },
    {
      id: 'r_ancestor',
      when: [{ predicate: 'grandparent', args: { gp: '?x', gc: '?y' } }],
      then: [{ predicate: 'ancestor', args: { a: '?x', d: '?y' } }],
    },
  ]
  const base: Fact[] = [
    { predicate: 'parent', args: { p: 'al', c: 'bo' } },
    { predicate: 'parent', args: { p: 'bo', c: 'cy' } },
    { predicate: 'parent', args: { p: 'cy', c: 'di' } },
  ]
  assertRoundtrip(rules, base, ['grandparent', 'ancestor'])
})

test('roundtrip: NAF anti-join (no matching fact exists)', () => {
  // valid_order iff order has no invalid reason recorded (NAF over has_invalid).
  const rules: RuleDefinition[] = [
    {
      id: 'r_has_invalid',
      when: [{ predicate: 'flag', args: { order: '?o', kind: 'bad' } }],
      then: [{ predicate: 'has_invalid', args: { order: '?o' } }],
    },
    {
      id: 'r_valid',
      when: [
        { predicate: 'order', args: { id: '?o' } },
        { predicate: 'has_invalid', args: { order: '?o' }, naf: true },
      ],
      then: [{ predicate: 'valid', args: { id: '?o' } }],
    },
  ]
  const base: Fact[] = [
    { predicate: 'order', args: { id: 'A' } },
    { predicate: 'order', args: { id: 'B' } },
    { predicate: 'flag', args: { order: 'B', kind: 'bad' } },
  ]
  // Expect: A valid (no flag), B not valid; has_invalid only for B.
  const compiled = compileBoard(rules).pipeline(base)
  const validIds = compiled.filter((f) => f.predicate === 'valid').map((f) => f.args.id)
  assert.deepEqual(validIds.sort(), ['A'], 'only A (no bad flag) is valid via NAF anti-join')
  assertRoundtrip(rules, base, ['has_invalid', 'valid'])
})

test('roundtrip: comparison builtin (gte guard)', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'r_big',
      when: [
        { predicate: 'order', args: { id: '?o', amount: '?a' } },
        { predicate: 'gte', args: { left: '?a', right: 200 } },
      ],
      then: [{ predicate: 'big_order', args: { id: '?o' } }],
    },
  ]
  const base: Fact[] = [
    { predicate: 'order', args: { id: 'A', amount: 250 } },
    { predicate: 'order', args: { id: 'B', amount: 100 } },
    { predicate: 'order', args: { id: 'C', amount: 200 } },
  ]
  const compiled = compileBoard(rules).pipeline(base)
  const big = compiled.filter((f) => f.predicate === 'big_order').map((f) => f.args.id)
  assert.deepEqual(big.sort(), ['A', 'C'], 'gte(amount, 200) selects A(250) and C(200), not B(100)')
  assertRoundtrip(rules, base, ['big_order'])
})

test('fail-visible: arithmetic builtin in body is rejected', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'r_arith',
      when: [
        { predicate: 'order', args: { id: '?o', amount: '?a' } },
        { predicate: 'add', args: { left: '?a', right: 10, result: '?t' } },
      ],
      then: [{ predicate: 'total', args: { id: '?o', t: '?t' } }],
    },
  ]
  assert.throws(
    () => compileBoard(rules),
    (err: unknown) =>
      err instanceof CompileBoardError &&
      /r_arith/.test((err as Error).message) &&
      /add/.test((err as Error).message),
    'must throw CompileBoardError naming the rule and the "add" builtin',
  )
})

test('fail-visible: recursive rule is rejected', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'r_reach_base',
      when: [{ predicate: 'edge', args: { from: '?x', to: '?y' } }],
      then: [{ predicate: 'reach', args: { from: '?x', to: '?y' } }],
    },
    {
      id: 'r_reach_step',
      when: [
        { predicate: 'reach', args: { from: '?x', to: '?y' } },
        { predicate: 'edge', args: { from: '?y', to: '?z' } },
      ],
      then: [{ predicate: 'reach', args: { from: '?x', to: '?z' } }],
    },
  ]
  assert.throws(
    () => compileBoard(rules),
    (err: unknown) =>
      err instanceof CompileBoardError && /recursive/.test((err as Error).message) && /reach/.test((err as Error).message),
    'must throw CompileBoardError naming the recursive predicate',
  )
})
