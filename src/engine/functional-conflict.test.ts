/**
 * functional-conflict.test — integration RED test for functional-dependency
 * conflicts in the live logic-context (the arith p10 pollution: a derived cost
 * and a bare-asserted cost for the same item coexisting). A domain board fact
 * functional_dependency(predicate, key) turns "same key, different value" into a
 * visible conflict + disputed taint. Inert without the declaration (zero regression).
 */
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext, formatLogicContextAsText } from './logic-context.js'

function seed(ops: WorkingMemoryOperation[]) {
  const store = new MemorySpaceStore()
  const { id } = store.createSpace({ title: 'fd-conflict' })
  applyWorkingMemoryOperations(store, id, ops, { source: 'system' })
  return getLogicContext(store, id)
}

const COSTS: WorkingMemoryOperation[] = [
  { op: 'assert_fact', id: 'A_sensor', predicate: 'cost', args: { item: 'sensor', total: 388752151850 } },
  { op: 'assert_fact', id: 'D_sensor', predicate: 'cost', args: { item: 'sensor', total: 3886450604850 } },
] as WorkingMemoryOperation[]
const FD: WorkingMemoryOperation = {
  op: 'assert_fact', id: 'FD', predicate: 'functional_dependency', args: { predicate: 'cost', key: 'item' },
} as WorkingMemoryOperation

describe('functional-dependency conflict in logic-context (arith p10 pollution)', () => {
  it('a declared functional_dependency turns two disagreeing costs into a visible conflict + disputed taint', () => {
    const ctx = seed([FD, ...COSTS])
    assert.equal(ctx.functionalConflicts.length, 1, JSON.stringify(ctx.functionalConflicts))
    assert.deepEqual(ctx.functionalConflicts[0]!.factIds, ['A_sensor', 'D_sensor'])
    const disputed = ctx.facts.filter((f) => f.disputed).map((f) => f.nodeId).sort()
    assert.deepEqual(disputed, ['A_sensor', 'D_sensor'], 'both disagreeing costs are tainted disputed')
    assert.match(formatLogicContextAsText(ctx), /functional conflict: cost\(item=sensor\)/)
    assert.ok(ctx.critique.some((c) => c.kind === 'functional_conflict'), 'board_critique surfaces a functional_conflict item')
  })

  it('inert without the declaration (zero regression for boards that never declare one)', () => {
    const ctx = seed(COSTS)
    assert.equal(ctx.functionalConflicts.length, 0)
    assert.equal(ctx.facts.some((f) => f.disputed), false)
  })

  it('identical value is not a conflict (a re-derivation of the same total)', () => {
    const ctx = seed([
      FD,
      { op: 'assert_fact', id: 'A', predicate: 'cost', args: { item: 'x', total: 5 } },
      { op: 'assert_fact', id: 'B', predicate: 'cost', args: { item: 'x', total: 5 } },
    ] as WorkingMemoryOperation[])
    assert.equal(ctx.functionalConflicts.length, 0)
  })

  it('② derive_aggregate refuses a source that has a functional conflict (no silent double-count)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'fd-agg' })
    applyWorkingMemoryOperations(store, id, [FD, ...COSTS], { source: 'system' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store,
          id,
          [{ op: 'derive_aggregate', id: 'agg', source: { predicate: 'cost', valueArg: 'total' }, into: { predicate: 'grand_total', valueArg: 'value' } }] as WorkingMemoryOperation[],
          { source: 'system' },
        ),
      /functional conflict/,
      'aggregating a conflicted source is refused with a teaching error',
    )
  })

  it('② derive_aggregate proceeds when the source is clean (one value per key)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'fd-agg-clean' })
    applyWorkingMemoryOperations(
      store,
      id,
      [
        FD,
        { op: 'assert_fact', id: 'c_a', predicate: 'cost', args: { item: 'a', total: 10 } },
        { op: 'assert_fact', id: 'c_b', predicate: 'cost', args: { item: 'b', total: 20 } },
      ] as WorkingMemoryOperation[],
      { source: 'system' },
    )
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'derive_aggregate', id: 'agg', source: { predicate: 'cost', valueArg: 'total' }, into: { predicate: 'grand_total', valueArg: 'value' } }] as WorkingMemoryOperation[],
      { source: 'system' },
    )
    const ctx = getLogicContext(store, id)
    assert.ok(
      ctx.facts.some((f) => f.atom.predicate === 'grand_total' && Number(f.atom.args?.value) === 30 && f.derived),
      'clean source aggregates to 30',
    )
  })
})
