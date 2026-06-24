import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { suggestPlanRepairs } from './plan-repair.js'
import { getLogicContext } from './logic-context.js'

/**
 * A linear pipeline gated DIRECTLY on the predecessor's effect fact (done(dep)),
 * which is exactly the shape repair search can mend: complete_X produces
 * done(X), and complete_Y requires done(X). Plans that skip a prerequisite can
 * be repaired by inserting the producer chain.
 */
function pipelineSpace(extra: WorkingMemoryOperation[] = []): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const space = store.createSpace({ title: 'pipeline' })
  applyWorkingMemoryOperations(store, space.id, [
    { op: 'declare_goal', id: 'G', label: 'ship test', desired: [{ predicate: 'done', args: { task: 'test' } }] },
    { op: 'assert_fact', id: 'T_d', predicate: 'task', args: { id: 'design' } },
    { op: 'assert_fact', id: 'T_i', predicate: 'task', args: { id: 'impl' } },
    { op: 'assert_fact', id: 'T_t', predicate: 'task', args: { id: 'test' } },
    {
      op: 'define_action', id: 'complete_design', action: 'complete_design', label: 'complete design',
      preconditions: [{ predicate: 'task', args: { id: 'design' } }],
      effects: [{ predicate: 'done', args: { task: 'design' } }],
    },
    {
      op: 'define_action', id: 'complete_impl', action: 'complete_impl', label: 'complete impl',
      preconditions: [{ predicate: 'task', args: { id: 'impl' } }, { predicate: 'done', args: { task: 'design' } }],
      effects: [{ predicate: 'done', args: { task: 'impl' } }],
    },
    {
      op: 'define_action', id: 'complete_test', action: 'complete_test', label: 'complete test',
      preconditions: [{ predicate: 'task', args: { id: 'test' } }, { predicate: 'done', args: { task: 'impl' } }],
      effects: [{ predicate: 'done', args: { task: 'test' } }],
    },
    ...extra,
  ] as WorkingMemoryOperation[])
  return { store, spaceId: space.id }
}

function doneCount(store: MemorySpaceStore, spaceId: string): number {
  return getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'done').length
}

describe('suggestPlanRepairs: board-grounded plan repair', () => {
  it('single-hop: inserts the one producer whose own preconditions already hold', () => {
    const { store, spaceId } = pipelineSpace([
      { op: 'assert_fact', id: 'D_design', predicate: 'done', args: { task: 'design' } },
    ])
    const r = suggestPlanRepairs(store, spaceId, ['complete_test'])
    assert.equal(r.failedIndex, 0)
    assert.equal(r.failedPrecondition?.predicate, 'done')
    assert.equal(r.failedPrecondition?.args?.task, 'impl')
    assert.ok(r.repairs.length > 0, r.note)
    const best = r.repairs[0]!
    assert.deepEqual(best.insertedActionNodeIds, ['complete_impl'])
    assert.deepEqual(best.actionNodeIds, ['complete_impl', 'complete_test'])
    assert.equal(best.validates, true, 'the repaired plan reaches the goal')
  })

  it('multi-hop: chases the producer chain (design -> impl) to unblock test', () => {
    const { store, spaceId } = pipelineSpace()
    const r = suggestPlanRepairs(store, spaceId, ['complete_test'])
    assert.ok(r.repairs.length > 0, r.note)
    const best = r.repairs[0]!
    assert.deepEqual(best.insertedActionNodeIds, ['complete_design', 'complete_impl'])
    assert.deepEqual(best.actionNodeIds, ['complete_design', 'complete_impl', 'complete_test'])
    assert.equal(best.validates, true)
  })

  it('suggestion only: the real board is never mutated by a repair search', () => {
    const { store, spaceId } = pipelineSpace()
    assert.equal(doneCount(store, spaceId), 0)
    suggestPlanRepairs(store, spaceId, ['complete_test'])
    assert.equal(doneCount(store, spaceId), 0, 'no done(...) fact was committed to the real board')
  })

  it('respects maxDepth: a chain deeper than the bound yields no repair', () => {
    const { store, spaceId } = pipelineSpace()
    // The fix needs 2 hops (design then impl); depth 1 cannot reach it.
    const r = suggestPlanRepairs(store, spaceId, ['complete_test'], undefined, { maxDepth: 1 })
    assert.equal(r.repairs.length, 0)
    assert.match(r.note ?? '', /no defined action produces/)
  })

  it('no actionable producer: a precondition nothing produces returns a teaching note', () => {
    const { store, spaceId } = pipelineSpace([
      {
        op: 'define_action', id: 'complete_extra', action: 'complete_extra', label: 'needs an external token',
        preconditions: [{ predicate: 'external', args: { kind: 'token' } }],
        effects: [{ predicate: 'done', args: { task: 'extra' } }],
      },
    ])
    const r = suggestPlanRepairs(store, spaceId, ['complete_extra'])
    assert.equal(r.failedIndex, 0)
    assert.equal(r.repairs.length, 0)
    assert.match(r.note ?? '', /no defined action produces external\(/)
  })

  it('plan already valid: nothing to repair', () => {
    const { store, spaceId } = pipelineSpace()
    const r = suggestPlanRepairs(store, spaceId, ['complete_design', 'complete_impl', 'complete_test'])
    assert.equal(r.failedIndex, undefined)
    assert.equal(r.repairs.length, 0)
    assert.match(r.note ?? '', /already validates/)
  })
})
