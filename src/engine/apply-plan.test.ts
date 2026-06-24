import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { applyPlan } from './apply-plan.js'
import { getLogicContext } from './logic-context.js'

/** Diamond plan: design -> {backend,frontend} -> integrate -> ship, complete
 *  actions gated on the derived ready(task). */
function planSpace(): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const space = store.createSpace({ title: 'plan' })
  const tasks = ['design', 'backend', 'frontend', 'integrate', 'ship']
  applyWorkingMemoryOperations(store, space.id, [
    { op: 'declare_goal', id: 'G1', label: 'ship', desired: [{ predicate: 'done', args: { task: 'ship' } }] },
    ...tasks.map((id) => ({ op: 'assert_fact' as const, id: `T_${id}`, predicate: 'task', args: { id } })),
    { op: 'assert_fact', id: 'R_BE', predicate: 'requires', args: { task: 'backend', dep: 'design' } },
    { op: 'assert_fact', id: 'R_FE', predicate: 'requires', args: { task: 'frontend', dep: 'design' } },
    { op: 'assert_fact', id: 'R_IB', predicate: 'requires', args: { task: 'integrate', dep: 'backend' } },
    { op: 'assert_fact', id: 'R_IF', predicate: 'requires', args: { task: 'integrate', dep: 'frontend' } },
    { op: 'assert_fact', id: 'R_SH', predicate: 'requires', args: { task: 'ship', dep: 'integrate' } },
    {
      op: 'add_axiom', id: 'AX_BLOCKED', label: 'blocked',
      when: [
        { predicate: 'requires', args: { task: '?t', dep: '?d' } },
        { predicate: 'done', args: { task: '?d' }, naf: true },
      ],
      then: [{ predicate: 'blocked', args: { task: '?t' } }],
    },
    {
      op: 'add_axiom', id: 'AX_READY', label: 'ready',
      when: [
        { predicate: 'task', args: { id: '?t' } },
        { predicate: 'done', args: { task: '?t' }, naf: true },
        { predicate: 'blocked', args: { task: '?t' }, naf: true },
      ],
      then: [{ predicate: 'ready', args: { task: '?t' } }],
    },
    ...tasks.map((id) => ({
      op: 'define_action' as const, id: `complete_${id}`, action: `complete_${id}`, label: `complete ${id}`,
      preconditions: [{ predicate: 'ready', args: { task: id } }],
      effects: [{ predicate: 'done', args: { task: id } }],
    })),
  ] as WorkingMemoryOperation[])
  return { store, spaceId: space.id }
}

function doneSet(store: MemorySpaceStore, spaceId: string): Set<string> {
  return new Set(
    getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'done').map((f) => String(f.atom.args?.task)),
  )
}

describe('applyPlan: validate then guard-commit a whole sequence', () => {
  it('commits a correct order to the real board and reaches the goal', () => {
    const { store, spaceId } = planSpace()
    const r = applyPlan(store, spaceId, ['complete_design', 'complete_backend', 'complete_frontend', 'complete_integrate', 'complete_ship'])
    assert.equal(r.applied, true, r.failureReason)
    assert.deepEqual(r.appliedActionNodeIds.length, 5)
    assert.equal(doneSet(store, spaceId).has('ship'), true, 'ship is done on the REAL board')
    assert.equal(getLogicContext(store, spaceId).goals.find((g) => g.nodeId === 'G1')?.satisfied, true)
    assert.equal(r.steps.length, 5)
  })

  it('an out-of-order plan validates false and commits NOTHING', () => {
    const { store, spaceId } = planSpace()
    const r = applyPlan(store, spaceId, ['complete_ship', 'complete_design'])
    assert.equal(r.applied, false)
    assert.equal(r.failedIndex, 0)
    assert.deepEqual(r.appliedActionNodeIds, [])
    assert.equal(doneSet(store, spaceId).size, 0, 'nothing committed when validation fails up front')
  })

  it('requireGoals: a valid prefix that misses the goal commits nothing', () => {
    const { store, spaceId } = planSpace()
    const r = applyPlan(store, spaceId, ['complete_design', 'complete_backend'], { requireGoals: true })
    assert.equal(r.applied, false)
    assert.match(r.failureReason ?? '', /does not reach all goals/)
    assert.deepEqual(r.appliedActionNodeIds, [])
    assert.equal(doneSet(store, spaceId).size, 0)
  })

  it('without requireGoals, a valid prefix commits in order (goal need not be reached)', () => {
    const { store, spaceId } = planSpace()
    const r = applyPlan(store, spaceId, ['complete_design', 'complete_backend'])
    assert.equal(r.applied, true, r.failureReason)
    assert.deepEqual([...doneSet(store, spaceId)].sort(), ['backend', 'design'])
  })

  it('the final revision changes as the board advances (drift token)', () => {
    const { store, spaceId } = planSpace()
    // Empty plan = no-op; its finalRevision is the board's start revision.
    const start = applyPlan(store, spaceId, []).finalRevision
    const r = applyPlan(store, spaceId, ['complete_design'])
    assert.equal(r.applied, true)
    assert.notEqual(r.finalRevision, start, 'committing a step must change the board revision')
    // And a second committed step advances it again.
    const r2 = applyPlan(store, spaceId, ['complete_backend'])
    assert.equal(r2.applied, true)
    assert.notEqual(r2.finalRevision, r.finalRevision, 'each committed step advances the revision')
  })

  it('an empty plan is a no-op success (nothing to commit)', () => {
    const { store, spaceId } = planSpace()
    const r = applyPlan(store, spaceId, [])
    assert.equal(r.applied, true)
    assert.deepEqual(r.appliedActionNodeIds, [])
    assert.equal(doneSet(store, spaceId).size, 0)
  })
})
