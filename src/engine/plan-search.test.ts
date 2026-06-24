import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { planToGoal } from './plan-search.js'
import { getLogicContext } from './logic-context.js'

/** Diamond pipeline: design -> {backend,frontend} -> integrate -> ship, each
 *  complete_X action gated on its predecessors' done(dep). Goal: done(ship). */
function diamond(extra: WorkingMemoryOperation[] = []): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const space = store.createSpace({ title: 'diamond' })
  const deps: Record<string, string[]> = {
    design: [], backend: ['design'], frontend: ['design'], integrate: ['backend', 'frontend'], ship: ['integrate'],
  }
  const ops: WorkingMemoryOperation[] = [
    { op: 'declare_goal', id: 'G', label: 'ship', desired: [{ predicate: 'done', args: { task: 'ship' } }] },
  ] as WorkingMemoryOperation[]
  for (const [id, ds] of Object.entries(deps)) {
    ops.push({ op: 'assert_fact', id: `T_${id}`, predicate: 'task', args: { id } } as WorkingMemoryOperation)
    ops.push({
      op: 'define_action', id: `complete_${id}`, action: `complete_${id}`, label: `complete ${id}`,
      preconditions: [
        { predicate: 'task', args: { id } },
        ...ds.map((d) => ({ predicate: 'done', args: { task: d } })),
      ],
      effects: [{ predicate: 'done', args: { task: id } }],
    } as WorkingMemoryOperation)
  }
  applyWorkingMemoryOperations(store, space.id, [...ops, ...extra])
  return { store, spaceId: space.id }
}

const idx = (plan: string[], a: string): number => plan.indexOf(a)

describe('planToGoal: bounded forward search, validated by the core', () => {
  it('finds a validated plan that reaches the goal', () => {
    const { store, spaceId } = diamond()
    const r = planToGoal(store, spaceId)
    assert.equal(r.found, true, r.note)
    assert.equal(r.validation?.ok, true, 'the returned plan must pass validatePlan')
    assert.ok(r.plan.includes('complete_ship'), 'plan must include the goal-producing action')
  })

  it('returns a valid topological order (prerequisites before dependents)', () => {
    const { store, spaceId } = diamond()
    const r = planToGoal(store, spaceId)
    assert.ok(idx(r.plan, 'complete_design') < idx(r.plan, 'complete_backend'))
    assert.ok(idx(r.plan, 'complete_design') < idx(r.plan, 'complete_frontend'))
    assert.ok(idx(r.plan, 'complete_backend') < idx(r.plan, 'complete_integrate'))
    assert.ok(idx(r.plan, 'complete_frontend') < idx(r.plan, 'complete_integrate'))
    assert.ok(idx(r.plan, 'complete_integrate') < idx(r.plan, 'complete_ship'))
  })

  it('does not mutate the real board (search is read-only)', () => {
    const { store, spaceId } = diamond()
    planToGoal(store, spaceId)
    const done = getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'done')
    assert.equal(done.length, 0, 'no done(...) fact should be committed to the real board by searching')
  })

  it('an already-satisfied goal returns the empty plan', () => {
    const { store, spaceId } = diamond([{ op: 'assert_fact', id: 'PRE', predicate: 'done', args: { task: 'ship' } }] as WorkingMemoryOperation[])
    const r = planToGoal(store, spaceId)
    assert.equal(r.found, true)
    assert.deepEqual(r.plan, [])
  })

  it('an unreachable goal returns found=false within the bound (no hang)', () => {
    // Goal wants done(moon) but no action produces it.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'unreachable' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'declare_goal', id: 'G', label: 'moon', desired: [{ predicate: 'done', args: { task: 'moon' } }] },
      { op: 'assert_fact', id: 'T', predicate: 'task', args: { id: 'design' } },
      {
        op: 'define_action', id: 'complete_design', action: 'complete_design', label: 'd',
        preconditions: [{ predicate: 'task', args: { id: 'design' } }],
        effects: [{ predicate: 'done', args: { task: 'design' } }],
      },
    ] as WorkingMemoryOperation[])
    const r = planToGoal(store, space.id, { maxDepth: 4 })
    assert.equal(r.found, false)
    assert.match(r.note ?? '', /no plan within depth/)
  })

  it('respects maxDepth: too shallow a bound finds nothing', () => {
    const { store, spaceId } = diamond()
    // The shortest plan is 5 steps; depth 2 cannot reach it.
    const r = planToGoal(store, spaceId, { maxDepth: 2 })
    assert.equal(r.found, false)
  })

  it('no declared goal: nothing to plan toward', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'no goal' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'T', predicate: 'task', args: { id: 'x' } },
    ] as WorkingMemoryOperation[])
    const r = planToGoal(store, space.id)
    assert.equal(r.found, false)
    assert.match(r.note ?? '', /no declared goal/)
  })
})
