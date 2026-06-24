import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { validatePlan } from './validate-plan.js'
import { getLogicContext } from './logic-context.js'

/** The diamond plan from planning-agent: design -> {backend,frontend} ->
 *  integrate -> ship, with complete(task) actions gated on ready(task). */
function planSpace(): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const space = store.createSpace({ title: 'plan' })
  const tasks = ['design', 'backend', 'frontend', 'integrate', 'ship']
  const ops: WorkingMemoryOperation[] = [
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
  ]
  applyWorkingMemoryOperations(store, space.id, ops)
  return { store, spaceId: space.id }
}

describe('validatePlan: dry-run a whole action sequence on a clone', () => {
  it('a correct topological order validates ok and reaches the goal', () => {
    const { store, spaceId } = planSpace()
    const v = validatePlan(store, spaceId, [
      'complete_design', 'complete_backend', 'complete_frontend', 'complete_integrate', 'complete_ship',
    ])
    assert.equal(v.ok, true, JSON.stringify(v))
    assert.equal(v.firstFailureIndex, undefined)
    assert.equal(v.steps.every((s) => s.applicable), true)
    assert.deepEqual(v.satisfiedGoalIds, ['G1'])
    assert.deepEqual(v.unmetGoalIds, [])
  })

  it('catches an out-of-order step: ship before its deps fails at index 1, naming the gap', () => {
    const { store, spaceId } = planSpace()
    const v = validatePlan(store, spaceId, ['complete_design', 'complete_ship'])
    assert.equal(v.ok, false)
    assert.equal(v.firstFailureIndex, 1)
    assert.equal(v.steps[0]!.applicable, true, 'design (no deps) is fine')
    assert.equal(v.steps[1]!.applicable, false, 'ship cannot run before integrate')
    // The blocked step names ready(ship) as the unmet precondition.
    assert.ok(
      v.steps[1]!.failedPrecondition?.predicate === 'ready' ||
        v.steps[1]!.unsatisfied.some((a) => a.predicate === 'ready'),
      `expected ready(...) as the unmet precondition, got ${JSON.stringify(v.steps[1])}`,
    )
    // It stopped at the failure - no step after index 1.
    assert.equal(v.steps.length, 2)
  })

  it('an applied-but-incomplete plan is not ok: goal unmet at the end', () => {
    const { store, spaceId } = planSpace()
    // Valid prefix that never ships - all steps apply, but the goal is unmet.
    const v = validatePlan(store, spaceId, ['complete_design', 'complete_backend'])
    assert.equal(v.steps.every((s) => s.applicable), true)
    assert.equal(v.firstFailureIndex, undefined)
    assert.equal(v.ok, false, 'goal ship is not reached by this prefix')
    assert.deepEqual(v.unmetGoalIds, ['G1'])
  })

  it('an unknown / non-action id is reported as an error, stopping the run', () => {
    const { store, spaceId } = planSpace()
    const v = validatePlan(store, spaceId, ['complete_design', 'complete_nope'])
    assert.equal(v.ok, false)
    assert.equal(v.firstFailureIndex, 1)
    assert.ok(v.steps[1]!.error, 'a missing action must carry an error')
    assert.equal(v.steps.length, 2)
  })

  it('validation does NOT mutate the real board (clone is discarded)', () => {
    const { store, spaceId } = planSpace()
    const before = store.listNodes(spaceId).map((n) => n.id).sort()
    const doneBefore = getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'done').length
    validatePlan(store, spaceId, ['complete_design', 'complete_backend', 'complete_frontend', 'complete_integrate', 'complete_ship'])
    const after = store.listNodes(spaceId).map((n) => n.id).sort()
    const doneAfter = getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'done').length
    assert.deepEqual(after, before, 'the real board must be byte-identical after validation')
    assert.equal(doneBefore, 0)
    assert.equal(doneAfter, 0, 'no task should be marked done on the real board')
  })

  it('an empty plan with no goals is trivially ok', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'empty' })
    const v = validatePlan(store, space.id, [])
    assert.equal(v.ok, true)
    assert.deepEqual(v.steps, [])
  })

  it('shortest prefix: a minimal correct plan needs all its steps (no redundancy)', () => {
    const { store, spaceId } = planSpace()
    const v = validatePlan(store, spaceId, [
      'complete_design', 'complete_backend', 'complete_frontend', 'complete_integrate', 'complete_ship',
    ])
    // The goal ship(done) is only reached by the LAST step, so the whole plan is the prefix.
    assert.equal(v.shortestPrefixLength, 5)
    assert.deepEqual(v.redundantStepIndices, [])
  })

  it('shortest prefix: trailing steps after the goal is reached are flagged redundant', () => {
    // A repeatable action (precondition is a persistent, non-consumed fact) so the
    // extra steps STAY applicable yet add nothing once the goal is reached.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'mine' })
    const ops: WorkingMemoryOperation[] = [
      { op: 'declare_goal', id: 'G', label: 'have gold', desired: [{ predicate: 'have', args: { item: 'gold' } }] },
      { op: 'assert_fact', id: 'ROCK', predicate: 'rock', args: { at: 'shaft' } },
      {
        op: 'define_action', id: 'mine', action: 'mine', label: 'mine gold',
        preconditions: [{ predicate: 'rock', args: { at: 'shaft' } }],
        effects: [{ predicate: 'have', args: { item: 'gold' } }],
      },
    ]
    applyWorkingMemoryOperations(store, space.id, ops)
    const v = validatePlan(store, space.id, ['mine', 'mine', 'mine'])
    // Goal first reached after 1 step; rock persists so steps 2,3 still apply but are useless.
    assert.equal(v.steps.every((s) => s.applicable), true)
    assert.equal(v.shortestPrefixLength, 1)
    assert.deepEqual(v.redundantStepIndices, [1, 2])
  })

  it('shortest prefix: undefined when the plan never reaches the goal', () => {
    const { store, spaceId } = planSpace()
    const v = validatePlan(store, spaceId, ['complete_design', 'complete_backend'])
    assert.equal(v.ok, false)
    assert.equal(v.shortestPrefixLength, undefined, 'goal never satisfied => no sufficing prefix')
    assert.deepEqual(v.redundantStepIndices, [])
  })

  it('shortest prefix: 0 when the goal already holds before any step runs', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'pre-met' })
    const ops: WorkingMemoryOperation[] = [
      { op: 'declare_goal', id: 'G', label: 'have x', desired: [{ predicate: 'have', args: { item: 'x' } }] },
      { op: 'assert_fact', id: 'F', predicate: 'have', args: { item: 'x' } },
      { op: 'assert_fact', id: 'W', predicate: 'widget', args: { id: 'w' } },
      {
        op: 'define_action', id: 'noop', action: 'noop', label: 'noop',
        preconditions: [{ predicate: 'widget', args: { id: 'w' } }],
        effects: [{ predicate: 'touched', args: { id: 'w' } }],
      },
    ]
    applyWorkingMemoryOperations(store, space.id, ops)
    const v = validatePlan(store, space.id, ['noop'])
    assert.equal(v.shortestPrefixLength, 0, 'goal pre-satisfied => zero-step prefix suffices')
    assert.deepEqual(v.redundantStepIndices, [0], 'the only step is redundant')
  })

  it('shortest prefix: undefined when there are no goals at all', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'goalless' })
    const ops: WorkingMemoryOperation[] = [
      { op: 'assert_fact', id: 'W', predicate: 'widget', args: { id: 'w' } },
      {
        op: 'define_action', id: 'noop', action: 'noop', label: 'noop',
        preconditions: [{ predicate: 'widget', args: { id: 'w' } }],
        effects: [{ predicate: 'touched', args: { id: 'w' } }],
      },
    ]
    applyWorkingMemoryOperations(store, space.id, ops)
    const v = validatePlan(store, space.id, ['noop'])
    assert.equal(v.shortestPrefixLength, undefined)
    assert.deepEqual(v.redundantStepIndices, [])
  })
})
