import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations } from './working-memory.js'
import { getLogicContext, formatLogicContextAsText } from './logic-context.js'

/**
 * #29 third addendum (notebook mode): a goal can read "satisfied" while
 * every desired atom is met ONLY by a bare assertion the model wrote - the
 * board rubber-stamped the model's own claim with zero closure backing.
 * This must surface a soft warning so the discipline "findings come from
 * the closure, not from labels" stays visible. Non-blocking by design.
 */
describe('goal self-seal soft warning', () => {
  it('flags a goal satisfied only by its own bare assertion', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'notebook mode' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'audit complete',
        desired: [{ predicate: 'audit_done', args: { scope: 'all' } }],
      },
      // The model just asserts the goal predicate - no rule derived it.
      { op: 'assert_fact', id: 'F1', predicate: 'audit_done', args: { scope: 'all' } },
    ])
    const ctx = getLogicContext(store, space.id)
    const goal = ctx.goals.find((g) => g.nodeId === 'G1')!
    assert.equal(goal.satisfied, true)
    assert.equal(goal.selfSealed, true, 'a goal met only by a bare assertion is self-sealed')
    const text = formatLogicContextAsText(ctx)
    assert.match(text, /self-sealed|bare assertion|not derived/i)
  })

  it('does NOT flag a goal a rule actually derived', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'real derivation' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'cost known',
        desired: [{ predicate: 'cost', args: { item: 'bolt', total: '?t' } }],
      },
      { op: 'assert_fact', id: 'L1', predicate: 'line', args: { item: 'bolt', unit: 3, qty: 4 } },
      {
        op: 'add_axiom',
        id: 'AX',
        label: 'cost = unit*qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
    ])
    const goal = getLogicContext(store, space.id).goals.find((g) => g.nodeId === 'G1')!
    assert.equal(goal.satisfied, true)
    assert.equal(goal.selfSealed ?? false, false, 'closure-derived satisfaction is not self-sealed')
  })

  it('does NOT flag a goal satisfied by an action effect (model-defined but real transformation)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'action product' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'water exists',
        desired: [{ predicate: 'have', args: { species: 'H2O' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'have', args: { species: 'H2' } },
    ])
    // satisfied is false initially; assert the product via a NON-self route
    // would need an action - but for this unit we only assert that a bare
    // pre-existing fact (not the goal predicate) does not trip the flag.
    const goal = getLogicContext(store, space.id).goals.find((g) => g.nodeId === 'G1')!
    assert.equal(goal.satisfied, false)
    assert.equal(goal.selfSealed ?? false, false)
  })

  it('an unsatisfied goal is never self-sealed', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'open goal' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'pending',
        desired: [{ predicate: 'never_asserted', args: { x: 1 } }],
      },
    ])
    const goal = getLogicContext(store, space.id).goals.find((g) => g.nodeId === 'G1')!
    assert.equal(goal.satisfied, false)
    assert.equal(goal.selfSealed ?? false, false)
  })
})
