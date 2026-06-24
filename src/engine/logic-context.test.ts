import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations } from './working-memory.js'
import { getLogicContext, formatLogicContextAsText, nextSteps } from './logic-context.js'

/**
 * P2 "next steps" rendering — pure re-presentation of already-derived board
 * data. No new inference: only open goals' hints/actionHints and the standing
 * critique are aggregated. Three non-vacuous cases:
 *   (a) open goal with a producible-via-action hint → step names the action
 *   (b) board with a standing critique item → step includes the critique message
 *   (c) all goals satisfied, healthy board → NO next steps
 */

describe('nextSteps()', () => {
  it('(a) an open goal with an applicable producing action shows a step naming that action', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'action hint' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'h2', predicate: 'have', args: { species: 'H2' } },
      {
        op: 'define_action',
        id: 'make',
        label: 'make water',
        action: 'combust',
        preconditions: [{ predicate: 'have', args: { species: 'H2' } }],
        effects: [
          { predicate: 'have', args: { species: 'H2' }, negated: true },
          { predicate: 'have', args: { species: 'H2O' } },
        ],
      },
      {
        op: 'declare_goal',
        id: 'g1',
        label: 'obtain water',
        desired: [{ predicate: 'have', args: { species: 'H2O' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)

    // Goal is open and action hint is already on the goal.
    const goal = ctx.goals.find((g) => g.nodeId === 'g1')!
    assert.equal(goal.satisfied, false)
    assert.equal(goal.actionHints[0]?.actionNodeId, 'make', 'actionHints already carry the hint')
    assert.equal(goal.actionHints[0]?.applicable, true)

    const steps = nextSteps(ctx)
    // Must mention the action node id.
    const actionStep = steps.find((s) => s.includes('make'))
    assert.ok(actionStep !== undefined, 'a step naming action "make" must appear')
    assert.match(actionStep, /producible via action make/)
    assert.match(actionStep, /preconditions hold/)

    // Also visible in the text rendering.
    const text = formatLogicContextAsText(ctx)
    assert.match(text, /next steps:/)
    assert.match(text, /producible via action make/)
  })

  it('(a-blocked) an open goal with a BLOCKED action shows the blocking guard in the step', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'blocked action hint' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'h2', predicate: 'amount', args: { species: 'H2', mol: 1 } },
      {
        op: 'define_action',
        id: 'burn1',
        label: 'consume 2 mol H2',
        action: 'combust_once',
        preconditions: [
          { predicate: 'amount', args: { species: 'H2', mol: '?h' } },
          { predicate: 'gte', args: { left: '?h', right: 2 } },
        ],
        effects: [{ predicate: 'produced', args: { species: 'H2O' } }],
      },
      {
        op: 'declare_goal',
        id: 'g1',
        label: 'react',
        desired: [{ predicate: 'produced', args: { species: 'H2O' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const steps = nextSteps(ctx)
    const actionStep = steps.find((s) => s.includes('burn1'))
    assert.ok(actionStep !== undefined, 'blocked action step must appear')
    assert.match(actionStep, /blocked on/)
  })

  it('(a-rule) an open goal with an abduction hint (rule path) shows the rule in the step', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'rule hint' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax1',
        label: 'wet means water',
        when: [{ predicate: 'wet', args: {} }],
        then: [{ predicate: 'have', args: { species: 'H2O' } }],
      },
      {
        op: 'declare_goal',
        id: 'g1',
        label: 'obtain water',
        desired: [{ predicate: 'have', args: { species: 'H2O' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const goal = ctx.goals.find((g) => g.nodeId === 'g1')!
    assert.equal(goal.satisfied, false)
    assert.ok(goal.hints.length > 0, 'abduction hints already on the goal')

    const steps = nextSteps(ctx)
    const ruleStep = steps.find((s) => s.includes('ax1'))
    assert.ok(ruleStep !== undefined, 'a step for rule ax1 must appear')
    assert.match(ruleStep, /needs via ax1/)
    assert.match(ruleStep, /wet\(\)/)
  })

  it('(b) a board with a standing critique item lists it under next steps', () => {
    // Trigger a self_sealed_goal critique: goal satisfied only by bare assertion.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'critique present' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'audit complete',
        desired: [{ predicate: 'audit_done', args: { scope: 'all' } }],
      },
      // Bare assertion — no rule derives it, so the goal is self-sealed.
      { op: 'assert_fact', id: 'F1', predicate: 'audit_done', args: { scope: 'all' } },
    ])
    const ctx = getLogicContext(store, space.id)

    // Confirm the critique is already there (pure rendering check).
    assert.ok(ctx.critique.length > 0, 'board has a standing critique item')
    const critiqueItem = ctx.critique.find((c) => c.kind === 'self_sealed_goal')
    assert.ok(critiqueItem !== undefined, 'self_sealed_goal critique must be present')

    const steps = nextSteps(ctx)
    const critiqueStep = steps.find((s) => s.includes('[critique self_sealed_goal]'))
    assert.ok(critiqueStep !== undefined, 'critique item must appear in next steps')
    // The step carries the pre-existing critique message verbatim.
    assert.ok(
      critiqueStep.includes(critiqueItem.message),
      'step must include the existing critique message',
    )

    // Visible in the text rendering.
    const text = formatLogicContextAsText(ctx)
    assert.match(text, /next steps:/)
    assert.match(text, /critique self_sealed_goal/)
  })

  it('(c) an all-satisfied healthy board shows NO next steps', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'all done' })
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
    const ctx = getLogicContext(store, space.id)

    // Confirm the goal is satisfied by the closure (not self-sealed).
    const goal = ctx.goals.find((g) => g.nodeId === 'G1')!
    assert.equal(goal.satisfied, true)
    assert.equal(goal.selfSealed ?? false, false)
    assert.equal(ctx.critique.length, 0, 'healthy board has no critique items')

    const steps = nextSteps(ctx)
    assert.deepEqual(steps, [], 'no next steps on a healthy, all-satisfied board')

    // Text rendering must NOT contain a next steps section.
    const text = formatLogicContextAsText(ctx)
    assert.doesNotMatch(text, /next steps:/)
  })

  it('(d) a satisfied goal nudges record_result first, then DONE once a result is recorded (no re-record loop)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'finish nudge' })
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

    // BEFORE a result is recorded: the satisfied goal points at record_result, not done.
    const before = formatLogicContextAsText(getLogicContext(store, space.id))
    assert.match(before, /call record_result/)
    assert.doesNotMatch(before, /call done to finish/)

    // Record a backed result citing the derived cost fact.
    const costId = getLogicContext(store, space.id).facts.find(
      (f) => f.derived && f.atom.predicate === 'cost',
    )!.nodeId
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'record_result', id: 'R1', label: 'cost computed', summary: 'cost(bolt)=12', evidenceRefs: [costId] },
    ])

    // AFTER: the nudge flips to DONE and stops saying record_result — the arith re-record loop
    // (the model re-issued the same record_result id to the turn limit) is closed.
    const after = formatLogicContextAsText(getLogicContext(store, space.id))
    assert.match(after, /call done to finish/)
    assert.doesNotMatch(after, /call record_result/)
  })
})
