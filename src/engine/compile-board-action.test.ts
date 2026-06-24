import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations } from './working-memory.js'
import { simulateActionEffects } from './simulate.js'
import {
  compileAction,
  assertCompilableAction,
  CompileBoardError,
  compileBoard,
  type ActionForCompile,
  type Fact,
} from './compile-board.js'
import { atomKey } from '../kernel/predicate.js'
import type { RuleDefinition } from '../kernel/predicate.js'
import type { PredicateAtom } from '../model/types.js'

/** Canonical-key set of a fact/atom list — the same identity simulate compares. */
function keys(facts: { predicate: string; args?: Record<string, unknown> }[]): string[] {
  return facts
    .map((f) => atomKey({ predicate: f.predicate, args: (f.args ?? {}) as PredicateAtom['args'] }))
    .sort()
}

/**
 * THE HEADLINE: compile an action and prove the compiled `apply` reproduces the
 * board's OWN simulateActionEffects on the same facts — same applicability, same
 * added set, same removed set. This is the action-layer analogue of
 * compileBoardAndCheck's derived-set equality. We run it over a board whose
 * facts are all asserted (no rules), so the engine's EDB == the fact list the
 * compiled transform sees, and the comparison is exact.
 */
function assertFaithful(
  store: MemorySpaceStore,
  spaceId: string,
  actionId: string,
  baseFacts: Fact[],
): void {
  const sim = simulateActionEffects(store, spaceId, actionId)
  const node = store.getNode(spaceId, actionId)
  const compiled = compileAction({
    id: actionId,
    preconditions: node.semantic?.preconditions ?? [],
    effects: node.semantic?.effects ?? [],
  })
  const result = compiled.apply(baseFacts)

  assert.equal(result.applied, sim.applicable, 'applicability must agree with the board')
  assert.deepEqual(keys(result.added), keys(sim.addedAtoms), 'added set must equal the board')
  assert.deepEqual(keys(result.removed), keys(sim.removedAtoms), 'removed set must equal the board')
}

describe('compileAction — faithfulness to deriveActionEffects (the headline)', () => {
  it('a produce+consume action: compiled added/removed equal the board simulation', () => {
    const store = new MemorySpaceStore()
    const { id: spaceId } = store.createSpace({ title: 'ship' })
    applyWorkingMemoryOperations(store, spaceId, [
      { op: 'assert_fact', id: 'F1', predicate: 'order', args: { id: 'o1', status: 'pending' } },
      { op: 'assert_fact', id: 'F2', predicate: 'paid', args: { order: 'o1' } },
      {
        op: 'define_action',
        id: 'A_SHIP',
        label: 'Ship the paid order',
        action: 'ship',
        preconditions: [
          { predicate: 'order', args: { id: '?o', status: 'pending' } },
          { predicate: 'paid', args: { order: '?o' } },
        ],
        effects: [
          { predicate: 'shipped', args: { order: '?o' } },
          { predicate: 'order', args: { id: '?o', status: 'pending' }, negated: true },
        ],
      },
    ])
    const baseFacts: Fact[] = [
      { predicate: 'order', args: { id: 'o1', status: 'pending' } },
      { predicate: 'paid', args: { order: 'o1' } },
    ]
    assertFaithful(store, spaceId, 'A_SHIP', baseFacts)

    // And spell the expectation out, not just the cross-check:
    const compiled = compileAction({
      id: 'A_SHIP',
      preconditions: store.getNode(spaceId, 'A_SHIP').semantic?.preconditions ?? [],
      effects: store.getNode(spaceId, 'A_SHIP').semantic?.effects ?? [],
    })
    const r = compiled.apply(baseFacts)
    assert.equal(r.applied, true)
    assert.deepEqual(keys(r.added), keys([{ predicate: 'shipped', args: { order: 'o1' } }]))
    assert.deepEqual(keys(r.removed), keys([{ predicate: 'order', args: { id: 'o1', status: 'pending' } }]))
    // Resulting set: paid stays, order(pending) consumed, shipped asserted.
    assert.deepEqual(
      keys(r.facts),
      keys([
        { predicate: 'paid', args: { order: 'o1' } },
        { predicate: 'shipped', args: { order: 'o1' } },
      ]),
    )
    assert.equal(r.candidates, 1) // unique binding
  })

  it('a counted-but-uncomputed move action with a variable flowing precondition → effect', () => {
    // carWash DRIVE: bind nothing computed, just route the object through.
    const store = new MemorySpaceStore()
    const { id: spaceId } = store.createSpace({ title: 'drive' })
    applyWorkingMemoryOperations(store, spaceId, [
      { op: 'assert_fact', id: 'F1', predicate: 'at', args: { object: 'car', location: 'home' } },
      { op: 'assert_fact', id: 'F2', predicate: 'at', args: { object: 'user', location: 'home' } },
      {
        op: 'define_action',
        id: 'A_DRIVE',
        label: 'Drive the car to the wash',
        action: 'drive',
        preconditions: [{ predicate: 'at', args: { object: 'car', location: '?from' } }],
        effects: [
          { predicate: 'at', args: { object: 'car', location: 'car_wash' } },
          { predicate: 'at', args: { object: 'car', location: '?from' }, negated: true },
        ],
      },
    ])
    const baseFacts: Fact[] = [
      { predicate: 'at', args: { object: 'car', location: 'home' } },
      { predicate: 'at', args: { object: 'user', location: 'home' } },
    ]
    assertFaithful(store, spaceId, 'A_DRIVE', baseFacts)
  })

  it('preconditions that do not hold: compiled reports not-applied, like the board', () => {
    const store = new MemorySpaceStore()
    const { id: spaceId } = store.createSpace({ title: 'blocked' })
    applyWorkingMemoryOperations(store, spaceId, [
      { op: 'assert_fact', id: 'F1', predicate: 'order', args: { id: 'o1', status: 'shipped' } },
      {
        op: 'define_action',
        id: 'A_SHIP',
        label: 'Ship a pending order',
        action: 'ship',
        preconditions: [{ predicate: 'order', args: { id: '?o', status: 'pending' } }],
        effects: [{ predicate: 'shipped', args: { order: '?o' } }],
      },
    ])
    const baseFacts: Fact[] = [{ predicate: 'order', args: { id: 'o1', status: 'shipped' } }]
    assertFaithful(store, spaceId, 'A_SHIP', baseFacts)

    const compiled = compileAction({
      id: 'A_SHIP',
      preconditions: store.getNode(spaceId, 'A_SHIP').semantic?.preconditions ?? [],
      effects: store.getNode(spaceId, 'A_SHIP').semantic?.effects ?? [],
    })
    const r = compiled.apply(baseFacts)
    assert.equal(r.applied, false)
    assert.deepEqual(r.added, [])
    assert.deepEqual(r.removed, [])
    assert.deepEqual(keys(r.facts), keys(baseFacts)) // unchanged
  })

  it('a naf precondition acts as an anti-join: blocked exactly when the fact is present', () => {
    const make = (): ActionForCompile => ({
      id: 'A_READY',
      preconditions: [
        { predicate: 'order', args: { id: '?o' } },
        { predicate: 'invalid', args: { order: '?o' }, naf: true },
      ],
      effects: [{ predicate: 'ready', args: { order: '?o' } }],
    })
    const compiled = compileAction(make())

    // No invalid fact → the action fires.
    const clean = compiled.apply([{ predicate: 'order', args: { id: 'o1' } }])
    assert.equal(clean.applied, true)
    assert.deepEqual(keys(clean.added), keys([{ predicate: 'ready', args: { order: 'o1' } }]))

    // invalid(order:o1) present → anti-join kills the binding.
    const blocked = compiled.apply([
      { predicate: 'order', args: { id: 'o1' } },
      { predicate: 'invalid', args: { order: 'o1' } },
    ])
    assert.equal(blocked.applied, false)
    assert.deepEqual(blocked.added, [])
  })
})

describe('compileAction — effect sequencing and set hygiene', () => {
  it('asserting an already-present fact adds nothing (idempotent), matching the board', () => {
    const compiled = compileAction({
      id: 'A_TAG',
      preconditions: [{ predicate: 'order', args: { id: '?o' } }],
      effects: [{ predicate: 'order', args: { id: '?o' } }], // re-assert what matched
    })
    const r = compiled.apply([{ predicate: 'order', args: { id: 'o1' } }])
    assert.equal(r.applied, true)
    assert.deepEqual(r.added, []) // already present → not re-added
    assert.deepEqual(keys(r.facts), keys([{ predicate: 'order', args: { id: 'o1' } }]))
  })

  it('add-then-consume of the same atom nets out (sequential effects)', () => {
    const compiled = compileAction({
      id: 'A_FLICKER',
      preconditions: [{ predicate: 'seed', args: { id: '?o' } }],
      effects: [
        { predicate: 'tmp', args: { id: '?o' } }, // assert
        { predicate: 'tmp', args: { id: '?o' }, negated: true }, // then consume it
      ],
    })
    const r = compiled.apply([{ predicate: 'seed', args: { id: 'o1' } }])
    assert.equal(r.applied, true)
    // The net resulting set carries neither a leftover tmp nor a removal of a
    // pre-existing fact: only the seed survives.
    assert.deepEqual(keys(r.facts), keys([{ predicate: 'seed', args: { id: 'o1' } }]))
    assert.equal(
      r.facts.some((f) => f.predicate === 'tmp'),
      false,
    )
  })

  it('consuming a fact that is not present removes nothing', () => {
    const compiled = compileAction({
      id: 'A_CLEAR',
      preconditions: [{ predicate: 'order', args: { id: '?o' } }],
      effects: [{ predicate: 'lock', args: { id: '?o' }, negated: true }],
    })
    const r = compiled.apply([{ predicate: 'order', args: { id: 'o1' } }])
    assert.equal(r.applied, true)
    assert.deepEqual(r.removed, [])
    assert.deepEqual(keys(r.facts), keys([{ predicate: 'order', args: { id: 'o1' } }]))
  })
})

describe('compileAction — composes with the rule pipeline (action then derivation)', () => {
  it('applying a compiled action then the compiled board reproduces the post-action derivation', () => {
    // Board: a rule derives can_receive_service from co-located object+service;
    // an action drives the car to the wash. The engine's simulate predicts the
    // new derived fact. The compiled pair (action ∘ pipeline) must reproduce it.
    const store = new MemorySpaceStore()
    const { id: spaceId } = store.createSpace({ title: 'wash' })
    applyWorkingMemoryOperations(store, spaceId, [
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'co-located object can receive the service',
        when: [
          { predicate: 'service_on', args: { service: '?s', object: '?o', location: '?l' } },
          { predicate: 'at', args: { object: '?o', location: '?l' } },
        ],
        then: [{ predicate: 'can_receive_service', args: { service: '?s', object: '?o' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'service_on', args: { service: 'wash', object: 'car', location: 'car_wash' } },
      { op: 'assert_fact', id: 'F2', predicate: 'at', args: { object: 'car', location: 'home' } },
      {
        op: 'define_action',
        id: 'A_DRIVE',
        label: 'Drive to the wash',
        action: 'drive',
        preconditions: [{ predicate: 'at', args: { object: 'car', location: 'home' } }],
        effects: [
          { predicate: 'at', args: { object: 'car', location: 'car_wash' } },
          { predicate: 'at', args: { object: 'car', location: 'home' }, negated: true },
        ],
      },
    ])

    // The engine's prediction: driving makes can_receive_service appear.
    const sim = simulateActionEffects(store, spaceId, 'A_DRIVE')
    assert.equal(
      sim.newDerivedAtoms.some((a) => a.predicate === 'can_receive_service'),
      true,
    )

    // The compiled pair: apply the action, then run the compiled rule pipeline
    // over the resulting facts.
    const action = compileAction({
      id: 'A_DRIVE',
      preconditions: store.getNode(spaceId, 'A_DRIVE').semantic?.preconditions ?? [],
      effects: store.getNode(spaceId, 'A_DRIVE').semantic?.effects ?? [],
    })
    const baseFacts: Fact[] = [
      { predicate: 'service_on', args: { service: 'wash', object: 'car', location: 'car_wash' } },
      { predicate: 'at', args: { object: 'car', location: 'home' } },
    ]
    const afterAction = action.apply(baseFacts)
    assert.equal(afterAction.applied, true)

    const rules: RuleDefinition[] = [
      {
        id: 'AX1',
        when: [
          { predicate: 'service_on', args: { service: '?s', object: '?o', location: '?l' } },
          { predicate: 'at', args: { object: '?o', location: '?l' } },
        ],
        then: [{ predicate: 'can_receive_service', args: { service: '?s', object: '?o' } }],
      },
    ]
    const derived = compileBoard(rules).pipeline(afterAction.facts)
    assert.equal(
      derived.some(
        (f) => f.predicate === 'can_receive_service' && f.args.service === 'wash' && f.args.object === 'car',
      ),
      true,
      'the compiled action ∘ pipeline derives can_receive_service, matching the board',
    )
  })
})

describe('assertCompilableAction — honest fragment boundaries (fail-visible)', () => {
  it('rejects an arithmetic/producing built-in in a precondition (out of fragment)', () => {
    assert.throws(
      () =>
        assertCompilableAction({
          id: 'A_SUM',
          preconditions: [
            { predicate: 'have', args: { a: '?a', b: '?b' } },
            { predicate: 'add', args: { left: '?a', right: '?b', result: '?c' } },
          ],
          effects: [{ predicate: 'total', args: { value: '?c' } }],
        }),
      (e: unknown) => e instanceof CompileBoardError && /built-in "add"/.test((e as Error).message),
    )
  })

  it('rejects a strong-negated precondition literal', () => {
    assert.throws(
      () =>
        assertCompilableAction({
          id: 'A_NEG',
          preconditions: [{ predicate: 'blocked', args: { id: '?o' }, negated: true }],
          effects: [],
        }),
      (e: unknown) => e instanceof CompileBoardError && /strong negation/.test((e as Error).message),
    )
  })

  it('rejects an unbound effect variable (delegates to the kernel action-safety validator)', () => {
    assert.throws(
      () =>
        assertCompilableAction({
          id: 'A_UNBOUND',
          preconditions: [{ predicate: 'p', args: { x: '?x' } }],
          effects: [{ predicate: 'q', args: { y: '?y' } }], // ?y never bound
        }),
      /not bound/,
    )
  })

  it('rejects a built-in predicate used as an effect (delegates to action-safety)', () => {
    assert.throws(
      () =>
        assertCompilableAction({
          id: 'A_BUILTIN_EFFECT',
          preconditions: [{ predicate: 'p', args: { x: '?x' } }],
          effects: [{ predicate: 'eq', args: { left: '?x', right: 1 } }],
        }),
      /built-in|reserved/,
    )
  })

  it('compileAction surfaces the same boundary (assert happens at compile time)', () => {
    assert.throws(
      () =>
        compileAction({
          id: 'A_SUM',
          preconditions: [
            { predicate: 'have', args: { a: '?a', b: '?b' } },
            { predicate: 'add', args: { left: '?a', right: '?b', result: '?c' } },
          ],
          effects: [{ predicate: 'total', args: { value: '?c' } }],
        }),
      CompileBoardError,
    )
  })
})

describe('compileAction — the readable source artifact', () => {
  it('renders a single-shot transformation function naming the action', () => {
    const compiled = compileAction({
      id: 'A_SHIP',
      preconditions: [
        { predicate: 'order', args: { id: '?o', status: 'pending' } },
        { predicate: 'paid', args: { order: '?o' } },
      ],
      effects: [
        { predicate: 'shipped', args: { order: '?o' } },
        { predicate: 'order', args: { id: '?o', status: 'pending' }, negated: true },
      ],
    })
    assert.match(compiled.source, /function applyAction_A_SHIP\(facts\)/)
    assert.match(compiled.source, /single-shot guarded transformation/)
    assert.match(compiled.source, /added\.push\(\{ predicate: 'shipped'/)
    assert.match(compiled.source, /removed\.push\(\{ predicate: 'order'/)
    assert.match(compiled.source, /applied: true/)
    assert.match(compiled.source, /preconditions unsatisfied/)
  })

  it('a ground (propositional) action renders without an unsatisfiable branch and always applies', () => {
    const compiled = compileAction({
      id: 'A_INIT',
      preconditions: [],
      effects: [{ predicate: 'started', args: {} }],
    })
    assert.doesNotMatch(compiled.source, /preconditions unsatisfied/)
    const r = compiled.apply([])
    assert.equal(r.applied, true)
    assert.deepEqual(keys(r.added), keys([{ predicate: 'started', args: {} }]))
  })
})
