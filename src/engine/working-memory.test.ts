import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import type { PredicateAtom } from '../model/types.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, AttestedPredicateError, GoalpostMovingError, type ApplyOptions, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'

type DerivationContract = NonNullable<ApplyOptions['attestedDerivations']>

describe('machine-attested predicate guard', () => {
  const attested = ['test_result', 'edited']
  const assertResult = (predicate: string): WorkingMemoryOperation =>
    ({ op: 'assert_fact', id: 'x', predicate, args: { test: 't1', status: 'pass' } } as WorkingMemoryOperation)

  it('rejects a MODEL-sourced batch that asserts an attested predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested' })
    assert.throws(
      () => applyWorkingMemoryOperations(store, space.id, [assertResult('test_result')], { source: 'model', attestedPredicates: attested }),
      AttestedPredicateError,
    )
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'test_result').length, 0)
  })

  it('allows a SYSTEM/harness batch to write the same attested predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-sys' })
    applyWorkingMemoryOperations(store, space.id, [assertResult('test_result')], { source: 'system', attestedPredicates: attested })
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'test_result').length, 1)
  })

  it('allows a MODEL batch to assert a NON-attested predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-other' })
    applyWorkingMemoryOperations(store, space.id, [assertResult('observation')], { source: 'model', attestedPredicates: attested })
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'observation').length, 1)
  })

  it('no attested set means no restriction (backward compatible)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-none' })
    applyWorkingMemoryOperations(store, space.id, [assertResult('test_result')], { source: 'model' })
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'test_result').length, 1)
  })

  it('rejects a MODEL rule (add_axiom) whose HEAD produces an attested predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-rule' })
    // The launder route: derive test_result(pass) from a vacuous rule.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store, space.id,
          [
            { op: 'assert_fact', id: 'seed', predicate: 'wishful', args: { test: 't1' } },
            {
              op: 'add_axiom', id: 'AX_FAKE', label: 'launder a green',
              when: [{ predicate: 'wishful', args: { test: '?t' } }],
              then: [{ predicate: 'test_result', args: { test: '?t', status: 'pass' } }],
            },
          ] as WorkingMemoryOperation[],
          { source: 'model', attestedPredicates: attested },
        ),
      AttestedPredicateError,
    )
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'test_result').length, 0)
  })

  it('rejects a MODEL action (define_action) whose EFFECT produces an attested predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-action' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store, space.id,
          [
            {
              op: 'define_action', id: 'fake_pass', action: 'fake_pass', label: 'declare green via action',
              preconditions: [],
              effects: [{ predicate: 'test_result', args: { test: 't1', status: 'pass' } }],
            },
          ] as WorkingMemoryOperation[],
          { source: 'model', attestedPredicates: attested },
        ),
      AttestedPredicateError,
    )
  })

  it('attestedDerivations: a MODEL rule producing finding(kind=fixed) must read the machine evidence', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'derivations' })
    const contract: DerivationContract = [{
      head: { predicate: 'finding', args: { kind: 'fixed' } },
      requires: [{ predicate: 'edited', args: {} }, { predicate: 'test_result', args: { status: 'pass' } }],
    }]
    // Weak rule: derives fixed WITHOUT test_result(pass) -> rejected.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store, space.id,
          [{
            op: 'add_axiom', id: 'WEAK', label: 'launder fixed',
            when: [{ predicate: 'issue', args: { id: '?i' } }],
            then: [{ predicate: 'finding', args: { kind: 'fixed', issue: '?i' } }],
          }] as WorkingMemoryOperation[],
          { source: 'model', attestedDerivations: contract },
        ),
      AttestedPredicateError,
    )
    // Strong rule: reads edited + test_result(status=pass) -> allowed.
    applyWorkingMemoryOperations(
      store, space.id,
      [{
        op: 'add_axiom', id: 'STRONG', label: 'fixed from evidence',
        when: [
          { predicate: 'issue', args: { id: '?i' } },
          { predicate: 'edited', args: { issue: '?i' } },
          { predicate: 'test_result', args: { test: '?t', status: 'pass' } },
        ],
        then: [{ predicate: 'finding', args: { kind: 'fixed', issue: '?i' } }],
      }] as WorkingMemoryOperation[],
      { source: 'model', attestedDerivations: contract },
    )
    assert.equal(getLogicContext(store, space.id).axioms.filter((a) => a.nodeId === 'STRONG').length, 1)
  })

  it('attestedDerivations: evidence must be CO-BOUND to the head (no laundering a different issue)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'cobind' })
    const contract: DerivationContract = [{
      head: { predicate: 'finding', args: { kind: 'fixed' } },
      requires: [{ predicate: 'edited', args: {} }, { predicate: 'test_result', args: { status: 'pass' } }],
    }]
    // edited/test_result PRESENT but bound to a DIFFERENT issue (?j) than the head (?i) -> rejected.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store, space.id,
          [{
            op: 'add_axiom', id: 'LAUNDER', label: 'evidence about another issue',
            when: [
              { predicate: 'issue', args: { id: '?i' } },
              { predicate: 'edited', args: { issue: '?j' } },
              { predicate: 'test_result', args: { test: '?t', status: 'pass' } },
            ],
            then: [{ predicate: 'finding', args: { kind: 'fixed', issue: '?i' } }],
          }] as WorkingMemoryOperation[],
          { source: 'model', attestedDerivations: contract },
        ),
      AttestedPredicateError,
    )
  })

  it('attestedDerivations: a rule producing an UNGUARDED finding kind is unaffected', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'derivations-other' })
    const contract: DerivationContract = [{ head: { predicate: 'finding', args: { kind: 'fixed' } }, requires: [{ predicate: 'test_result', args: { status: 'pass' } }] }]
    // head kind='style' does not match the guarded kind='fixed' -> allowed without evidence.
    applyWorkingMemoryOperations(
      store, space.id,
      [{ op: 'add_axiom', id: 'STYLE', label: 'style finding', when: [{ predicate: 'lint', args: { file: '?f' } }], then: [{ predicate: 'finding', args: { kind: 'style', file: '?f' } }] }] as WorkingMemoryOperation[],
      { source: 'model', attestedDerivations: contract },
    )
    assert.equal(getLogicContext(store, space.id).axioms.filter((a) => a.nodeId === 'STYLE').length, 1)
  })

  it('rejects a MODEL derive_aggregate whose INTO produces an attested predicate (borrowed from Codex)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-agg' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(
          store, space.id,
          [
            { op: 'assert_fact', id: 'm1', predicate: 'metric', args: { test: 't1', n: 3 } },
            { op: 'derive_aggregate', id: 'AGG', kind: 'count', source: { predicate: 'metric' }, into: { predicate: 'test_result', valueArg: 'status' } },
          ] as WorkingMemoryOperation[],
          { source: 'model', attestedPredicates: attested },
        ),
      AttestedPredicateError,
    )
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'test_result').length, 0)
  })

  it('a MODEL rule may READ an attested predicate in its body (only producing is blocked)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'attested-read' })
    // Harness writes the attested fact; the model reads it in a rule body - allowed.
    applyWorkingMemoryOperations(store, space.id, [assertResult('test_result')], { source: 'system', attestedPredicates: attested })
    applyWorkingMemoryOperations(
      store, space.id,
      [
        {
          op: 'add_axiom', id: 'AX_OK', label: 'derive verified from the runner fact',
          when: [{ predicate: 'test_result', args: { test: '?t', status: 'pass' } }],
          then: [{ predicate: 'verified', args: { test: '?t' } }],
        },
      ] as WorkingMemoryOperation[],
      { source: 'model', attestedPredicates: attested },
    )
    assert.equal(getLogicContext(store, space.id).facts.filter((f) => f.atom.predicate === 'verified').length, 1)
  })
})

describe('applyWorkingMemoryOperations', () => {
  it('applies a batch of operations and returns the updated working memory', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Working memory batch' })

    const result = applyWorkingMemoryOperations(
      store,
      space.id,
      [
        {
          op: 'declare_goal',
          id: 'G1',
          label: 'Car must be at wash shop',
          desired: [
            { predicate: 'must_be_at', args: { object: 'car', location: 'car_wash' } },
          ],
        },
        {
          op: 'add_axiom',
          id: 'AX1',
          label: 'Service requires object at location',
          when: [
            {
              predicate: 'service_on',
              args: { service: '?service', object: '?object', location: '?location' },
            },
          ],
          then: [
            {
              predicate: 'must_be_at',
              args: { object: '?object', location: '?location' },
            },
          ],
        },
        {
          op: 'assert_fact',
          id: 'F1',
          predicate: 'service_on',
          args: { service: 'wash', object: 'car', location: 'car_wash' },
        },
      ],
      { format: 'text' },
    )

    assert.equal(result.operationResults.length, 3)
    assert.equal(
      result.workingMemory.facts.some(
        (fact) => fact.atom.predicate === 'must_be_at',
      ),
      true,
    )
    assert.match(result.workingMemoryText ?? '', /must_be_at/)
  })

  it('retracts and replaces facts inside a batch before rule closure', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Working memory revision' })

    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'A implies B',
        when: [{ predicate: 'a', args: { item: '?x' } }],
        then: [{ predicate: 'b', args: { item: '?x' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'a', args: { item: 'wrong' } },
    ])

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'revise_fact',
        nodeId: 'F1',
        id: 'F2',
        predicate: 'a',
        args: { item: 'right' },
      },
    ])

    assert.equal(
      result.workingMemory.facts.some(
        (fact) => fact.atom.predicate === 'b' && fact.atom.args?.item === 'wrong',
      ),
      false,
    )
    assert.equal(
      result.workingMemory.facts.some(
        (fact) => fact.atom.predicate === 'b' && fact.atom.args?.item === 'right',
      ),
      true,
    )
  })

  it('physically removes retracted nodes and dependents, then re-derives what is still supported', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Working memory rebalance' })

    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX_A',
        label: 'A implies B',
        when: [{ predicate: 'a', args: { item: '?x' } }],
        then: [{ predicate: 'b', args: { item: '?x' } }],
      },
      { op: 'assert_fact', id: 'F_A', predicate: 'a', args: { item: 'one' } },
      {
        op: 'add_axiom',
        id: 'AX_C',
        label: 'C implies B',
        when: [{ predicate: 'c', args: { item: '?x' } }],
        then: [{ predicate: 'b', args: { item: '?x' } }],
      },
      { op: 'assert_fact', id: 'F_C', predicate: 'c', args: { item: 'one' } },
    ])

    // b(one) was derived once (from a). Retracting a removes the node and the
    // derived b(one) physically, then the closure re-derives b(one) from c.
    const afterRetractA = applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'F_A', reason: 'a was wrong' },
    ])
    assert.equal(afterRetractA.operationResults[0]?.retractedNodeIds.includes('F_A'), true)
    assert.equal(store.listNodes(space.id).some((node) => node.id === 'F_A'), false)
    assert.equal(
      afterRetractA.workingMemory.facts.some(
        (fact) => fact.atom.predicate === 'b' && fact.atom.args?.item === 'one',
      ),
      true,
    )

    // Retracting c as well removes the last support; b(one) disappears for good.
    const afterRetractC = applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'F_C', reason: 'c was wrong too' },
    ])
    assert.equal(
      afterRetractC.workingMemory.facts.some((fact) => fact.atom.predicate === 'b'),
      false,
    )
    assert.equal(
      store.listNodes(space.id).every((node) => node.type === 'axiom'),
      true,
    )
  })

  it('judges hypotheses automatically after each closure', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Hypothesis lifecycle' })

    // Open while nothing supports it.
    const opened = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_hypothesis',
        id: 'H1',
        predicate: 'finding',
        args: { kind: 'possible_null_deref', function: 'renderUserName' },
      },
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'Nullable deref is a finding',
        when: [{ predicate: 'nullable', args: { function: '?f' } }],
        then: [{ predicate: 'finding', args: { kind: 'possible_null_deref', function: '?f' } }],
      },
    ])
    assert.deepEqual(
      opened.workingMemory.hypotheses.map((h) => h.status),
      ['open'],
    )

    // Supported once the closure derives the hypothesized atom.
    const supported = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'OBS1', predicate: 'nullable', args: { function: 'renderUserName' } },
    ])
    assert.deepEqual(
      supported.workingMemory.hypotheses.map((h) => h.status),
      ['supported'],
    )

    // Refuted when the negated atom enters the working memory.
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'OBS1', reason: 'observation was wrong' },
      {
        op: 'assert_fact',
        id: 'OBS2',
        predicate: 'finding',
        args: { kind: 'possible_null_deref', function: 'renderUserName' },
        negated: true,
      },
    ])
    const refuted = applyWorkingMemoryOperations(store, space.id, [])
    assert.deepEqual(
      refuted.workingMemory.hypotheses.map((h) => h.status),
      ['refuted'],
    )
  })

  it('warns when an atom deviates from the registered predicate signature', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Vocabulary drift' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F1', predicate: 'at', args: { object: 'car', location: 'home' } },
      { op: 'assert_fact', id: 'F2', predicate: 'at', args: { object: 'user', place: 'home' } },
    ])

    assert.equal(result.warnings.length, 1)
    assert.match(result.warnings[0] ?? '', /signature mismatch/)
    assert.match(result.warnings[0] ?? '', /at\(location, object\)/)
    assert.equal(result.workingMemory.vocabulary.includes('at(location, object)'), true)
  })

  it('records results and conflicts, and rejects unknown operations clearly', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Conclusions' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F1', predicate: 'observed', args: { item: 'x' } },
      {
        op: 'record_result',
        id: 'R1',
        label: 'Conclusion',
        summary: 'Based on F1.',
        evidenceRefs: ['F1'],
      },
      { op: 'record_conflict', id: 'C1', label: 'Tension', summary: 'F1 is surprising.' },
    ])
    assert.equal(result.workingMemory.results.some((note) => note.nodeId === 'R1'), true)
    assert.equal(result.workingMemory.conflicts.some((note) => note.nodeId === 'C1'), true)

    // Retracting the evidence removes the conclusion that rested on it.
    const afterRetract = applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'F1', reason: 'observation was wrong' },
    ])
    assert.equal(afterRetract.workingMemory.results.length, 0)

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'bogus_op' } as unknown as Parameters<typeof applyWorkingMemoryOperations>[2][number],
        ]),
      /unknown op "bogus_op"; valid ops/,
    )
  })

  it('reports predicate conflicts as working-memory state', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Working memory conflict' })

    const result = applyWorkingMemoryOperations(
      store,
      space.id,
      [
        {
          op: 'assert_fact',
          id: 'F_POS',
          predicate: 'at',
          args: { object: 'car', location: 'home' },
        },
        {
          op: 'assert_fact',
          id: 'F_NEG',
          predicate: 'at',
          args: { location: 'home', object: 'car' },
          negated: true,
        },
      ],
      { format: 'text' },
    )

    assert.deepEqual(result.workingMemory.predicateConflicts, [
      {
        atom: { predicate: 'at', args: { object: 'car', location: 'home' } },
        positiveFactId: 'F_POS',
        negativeFactId: 'F_NEG',
      },
    ])
    assert.match(result.workingMemoryText ?? '', /predicate contradiction/)
    assert.match(result.workingMemoryText ?? '', /not at\(location=home, object=car\)/)
  })
})

describe('pattern hypotheses and disputed taint', () => {
  it('supports non-ground hypotheses with instances and never refutes them', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Exists finding' })

    const open = applyWorkingMemoryOperations(store, space.id, [
      { op: 'declare_hypothesis', id: 'H1', predicate: 'finding', args: { kind: '?any' } },
      {
        op: 'assert_fact',
        id: 'F_NEG',
        predicate: 'finding',
        args: { kind: 'npe' },
        negated: true,
      },
    ])
    // A negative instance does not refute an existential pattern.
    assert.equal(open.workingMemory.hypotheses[0]?.status, 'open')

    const supported = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F1', predicate: 'finding', args: { kind: 'race_condition' } },
    ])
    assert.equal(supported.workingMemory.hypotheses[0]?.status, 'supported')
    assert.deepEqual(supported.workingMemory.hypotheses[0]?.instances, [
      { predicate: 'finding', args: { kind: 'race_condition' }, negated: undefined },
    ])

    // Pattern goals: "any finding" satisfies the goal.
    const goal = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'Find anything',
        desired: [{ predicate: 'finding', args: { kind: '?k' } }],
      },
    ])
    assert.equal(goal.workingMemory.goals[0]?.satisfied, true)
  })

  it('marks conclusions resting on contradicted facts as disputed', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Contradiction taint' })

    const result = applyWorkingMemoryOperations(
      store,
      space.id,
      [
        {
          op: 'add_axiom',
          id: 'AX1',
          label: 'A implies B',
          when: [{ predicate: 'a', args: { item: '?x' } }],
          then: [{ predicate: 'b', args: { item: '?x' } }],
        },
        { op: 'assert_fact', id: 'F_POS', predicate: 'a', args: { item: 'one' } },
        { op: 'assert_fact', id: 'F_NEG', predicate: 'a', args: { item: 'one' }, negated: true },
        { op: 'assert_fact', id: 'F_OK', predicate: 'a', args: { item: 'two' } },
        { op: 'record_result', id: 'R1', label: 'Rests on b(one)', summary: 'x', evidenceRefs: ['derived:b|item:"one"'] },
      ],
      { format: 'text' },
    )

    const byId = new Map(result.workingMemory.facts.map((fact) => [fact.nodeId, fact]))
    // Both sides of the contradiction and the derived b(one) are disputed.
    assert.equal(byId.get('F_POS')?.disputed, true)
    assert.equal(byId.get('F_NEG')?.disputed, true)
    assert.equal(byId.get('derived:b|item:"one"')?.disputed, true)
    // The untouched branch is clean.
    assert.equal(byId.get('F_OK')?.disputed, undefined)
    assert.equal(byId.get('derived:b|item:"two"')?.disputed, undefined)
    // The result resting on the disputed derivation is disputed too.
    assert.equal(result.workingMemory.results[0]?.disputed, true)
    assert.match(result.workingMemoryText ?? '', /b\(item=one\) \[derived\] \[disputed\]/)
  })
})

describe('vocabulary with pattern atoms', () => {
  it('does not flag full-signature atoms after an existential pattern registered first', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Provisional signatures' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'Find anything',
        desired: [{ predicate: 'finding', args: { kind: '?k' } }],
      },
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'Rule with the full signature',
        when: [{ predicate: 'observed', args: { file: '?f' } }],
        then: [{ predicate: 'finding', args: { kind: 'issue', file: '?f' } }],
      },
      { op: 'assert_fact', id: 'O1', predicate: 'observed', args: { file: 'a.ts' } },
    ])
    assert.deepEqual(result.warnings, [])

    // Ground facts firm the signature; a later deviating ground fact warns.
    const drift = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O2', predicate: 'observed', args: { path: 'b.ts' } },
    ])
    assert.equal(drift.warnings.length, 1)
  })
})

describe('model-input tolerance', () => {
  it('normalizes the {op_name: {...}} operation shape', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Shape tolerance' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        assert_fact: { id: 'F1', predicate: 'observed', args: { file: 'a.ts' } },
      } as unknown as Parameters<typeof applyWorkingMemoryOperations>[2][number],
    ])
    assert.equal(
      result.workingMemory.facts.some((fact) => fact.atom.predicate === 'observed'),
      true,
    )
  })

  it('teaches the exact op when an operation omits "op" but is shaped like a fact (qwen arith p1)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'missing op (fact)' })
    // The model wrote predicate:"line" + args directly on the operation with NO "op", using predicate
    // as if it were the op type. Old: opaque `unknown op "undefined"`. New: name the fix + infer the op.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { id: 'f1', predicate: 'line', args: { item: 'coupling', unit: 3, qty: 2 } } as unknown as Parameters<
            typeof applyWorkingMemoryOperations
          >[2][number],
        ]),
      /missing the required "op".*predicate:"line".*op:"assert_fact"/s,
    )
    assert.equal(store.listNodes(space.id).length, 0, 'rejected batch leaves the board untouched (atomic)')
  })

  it('infers declare_goal from a "desired" field when "op" is omitted', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'missing op (goal)' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { id: 'g1', desired: [{ predicate: 'cost', args: { item: 'x' } }] } as unknown as Parameters<
            typeof applyWorkingMemoryOperations
          >[2][number],
        ]),
      /missing the required "op".*op:"declare_goal"/s,
    )
  })

  it('a present-but-unknown op still gets the plain "unknown op" message (path unchanged)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'unknown op' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'frobnicate', id: 'X' } as unknown as Parameters<typeof applyWorkingMemoryOperations>[2][number],
        ]),
      /unknown op "frobnicate"; valid ops/,
    )
  })

  it('rejects string atoms and missing predicates with a readable error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Atom validation' })

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'declare_goal',
            id: 'G1',
            label: 'Bad goal',
            desired: ['finding(robustness_issue)'],
          } as unknown as Parameters<typeof applyWorkingMemoryOperations>[2][number],
        ]),
      /invalid atom in declare_goal\.desired.*"predicate"/,
    )
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'declare_goal',
            id: 'G2',
            label: 'Empty goal',
            desired: [],
          },
        ]),
      /non-empty atom array/,
    )
    // Nothing was applied from the rejected batches.
    assert.equal(store.listNodes(space.id).length, 0)
  })
})

describe('derivation gate for record_result', () => {
  it('blocks record_result whose evidenceRefs do not resolve to current board nodes', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: result refs' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F1', predicate: 'observed', args: { item: 'x' } },
    ])
    const nodesBefore = store.listNodes(space.id).length

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'record_result', id: 'R1', label: 'Done', summary: 'x observed', evidenceRefs: ['MISSING'] },
        ]),
      (error: Error) =>
        error.name === 'EvidenceReferenceError' &&
        /record_result blocked/.test(error.message) &&
        /MISSING/.test(error.message),
    )
    assert.equal(store.listNodes(space.id).length, nodesBefore)
    assert.equal(getLogicContext(store, space.id).results.length, 0)
  })

  it('allows record_result to cite a derived node produced by the same batch', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: same-batch derived refs' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'observed implies verified',
        when: [{ predicate: 'observed', args: { item: '?x' } }],
        then: [{ predicate: 'verified', args: { item: '?x' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'observed', args: { item: 'x' } },
      {
        op: 'record_result',
        id: 'R1',
        label: 'Done',
        summary: 'verified x',
        evidenceRefs: ['AX1', 'F1', 'derived:verified|item:"x"'],
      },
    ])

    assert.equal(result.workingMemory.results.length, 1)
    assert.equal(result.workingMemory.results[0]?.nodeId, 'R1')
  })

  it('blocks record_result while a positive asserted finding is on the board, applying nothing', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: board' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F1', predicate: 'finding', args: { type: 'resource_leak', file: 'A.java' } },
    ])
    const nodesBefore = store.listNodes(space.id).length

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'record_result', id: 'R1', label: 'Done', summary: 'audit complete' },
        ]),
      (error: Error) =>
        error.name === 'DerivationGateError' &&
        /record_result blocked/.test(error.message) &&
        /F1/.test(error.message),
    )
    assert.equal(store.listNodes(space.id).length, nodesBefore)
  })

  it('blocks a batch that asserts a finding and records a result together', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: same batch' })

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'F1', predicate: 'finding', args: { type: 'race' } },
          { op: 'record_result', id: 'R1', label: 'Done', summary: 'x' },
        ]),
      /asserted in this same batch.*F1/,
    )
    assert.equal(store.listNodes(space.id).length, 0)
  })

  it('passes once findings are derived, and when retraction happens in the same batch', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: derived ok' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F_BAD', predicate: 'finding', args: { type: 'leak', file: 'A.java' } },
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'no close in finally leaks',
        when: [{ predicate: 'no_close_in_finally', args: { file: '?f' } }],
        then: [{ predicate: 'finding', args: { type: 'leak', file: '?f' } }],
      },
      { op: 'assert_fact', id: 'OBS1', predicate: 'no_close_in_finally', args: { file: 'A.java' } },
    ])

    // Retract the bare claim in the same batch as record_result: gate passes,
    // and the finding survives because the closure stands behind it.
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'F_BAD', reason: 'replaced by observation + rule' },
      { op: 'record_result', id: 'R1', label: 'Done', summary: 'leak in A.java', evidenceRefs: ['OBS1'] },
    ])
    assert.equal(result.workingMemory.results.length, 1)
    assert.equal(
      result.workingMemory.findings.some(
        (finding) => finding.derived && finding.atom.args?.file === 'A.java',
      ),
      true,
    )
  })

  it('does not gate negated findings, finding-free boards, or record_conflict', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Gate: exemptions' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F_NEG', predicate: 'finding', args: { type: 'npe' }, negated: true },
      { op: 'assert_fact', id: 'F_OBS', predicate: 'observed', args: { item: 'x' } },
      { op: 'record_result', id: 'R1', label: 'Nothing found', summary: 'clean' },
    ])
    assert.equal(result.workingMemory.results.length, 1)

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F_CLAIM', predicate: 'finding', args: { type: 'race' } },
      { op: 'record_conflict', id: 'C1', label: 'Tension', summary: 'conflicting evidence' },
    ])
    assert.equal(store.listNodes(space.id).some((node) => node.id === 'C1'), true)
  })
})

describe('placeholder and duplicate warnings', () => {
  it('hard-rejects unambiguous type-name placeholders in goals, applying nothing', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Strict placeholders' })

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'declare_goal',
            id: 'G1',
            label: 'Find leaks',
            desired: [{ predicate: 'finding', args: { type: 'leak', file: 'string', line: 'number' } }],
          },
        ]),
      /file=string, line=number.*\?variable/s,
    )
    assert.equal(store.listNodes(space.id).length, 0)
  })

  it('hard-rejects alternation literals in goal args', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Pipe placeholder' })

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'declare_goal',
            id: 'G1',
            label: 'Verdicts',
            desired: [{ predicate: 'verdict', args: { claim: '1', judgment: 'confirmed|refuted' } }],
          },
        ]),
      /judgment=confirmed\|refuted.*\?variable/s,
    )
  })

  it('warns (not errors) on fuzzy placeholder values', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Soft placeholders' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'declare_hypothesis', id: 'H1', predicate: 'issue', args: { kind: 'any' } },
    ])
    assert.equal(result.warnings.length, 1)
    assert.match(result.warnings[0] ?? '', /kind=any/)
  })

  it('does not warn on real constants or ?variables', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Real values' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'Car at wash',
        desired: [{ predicate: 'at', args: { object: 'car', location: 'car_wash' } }],
      },
      { op: 'declare_hypothesis', id: 'H1', predicate: 'finding', args: { kind: '?any' } },
    ])
    assert.deepEqual(result.warnings, [])
  })

  it('reuses the existing node when an identical fact is re-asserted (idempotent)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Duplicates' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O1', predicate: 'empty_catch', args: { file: 'A.java', line: '5' } },
    ])

    const acrossBatch = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O2', predicate: 'empty_catch', args: { line: '5', file: 'A.java' } },
    ])
    assert.equal(acrossBatch.warnings.length, 1)
    assert.match(acrossBatch.warnings[0] ?? '', /already on the board as O1/)
    // No duplicate node: the operation resolved to the existing one.
    assert.deepEqual(acrossBatch.operationResults[0]?.nodeIds, ['O1'])
    assert.equal(store.listNodes(space.id).some((node) => node.id === 'O2'), false)

    const withinBatch = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O3', predicate: 'empty_catch', args: { file: 'B.java', line: '9' } },
      { op: 'assert_fact', id: 'O4', predicate: 'empty_catch', args: { file: 'B.java', line: '9' } },
    ])
    assert.equal(withinBatch.warnings.length, 1)
    assert.match(withinBatch.warnings[0] ?? '', /already on the board as O3/)
    assert.equal(store.listNodes(space.id).some((node) => node.id === 'O4'), false)

    // Different args or different sign: distinct facts, no warning.
    const distinct = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O5', predicate: 'empty_catch', args: { file: 'A.java', line: '6' } },
      { op: 'assert_fact', id: 'O6', predicate: 'empty_catch', args: { file: 'A.java', line: '5' }, negated: true },
    ])
    assert.equal(distinct.warnings.length, 0)
  })
})

describe('node reference and id collision errors', () => {
  it('rejects retract_node without nodeId with a readable error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Missing nodeId' })

    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'retract_node' } as unknown as Parameters<typeof applyWorkingMemoryOperations>[2][number],
        ]),
      /retract_node requires "nodeId"/,
    )
  })

  it('rejects a duplicate node id before applying anything (batch atomicity)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Duplicate id' })

    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'A implies B',
        when: [{ predicate: 'a', args: { item: '?x' } }],
        then: [{ predicate: 'b', args: { item: '?x' } }],
      },
    ])
    const nodesBefore = store.listNodes(space.id).length

    // The colliding op comes AFTER a valid one: nothing may be applied.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'F_NEW', predicate: 'a', args: { item: 'one' } },
          {
            op: 'add_axiom',
            id: 'AX1',
            label: 'Re-added axiom',
            when: [{ predicate: 'a', args: { item: '?x' } }],
            then: [{ predicate: 'b', args: { item: '?x' } }],
          },
        ]),
      /already exists on the board.*no action needed/,
    )
    assert.equal(store.listNodes(space.id).length, nodesBefore)

    // Same-batch duplicate ids are also caught.
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'F1', predicate: 'a', args: { item: 'one' } },
          { op: 'assert_fact', id: 'F1', predicate: 'a', args: { item: 'two' } },
        ]),
      /already exists/,
    )
  })

  it('reuses the existing rule when an identical axiom is re-added (idempotent)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Rule dedup' })

    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'race rule',
        when: [
          { predicate: 'mutable_static_field', args: { file: '?f', line: '?l' } },
          { predicate: 'unsynchronized_mutable_static', args: { file: '?f', line: '?l' } },
        ],
        then: [{ predicate: 'finding', args: { type: 'race_condition', file: '?f', line: '?l' } }],
      },
    ])

    // Same rule, different id, body literals in a different order: reused.
    const redo = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX_RACE_3',
        label: 'race rule again',
        when: [
          { predicate: 'unsynchronized_mutable_static', args: { file: '?f', line: '?l' } },
          { predicate: 'mutable_static_field', args: { file: '?f', line: '?l' } },
        ],
        then: [{ predicate: 'finding', args: { type: 'race_condition', file: '?f', line: '?l' } }],
      },
    ])
    assert.equal(redo.warnings.length, 1)
    assert.match(redo.warnings[0] ?? '', /identical rule.*AX1/)
    assert.deepEqual(redo.operationResults[0]?.nodeIds, ['AX1'])
    assert.equal(store.listNodes(space.id).some((node) => node.id === 'AX_RACE_3'), false)

    // A genuinely different rule still goes in without warnings.
    const different = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX2',
        label: 'other rule',
        when: [{ predicate: 'empty_catch', args: { file: '?f', line: '?l' } }],
        then: [{ predicate: 'finding', args: { type: 'swallowed_exception', file: '?f', line: '?l' } }],
      },
    ])
    assert.deepEqual(different.warnings, [])
  })

  it('still allows retract-then-reuse of an id in one batch', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Id reuse' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'O1', predicate: 'a', args: { item: 'wrong' } },
    ])
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'retract_node', nodeId: 'O1', reason: 'wrong' },
      { op: 'assert_fact', id: 'O1', predicate: 'a', args: { item: 'right' } },
    ])
    assert.equal(
      result.workingMemory.facts.some((fact) => fact.atom.args?.item === 'right'),
      true,
    )
  })
})

describe('vacuous rule warnings', () => {
  it('warns on a name-echo passthrough rule but still applies it', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Vacuous rule' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax_leak',
        label: 'suspected leak is a leak',
        when: [{ predicate: 'suspected_resource_leak', args: { file: '?f' } }],
        then: [{ predicate: 'finding', args: { type: 'resource_leak', file: '?f' } }],
      },
      { op: 'assert_fact', id: 'S1', predicate: 'suspected_resource_leak', args: { file: 'A.java' } },
    ])

    assert.equal(result.warnings.some((w) => /vacuous/.test(w)), true)
    // The rule still fires — it is a warning, not a hard error.
    assert.equal(
      result.workingMemory.findings.some((f) => f.atom.args?.type === 'resource_leak'),
      true,
    )
  })

  it('does not warn on a genuine observation->finding rule', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'Genuine rule' })

    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax_swallow',
        label: 'empty catch swallows',
        when: [{ predicate: 'empty_catch', args: { file: '?f', line: '?l' } }],
        then: [{ predicate: 'finding', args: { type: 'swallowed_exception', file: '?f', line: '?l' } }],
      },
    ])
    assert.equal(result.warnings.some((w) => /vacuous/.test(w)), false)
  })
})

describe('provenance cannot be forged (external review P0)', () => {
  it('rejects assert_fact carrying a reserved provenance summary', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'forge: summary' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'assert_fact',
            id: 'F1',
            predicate: 'finding',
            args: { kind: 'fake' },
            summary: 'Rule-derived fact: finding(kind=fake)',
          },
        ]),
      /reserved provenance|assigns provenance/i,
    )
    assert.equal(store.listNodes(space.id).length, 0)
  })

  it('rejects assert_fact carrying a derived: node id', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'forge: id' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'assert_fact',
            id: 'derived:finding|kind:"fake"',
            predicate: 'finding',
            args: { kind: 'fake' },
          },
        ]),
      /reserved id prefix|derived:/i,
    )
  })
})

describe('declare_hypothesis validation (self-audit #29 finding)', () => {
  it('rejects declare_hypothesis without a predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'hyp: no predicate' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'declare_hypothesis', id: 'H1', args: { x: 1 } } as never,
        ]),
      /predicate/i,
    )
  })
})

describe('assert_fact validation (external review P2)', () => {
  it('rejects assert_fact without a predicate instead of minting an undefined() fact', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'no predicate' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'F1', args: { x: 1 } } as never,
        ]),
      /predicate/i,
    )
  })
})

describe('unfirable-rule teaching warning (repair round 2026-06-13: sumTo burned 8 turns)', () => {
  it('warns when a rule body references a predicate nothing supplies, naming the builtins', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'silent no-fire' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'assert_fact',
        id: 'f1',
        predicate: 'failing_case',
        args: { input: 1, expected: 1, got: 0 },
      },
      {
        op: 'add_axiom',
        id: 'ax1',
        label: 'off-by-one detector',
        when: [
          { predicate: 'failing_case', args: { input: '?n', expected: '?e', got: '?g' } },
          { predicate: 'sub', args: { left: '?e', right: '?n', result: '?d' } },
          // The real model invented compare(op=eq,...) - NOT a builtin, and
          // nothing on the board asserts or derives it: the rule can never fire.
          { predicate: 'compare', args: { op: 'eq', left: '?g', right: '?d' } },
        ],
        then: [{ predicate: 'diagnosis', args: { kind: 'off_by_one' } }],
      },
    ])
    const warning = result.warnings.find((w) => w.includes('compare'))
    assert.ok(warning, `expected an unfirable-rule warning, got: ${JSON.stringify(result.warnings)}`)
    assert.match(warning!, /never fire|cannot fire/i)
    assert.match(warning!, /eq/, 'must name the comparison builtins so the fix is copy-pasteable')
    assert.match(warning!, /assert|derive/i, 'must teach the two ways to supply the predicate')
  })

  it('stays silent when the batch itself supplies the body predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'supplied in batch' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'o1', predicate: 'observation', args: { line: 7 } },
      {
        op: 'add_axiom',
        id: 'ax1',
        label: 'finding from observation',
        when: [{ predicate: 'observation', args: { line: '?l' } }],
        then: [{ predicate: 'hit', args: { line: '?l' } }],
      },
    ])
    assert.equal(
      result.warnings.filter((w) => /never fire|cannot fire/i.test(w)).length,
      0,
      `no unfirable warning expected, got: ${JSON.stringify(result.warnings)}`,
    )
  })

  it('stays silent when another rule head derives the body predicate', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'derived by sibling rule' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax_lower',
        label: 'base signal',
        when: [{ predicate: 'raw', args: { v: '?v' } }],
        then: [{ predicate: 'signal', args: { v: '?v' } }],
      },
      {
        op: 'add_axiom',
        id: 'ax_upper',
        label: 'conclusion from signal',
        when: [{ predicate: 'signal', args: { v: '?v' } }],
        then: [{ predicate: 'conclusion', args: { v: '?v' } }],
      },
    ])
    // signal(...) has no facts yet, but ax_lower can derive it - not unfirable
    // (raw(...) itself WILL warn: nothing supplies it; that one is correct).
    assert.equal(
      result.warnings.filter((w) => w.includes('"signal"')).length,
      0,
      `signal is derivable, got: ${JSON.stringify(result.warnings)}`,
    )
  })

  it('naf literals are exempt - absence is their job', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'naf exempt' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'x1', predicate: 'item', args: { id: 'a' } },
      {
        op: 'add_axiom',
        id: 'ax_gap',
        label: 'unpaired item',
        when: [
          { predicate: 'item', args: { id: '?i' } },
          { predicate: 'paired', args: { id: '?i' }, naf: true },
        ],
        then: [{ predicate: 'gap', args: { id: '?i' } }],
      },
    ])
    assert.equal(
      result.warnings.filter((w) => w.includes('"paired"')).length,
      0,
      `naf literal must not warn, got: ${JSON.stringify(result.warnings)}`,
    )
  })
})

describe('self-recursive arithmetic rules (#32 problem 8: the sum_sofar fold wreck)', () => {
  const accumulatorRule = (id: string, item = '?i'): WorkingMemoryOperation => ({
    op: 'add_axiom' as const,
    id,
    label: 'accumulate',
    when: [
      { predicate: 'cost', args: { item, total: '?t' } },
      { predicate: 'sum_sofar', args: { value: '?s' } },
      { predicate: 'add', args: { left: '?s', right: '?t', result: '?n' } },
    ],
    then: [{ predicate: 'sum_sofar', args: { value: '?n' } }],
  })

  it('warns at add time: unguarded recursion + arithmetic cannot fold on a monotonic board', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'fold warning' })
    // The warning must arrive BEFORE any seed fact exists - by the time the
    // closure explodes, every subsequent apply on the space fails too.
    const result = applyWorkingMemoryOperations(store, space.id, [accumulatorRule('ax_acc')])
    const w = result.warnings.find((x) => x.includes('ax_acc') && /self-recursive/i.test(x))
    assert.ok(w, `expected a self-recursive warning, got: ${JSON.stringify(result.warnings)}`)
    assert.match(w!, /chain/i, 'must offer the single chain rule alternative')
    assert.match(w!, /partial_sum|step/i, 'must offer the stepped-predicate alternative')
    assert.match(w!, /lt/, 'must mention the comparison-guard escape for bounded recursion')
  })

  it('transitive closure (recursion without value generation) stays unlectured', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'datalog classic' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'e1', predicate: 'edge', args: { from: 'a', to: 'b' } },
      {
        op: 'add_axiom',
        id: 'ax_base',
        label: 'edge reaches',
        when: [{ predicate: 'edge', args: { from: '?a', to: '?b' } }],
        then: [{ predicate: 'reachable', args: { from: '?a', to: '?b' } }],
      },
      {
        op: 'add_axiom',
        id: 'ax_trans',
        label: 'reach extends',
        when: [
          { predicate: 'reachable', args: { from: '?a', to: '?m' } },
          { predicate: 'edge', args: { from: '?m', to: '?b' } },
        ],
        then: [{ predicate: 'reachable', args: { from: '?a', to: '?b' } }],
      },
    ])
    assert.equal(
      result.warnings.filter((w) => /self-recursive/i.test(w)).length,
      0,
      `finite-domain recursion is legitimate datalog, got: ${JSON.stringify(result.warnings)}`,
    )
  })

  it('comparison-guarded recursive arithmetic is the sanctioned bounded form - silent', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'bounded counter' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax_count',
        label: 'count to ten',
        when: [
          { predicate: 'counter', args: { value: '?v' } },
          { predicate: 'lt', args: { left: '?v', right: 10 } },
          { predicate: 'add', args: { left: '?v', right: 1, result: '?n' } },
        ],
        then: [{ predicate: 'counter', args: { value: '?n' } }],
      },
    ])
    assert.equal(
      result.warnings.filter((w) => /self-recursive/i.test(w)).length,
      0,
      `lt-guarded recursion is the form the non-convergence teaching itself recommends`,
    )
  })

  it('when the closure does explode, the error diagnoses recursion - not just cross products', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'explosion diagnosis' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'c1', predicate: 'cost', args: { item: 'flange', total: 866982943513 } },
          { op: 'assert_fact', id: 'c2', predicate: 'cost', args: { item: 'turbine', total: 3090409370307 } },
          { op: 'assert_fact', id: 's0', predicate: 'sum_sofar', args: { value: 0 } },
          accumulatorRule('ax_a', 'flange'),
          accumulatorRule('ax_b', 'turbine'),
        ]),
      (error: Error) => {
        assert.match(error.message, /self-recursive/i, 'must name the real root cause')
        assert.match(error.message, /retract/i, 'must tell the model the way OUT (pinning will not fix it)')
        return true
      },
    )
  })
})


describe('stepped recursion is the RECOMMENDED pattern - it must not be lectured (cross-review catch)', () => {
  it('partial_sum(step=1) -> partial_sum(step=2) with add stays silent', () => {
    // Same predicate in head and body, arithmetic, no comparison guard -
    // but the step CONSTANTS differ between body and head, grounding the
    // recursion: each rule maps one fixed stratum to another, so a finite
    // rule set can only take finitely many steps. This is exactly the
    // alternative our own warning text recommends; warning on it would
    // lecture a model for following our advice.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'stepped fold' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'ax_s2',
        label: 'sum step1+gasket',
        when: [
          { predicate: 'partial_sum', args: { step: 1, value: '?s' } },
          { predicate: 'cost', args: { item: 'gasket', total: '?c' } },
          { predicate: 'add', args: { left: '?s', right: '?c', result: '?n' } },
        ],
        then: [{ predicate: 'partial_sum', args: { step: 2, value: '?n' } }],
      },
    ])
    assert.equal(
      result.warnings.filter((w) => /self-recursive/i.test(w)).length,
      0,
      `the sanctioned stepped pattern must not warn, got: ${JSON.stringify(result.warnings)}`,
    )
  })
})

describe('derive_aggregate v1: the engine writes the chain rule (compiler, not assembly)', () => {
  const costs = (...rows: Array<[string, number]>): WorkingMemoryOperation[] =>
    rows.map(([item, total], i) => ({
      op: 'assert_fact' as const,
      id: `c${i + 1}`,
      predicate: 'cost',
      args: { item, total },
    }))

  it('sums numeric facts into a single closure-derived total (exact, 13-digit)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate sum' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...costs(['a', 1302015645420], ['b', 1692229929455], ['c', 866982943513]),
      {
        op: 'derive_aggregate',
        id: 'agg_total',
        source: { predicate: 'cost', valueArg: 'total' },
        into: { predicate: 'grand_total', valueArg: 'value' },
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'grand_total')
    assert.equal(totals.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(totals[0]!.atom.args?.value, 1302015645420 + 1692229929455 + 866982943513)
    assert.ok(totals[0]!.derived, 'the total must be closure-derived, not asserted')
  })

  it('count needs no valueArg and derives the fact count', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate count' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...costs(['a', 1], ['b', 2], ['c', 3]),
      {
        op: 'derive_aggregate',
        id: 'agg_n',
        kind: 'count',
        source: { predicate: 'cost' },
        into: { predicate: 'cost_count', valueArg: 'value' },
      },
    ])
    const counts = result.workingMemory.facts.filter((f) => f.atom.predicate === 'cost_count')
    assert.equal(counts.length, 1)
    assert.equal(counts[0]!.atom.args?.value, 3)
    assert.ok(counts[0]!.derived)
  })

  it('re-running with the same id replaces the rule - the total follows the facts', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate rerun' })
    const agg: WorkingMemoryOperation = {
      op: 'derive_aggregate',
      id: 'agg_total',
      source: { predicate: 'cost', valueArg: 'total' },
      into: { predicate: 'grand_total', valueArg: 'value' },
    }
    applyWorkingMemoryOperations(store, space.id, [...costs(['a', 10], ['b', 32]), agg])
    const after = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'c9', predicate: 'cost', args: { item: 'z', total: 100 } },
      agg,
    ])
    const totals = after.workingMemory.facts.filter((f) => f.atom.predicate === 'grand_total')
    assert.equal(totals.length, 1, `stale totals must not survive: ${JSON.stringify(totals.map((t) => t.atom))}`)
    assert.equal(totals[0]!.atom.args?.value, 142)
  })

  it('zero matching facts is a teaching error, not a silent empty rule', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate empty' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'derive_aggregate',
            id: 'agg_total',
            source: { predicate: 'cost', valueArg: 'total' },
            into: { predicate: 'grand_total', valueArg: 'value' },
          },
        ]),
      /no active "cost" facts|assert.*first/i,
    )
  })

  it('a source fact missing the numeric value arg is named in the error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate bad fact' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'ok1', predicate: 'cost', args: { item: 'a', total: 5 } },
          { op: 'assert_fact', id: 'bad1', predicate: 'cost', args: { item: 'b', note: 'missing total' } },
          {
            op: 'derive_aggregate',
            id: 'agg_total',
            source: { predicate: 'cost', valueArg: 'total' },
            into: { predicate: 'grand_total', valueArg: 'value' },
          },
        ]),
      (error: Error) => /bad1/.test(error.message) && /total/.test(error.message),
    )
  })

  it('aggregating straight into finding(...) is refused - findings need real derivation', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate finding guard' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          ...costs(['a', 1]),
          {
            op: 'derive_aggregate',
            id: 'agg_f',
            source: { predicate: 'cost', valueArg: 'total' },
            into: { predicate: 'finding', valueArg: 'value' },
          },
        ]),
      /finding/i,
    )
  })
})

describe('derive_aggregate same batch (defer expansion past the first closure)', () => {
  // The model writes line + cost-rule + aggregate in ONE update_working_memory.
  // Before deferral, expandAggregate ran inside the op loop (BEFORE closure), so
  // the same-batch add_axiom had not derived any cost facts yet → "no active cost
  // facts" and the whole batch rolled back. After deferral the aggregate expands
  // AFTER the first closure, so it sees the derived cost facts.
  it('same batch: line + add_axiom(cost=unit*qty) + derive_aggregate(sum) derives grand_total', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate same batch' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'l1', predicate: 'line', args: { item: 'bolt', unit: 3, qty: 4 } },
      { op: 'assert_fact', id: 'l2', predicate: 'line', args: { item: 'nut', unit: 5, qty: 2 } },
      {
        op: 'add_axiom',
        id: 'ax_cost',
        label: 'cost = unit * qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
      {
        op: 'derive_aggregate',
        id: 'agg_total',
        source: { predicate: 'cost', valueArg: 'total' },
        into: { predicate: 'grand_total', valueArg: 'value' },
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'grand_total')
    assert.equal(totals.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(totals[0]!.atom.args?.value, 22) // 3*4 + 5*2
    assert.ok(totals[0]!.derived, 'the total is materialized by the 2nd closure - still a derived fact')
  })

  it('same batch is order-independent: derive_aggregate before a later source assert still counts it', () => {
    // derive_aggregate sits at index 2, the second line at index 3 (AFTER it). Deferral makes the
    // aggregate see the WHOLE batch's source facts (the board is a set, not a sequence), so the
    // line asserted after it is still aggregated.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate order independent' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'l1', predicate: 'line', args: { item: 'bolt', unit: 3, qty: 4 } },
      {
        op: 'add_axiom',
        id: 'ax_cost',
        label: 'cost = unit * qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
      {
        op: 'derive_aggregate',
        id: 'agg_total',
        source: { predicate: 'cost', valueArg: 'total' },
        into: { predicate: 'grand_total', valueArg: 'value' },
      },
      { op: 'assert_fact', id: 'l2', predicate: 'line', args: { item: 'nut', unit: 5, qty: 2 } },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'grand_total')
    assert.equal(totals.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(totals[0]!.atom.args?.value, 22) // both lines counted despite l2 coming after the aggregate
  })

  it('same batch group_by over a derived source emits one fact per bucket', () => {
    // Raw txn facts → a rule derives sale(region, amount) → derive_aggregate group_by region.
    // Exercises the deferred path with multi-bucket (__g_*) expansion over CLOSURE-derived source facts.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate same batch group_by' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 't1', predicate: 'txn', args: { region: 'east', amt: 10 } },
      { op: 'assert_fact', id: 't2', predicate: 'txn', args: { region: 'east', amt: 5 } },
      { op: 'assert_fact', id: 't3', predicate: 'txn', args: { region: 'west', amt: 20 } },
      {
        op: 'add_axiom',
        id: 'ax_sale',
        label: 'txn -> sale',
        when: [{ predicate: 'txn', args: { region: '?r', amt: '?a' } }],
        then: [{ predicate: 'sale', args: { region: '?r', amount: '?a' } }],
      },
      {
        op: 'derive_aggregate',
        id: 'agg_region',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'region_total', valueArg: 'value' },
        group_by: 'region',
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    assert.equal(totals.length, 2, JSON.stringify(totals.map((t) => t.atom)))
    const byRegion = new Map(totals.map((t) => [t.atom.args?.region, t.atom.args?.value]))
    assert.equal(byRegion.get('east'), 15)
    assert.equal(byRegion.get('west'), 20)
    assert.ok(totals.every((t) => t.derived))
  })

  it('chained aggregates in ONE batch resolve: B sums A.into (group then total)', () => {
    // agg_grand.source = agg_region.into. Declared BEFORE its dependency to prove the resolver is
    // iterative (closure -> expand the ready ones -> closure again), not order-following: closure #1
    // makes agg_region ready, it expands, closure #2 materializes region_total, which makes agg_grand
    // ready, it expands, closure #3 materializes grand_total.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate chained' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 's1', predicate: 'sale', args: { region: 'east', amount: 10 } },
      { op: 'assert_fact', id: 's2', predicate: 'sale', args: { region: 'east', amount: 5 } },
      { op: 'assert_fact', id: 's3', predicate: 'sale', args: { region: 'west', amount: 20 } },
      // declared first, but depends on region_total which does not exist yet
      {
        op: 'derive_aggregate',
        id: 'agg_grand',
        source: { predicate: 'region_total', valueArg: 'value' },
        into: { predicate: 'grand_total', valueArg: 'value' },
      },
      {
        op: 'derive_aggregate',
        id: 'agg_region',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'region_total', valueArg: 'value' },
        group_by: 'region',
      },
    ])
    const regionTotals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    assert.equal(regionTotals.length, 2, JSON.stringify(regionTotals.map((t) => t.atom)))
    const grand = result.workingMemory.facts.filter((f) => f.atom.predicate === 'grand_total')
    assert.equal(grand.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(grand[0]!.atom.args?.value, 35) // (10+5) + 20
    assert.ok(grand[0]!.derived, 'the chained total is closure-derived')
  })

  it('a genuinely missing source still throws (the resolver does not loop or swallow the error)', () => {
    // no rule and no aggregate ever produces "ghost", so iterating cannot make it ready - the resolver
    // must surface the standard teaching error and roll the batch back, not spin.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate missing source' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'c1', predicate: 'cost', args: { item: 'a', total: 10 } },
          {
            op: 'derive_aggregate',
            id: 'agg_x',
            source: { predicate: 'ghost', valueArg: 'amount' },
            into: { predicate: 'ghost_total', valueArg: 'value' },
          },
        ]),
      /no active "ghost" facts/i,
    )
  })

  it('circular aggregate dependencies throw rather than spin (A.source=B.into, B.source=A.into)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate circular' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          {
            op: 'derive_aggregate',
            id: 'agg_a',
            source: { predicate: 'b_total', valueArg: 'value' },
            into: { predicate: 'a_total', valueArg: 'value' },
          },
          {
            op: 'derive_aggregate',
            id: 'agg_b',
            source: { predicate: 'a_total', valueArg: 'value' },
            into: { predicate: 'b_total', valueArg: 'value' },
          },
        ]),
      /no active "(a_total|b_total)" facts/i,
    )
  })
})

// ============================================================ atomicity (RED)
// Contract tests for closure-phase atomicity - intentionally RED on main
// until the staging-store implementation lands (Codex, agent-sync handoff
// 2026-06-13). The wedge they pin: ops persist BEFORE the closure runs, so
// a closure-throwing batch leaves poison on the board and every later
// apply re-throws (#32 problem 8 burned its remaining turns this way).
// Do not weaken the assertions to get green - implement the rollback.

describe('closure-phase atomicity: a failed batch must leave NO trace (staging contract)', () => {
  const explosiveBatch = (): WorkingMemoryOperation[] => [
    { op: 'assert_fact', id: 'c1', predicate: 'cost', args: { item: 'flange', total: 866982943513 } },
    { op: 'assert_fact', id: 'c2', predicate: 'cost', args: { item: 'turbine', total: 3090409370307 } },
    { op: 'assert_fact', id: 's0', predicate: 'sum_sofar', args: { value: 0 } },
    {
      op: 'add_axiom',
      id: 'ax_a',
      label: 'accumulate flange',
      when: [
        { predicate: 'cost', args: { item: 'flange', total: '?t' } },
        { predicate: 'sum_sofar', args: { value: '?s' } },
        { predicate: 'add', args: { left: '?s', right: '?t', result: '?n' } },
      ],
      then: [{ predicate: 'sum_sofar', args: { value: '?n' } }],
    },
    {
      op: 'add_axiom',
      id: 'ax_b',
      label: 'accumulate turbine',
      when: [
        { predicate: 'cost', args: { item: 'turbine', total: '?t' } },
        { predicate: 'sum_sofar', args: { value: '?s' } },
        { predicate: 'add', args: { left: '?s', right: '?t', result: '?n' } },
      ],
      then: [{ predicate: 'sum_sofar', args: { value: '?n' } }],
    },
  ]

  it('T1: a closure explosion still teaches, but the board stays untouched', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'atomic T1' })
    const before = store.listNodes(space.id).length
    assert.throws(
      () => applyWorkingMemoryOperations(store, space.id, explosiveBatch()),
      /self-recursive|join explosion|did not converge/i,
      'the teaching diagnosis must survive the rollback',
    )
    assert.equal(
      store.listNodes(space.id).length,
      before,
      'a batch whose closure throws must not leave ANY node behind (facts or rules)',
    )
  })

  it('T2: an execution-phase error mid-batch rolls back the ops applied before it', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'atomic T2' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 'f1', predicate: 'observation', args: { line: 7 } },
          // throws during the post-closure aggregate expansion: this batch has no rule that
          // derives "cost" and asserts none, so even after closure #1 there is nothing to fold.
          {
            op: 'derive_aggregate',
            id: 'agg1',
            source: { predicate: 'cost', valueArg: 'total' },
            into: { predicate: 'grand_total', valueArg: 'value' },
          },
        ]),
      /no active "cost" facts/i,
    )
    assert.equal(
      store.listNodes(space.id).length,
      0,
      'the assert_fact before the failing op must be rolled back with it',
    )
  })

  it('T3: the space is NOT wedged after a failed batch - later applies work normally', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'atomic T3' })
    assert.throws(() => applyWorkingMemoryOperations(store, space.id, explosiveBatch()))
    // The #32 wedge: this second apply used to re-run the poisoned closure
    // and re-throw forever. After atomicity it must succeed and derive.
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'l1', predicate: 'line', args: { item: 'bolt', unit: 3, qty: 4 } },
      {
        op: 'add_axiom',
        id: 'ax_ok',
        label: 'cost = unit*qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
    ])
    const cost = result.workingMemory.facts.find((f) => f.atom.predicate === 'cost')
    assert.ok(cost?.derived, 'post-failure batch must apply and derive normally')
    assert.equal(cost?.atom.args?.total, 12)
  })

  it('T4: rollback preserves the pre-batch board exactly - earlier derivations intact', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'atomic T4' })
    const good = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'l1', predicate: 'line', args: { item: 'bolt', unit: 5, qty: 6 } },
      {
        op: 'add_axiom',
        id: 'ax_ok',
        label: 'cost = unit*qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
    ])
    assert.ok(good.workingMemory.facts.some((f) => f.derived && f.atom.args?.total === 30))
    const snapshot = store
      .listNodes(space.id)
      .map((n) => n.id)
      .sort()
    assert.throws(() => applyWorkingMemoryOperations(store, space.id, explosiveBatch()))
    const after = store
      .listNodes(space.id)
      .map((n) => n.id)
      .sort()
    assert.deepEqual(after, snapshot, 'node set must be byte-identical to pre-batch')
    const cost = getLogicContext(store, space.id).facts.find((f) => f.atom.predicate === 'cost')
    assert.ok(cost?.derived, 'derived facts from EARLIER batches must survive the rollback')
  })
})


describe('derive_aggregate v2: where filter', () => {
  const sales = (...rows: Array<[string, number]>): WorkingMemoryOperation[] =>
    rows.map(([region, amount], i) => ({
      op: 'assert_fact' as const,
      id: `s${i + 1}`,
      predicate: 'sale',
      args: { region, amount },
    }))

  it('sums only the subset matching an equality filter', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where subset' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 100], ['east', 250], ['west', 999]),
      {
        op: 'derive_aggregate',
        id: 'agg_east',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'east_total', valueArg: 'value' },
        where: { arg: 'region', equals: 'east' },
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'east_total')
    assert.equal(totals.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(totals[0]!.atom.args?.value, 350, 'only east rows summed (west excluded)')
    assert.ok(totals[0]!.derived, 'the filtered total must be closure-derived')
  })

  it('equals matches a canonicalized numeric value (string "2" == stored 2)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where numeric' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'q1', predicate: 'sale', args: { tier: 2, amount: 10 } },
      { op: 'assert_fact', id: 'q2', predicate: 'sale', args: { tier: 2, amount: 30 } },
      { op: 'assert_fact', id: 'q3', predicate: 'sale', args: { tier: 1, amount: 99 } },
      {
        op: 'derive_aggregate',
        id: 'agg_tier2',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'tier2_total', valueArg: 'value' },
        where: { arg: 'tier', equals: '2' },
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'tier2_total')
    assert.equal(totals.length, 1)
    assert.equal(totals[0]!.atom.args?.value, 40, 'string "2" matched stored 2')
  })

  it('zero matches is a teaching error that names the filter', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where empty' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          ...sales(['east', 5]),
          {
            op: 'derive_aggregate',
            id: 'agg_none',
            source: { predicate: 'sale', valueArg: 'amount' },
            into: { predicate: 'south_total', valueArg: 'value' },
            where: { arg: 'region', equals: 'south' },
          },
        ]),
      (error: Error) => /no active "sale" facts where region=/.test(error.message) && /south/.test(error.message),
    )
  })

  it('an arg absent on every source fact is named in the error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where bad arg' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          ...sales(['east', 5], ['west', 7]),
          {
            op: 'derive_aggregate',
            id: 'agg_bad',
            source: { predicate: 'sale', valueArg: 'amount' },
            into: { predicate: 'q_total', valueArg: 'value' },
            where: { arg: 'quarter', equals: 'Q1' },
          },
        ]),
      (error: Error) => /where\.arg "quarter" is not present/.test(error.message) && /region/.test(error.message),
    )
  })

  it('omitting where still aggregates every source fact (v1 unchanged)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'no where' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 4], ['west', 6]),
      {
        op: 'derive_aggregate',
        id: 'agg_all',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'all_total', valueArg: 'value' },
      },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'all_total')
    assert.equal(totals.length, 1)
    assert.equal(totals[0]!.atom.args?.value, 10, 'no filter -> all rows summed')
  })

  it('the rule label records the filter so the stale-pin risk is visible', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where label' })
    applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 100]),
      {
        op: 'derive_aggregate',
        id: 'agg_east',
        source: { predicate: 'sale', valueArg: 'amount' },
        into: { predicate: 'east_total', valueArg: 'value' },
        where: { arg: 'region', equals: 'east' },
      },
    ])
    const rule = store.listNodes(space.id).find((n) => n.id === 'agg_east')
    assert.ok(rule, 'the expanded rule node must exist under the op id')
    assert.match(rule!.label ?? '', /where region="east"/)
  })
})


describe('derive_aggregate min/max: extremum folds through the matching built-in', () => {
  const costs = (...rows: Array<[string, number]>): WorkingMemoryOperation[] =>
    rows.map(([item, total], i) => ({
      op: 'assert_fact' as const,
      id: `c${i + 1}`,
      predicate: 'cost',
      args: { item, total },
    }))

  it('min folds many facts down to the smallest value (closure-derived)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate min' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...costs(['a', 42], ['b', 7], ['c', 19]),
      { op: 'derive_aggregate', id: 'agg_min', kind: 'min', source: { predicate: 'cost', valueArg: 'total' }, into: { predicate: 'cheapest', valueArg: 'value' } },
    ])
    const mins = result.workingMemory.facts.filter((f) => f.atom.predicate === 'cheapest')
    assert.equal(mins.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(mins[0]!.atom.args?.value, 7)
    assert.ok(mins[0]!.derived, 'the min must be closure-derived, not asserted')
  })

  it('max folds many facts down to the largest value', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate max' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...costs(['a', 42], ['b', 7], ['c', 19]),
      { op: 'derive_aggregate', id: 'agg_max', kind: 'max', source: { predicate: 'cost', valueArg: 'total' }, into: { predicate: 'priciest', valueArg: 'value' } },
    ])
    const maxes = result.workingMemory.facts.filter((f) => f.atom.predicate === 'priciest')
    assert.equal(maxes.length, 1)
    assert.equal(maxes[0]!.atom.args?.value, 42)
    assert.ok(maxes[0]!.derived)
  })

  it('a single source fact yields that value as the extremum', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate min singleton' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...costs(['solo', 314]),
      { op: 'derive_aggregate', id: 'agg_min1', kind: 'min', source: { predicate: 'cost', valueArg: 'total' }, into: { predicate: 'cheapest', valueArg: 'value' } },
    ])
    const mins = result.workingMemory.facts.filter((f) => f.atom.predicate === 'cheapest')
    assert.equal(mins.length, 1)
    assert.equal(mins[0]!.atom.args?.value, 314, 'min of one = that value')
    assert.ok(mins[0]!.derived)
  })

  it('min honors the where filter - extremum of the matching subset only', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate min where' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 's1', predicate: 'sale', args: { region: 'east', amount: 100 } },
      { op: 'assert_fact', id: 's2', predicate: 'sale', args: { region: 'east', amount: 40 } },
      { op: 'assert_fact', id: 's3', predicate: 'sale', args: { region: 'west', amount: 5 } },
      { op: 'derive_aggregate', id: 'agg_east_min', kind: 'min', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'east_min', valueArg: 'value' }, where: { arg: 'region', equals: 'east' } },
    ])
    const mins = result.workingMemory.facts.filter((f) => f.atom.predicate === 'east_min')
    assert.equal(mins.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(mins[0]!.atom.args?.value, 40, 'west=5 excluded; min of east subset is 40')
    assert.ok(mins[0]!.derived)
  })

  it('min without source.valueArg is a teaching error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'aggregate min no valueArg' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          ...costs(['a', 1]),
          { op: 'derive_aggregate', id: 'agg_bad_min', kind: 'min', source: { predicate: 'cost' }, into: { predicate: 'cheapest', valueArg: 'value' } },
        ]),
      (error: Error) => /kind "min" needs source\.valueArg/.test(error.message),
    )
  })
})


describe('derive_aggregate v3: group_by emits one fact per bucket', () => {
  const sales = (...rows: Array<[string, number]>): WorkingMemoryOperation[] =>
    rows.map(([region, amount], i) => ({
      op: 'assert_fact' as const, id: `s${i + 1}`, predicate: 'sale', args: { region, amount },
    }))

  it('sums per region and carries the region value into each result fact', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group sum' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 100], ['east', 250], ['west', 999], ['west', 1]),
      { op: 'derive_aggregate', id: 'agg_region', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, group_by: 'region' },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    assert.equal(totals.length, 2, JSON.stringify(totals.map((t) => t.atom)))
    const byRegion = new Map(totals.map((t) => [t.atom.args?.region, t.atom.args?.value]))
    assert.equal(byRegion.get('east'), 350)
    assert.equal(byRegion.get('west'), 1000)
    assert.ok(totals.every((t) => t.derived), 'every per-group total must be closure-derived')
  })

  it('composes with where: filter first, then group', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group + where' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'a1', predicate: 'sale', args: { region: 'east', q: 'Q1', amount: 10 } },
      { op: 'assert_fact', id: 'a2', predicate: 'sale', args: { region: 'east', q: 'Q2', amount: 5 } },
      { op: 'assert_fact', id: 'a3', predicate: 'sale', args: { region: 'west', q: 'Q1', amount: 7 } },
      { op: 'assert_fact', id: 'a4', predicate: 'sale', args: { region: 'west', q: 'Q1', amount: 3 } },
      { op: 'derive_aggregate', id: 'agg_q1', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'q1_region_total', valueArg: 'value' }, where: { arg: 'q', equals: 'Q1' }, group_by: 'region' },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'q1_region_total')
    assert.equal(totals.length, 2)
    const byRegion = new Map(totals.map((t) => [t.atom.args?.region, t.atom.args?.value]))
    assert.equal(byRegion.get('east'), 10)
    assert.equal(byRegion.get('west'), 10)
  })

  it('a single group degenerates to one fact still carrying the group value', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group single' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 4], ['east', 6]),
      { op: 'derive_aggregate', id: 'agg_one', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, group_by: 'region' },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    assert.equal(totals.length, 1)
    assert.equal(totals[0]!.atom.args?.region, 'east')
    assert.equal(totals[0]!.atom.args?.value, 10)
  })

  it('a group_by arg absent on every source fact is named in the error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group bad arg' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          ...sales(['east', 5], ['west', 7]),
          { op: 'derive_aggregate', id: 'agg_bad', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'dept_total', valueArg: 'value' }, group_by: 'dept' },
        ]),
      (error: Error) => /group_by "dept" is not present/.test(error.message) && /region/.test(error.message),
    )
  })

  it('re-running the same id replaces the WHOLE family - no stale per-group rules', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group rerun' })
    const agg: WorkingMemoryOperation = { op: 'derive_aggregate', id: 'agg_region', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, group_by: 'region' }
    applyWorkingMemoryOperations(store, space.id, [...sales(['east', 10], ['west', 20]), agg])
    const after = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 's9', predicate: 'sale', args: { region: 'east', amount: 5 } },
      { op: 'assert_fact', id: 's10', predicate: 'sale', args: { region: 'south', amount: 99 } },
      agg,
    ])
    const totals = after.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    const byRegion = new Map(totals.map((t) => [t.atom.args?.region, t.atom.args?.value]))
    assert.equal(totals.length, 3, JSON.stringify(totals.map((t) => t.atom)))
    assert.equal(byRegion.get('east'), 15)
    assert.equal(byRegion.get('west'), 20)
    assert.equal(byRegion.get('south'), 99)
    const ruleNodes = store.listNodes(space.id).filter((n) => n.id.startsWith('agg_region__g_'))
    assert.equal(ruleNodes.length, 3, 'exactly one rule node per current bucket')
  })

  it('the rule label records the group', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'group label' })
    applyWorkingMemoryOperations(store, space.id, [
      ...sales(['east', 100]),
      { op: 'derive_aggregate', id: 'agg_region', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, group_by: 'region' },
    ])
    const rule = store.listNodes(space.id).find((n) => n.id.startsWith('agg_region__g_'))
    assert.ok(rule)
    assert.match(rule!.label ?? '', /grouped by region="east"/)
  })
})


describe('derive_aggregate group_by: partial key coverage fails visibly (Codex review)', () => {
  it('throws when some facts lack the group key (silent-undercount guard)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'partial group key' })
    assert.throws(
      () =>
        applyWorkingMemoryOperations(store, space.id, [
          { op: 'assert_fact', id: 's1', predicate: 'sale', args: { region: 'east', amount: 10 } },
          { op: 'assert_fact', id: 's2', predicate: 'sale', args: { amount: 5 } }, // no region
          { op: 'derive_aggregate', id: 'agg', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, group_by: 'region' },
        ]),
      (error: Error) => /group_by "region" is missing on 1 of 2/.test(error.message) && /undercount/i.test(error.message),
    )
  })

  it('a where filter that leaves only keyed facts makes group_by clean again', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'where rescues group' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 's1', predicate: 'sale', args: { region: 'east', tier: 'a', amount: 10 } },
      { op: 'assert_fact', id: 's2', predicate: 'sale', args: { tier: 'b', amount: 5 } }, // no region, tier b
      { op: 'derive_aggregate', id: 'agg', source: { predicate: 'sale', valueArg: 'amount' }, into: { predicate: 'region_total', valueArg: 'value' }, where: { arg: 'tier', equals: 'a' }, group_by: 'region' },
    ])
    const totals = result.workingMemory.facts.filter((f) => f.atom.predicate === 'region_total')
    assert.equal(totals.length, 1)
    assert.equal(totals[0]!.atom.args?.region, 'east')
    assert.equal(totals[0]!.atom.args?.value, 10)
  })
})

describe('derive_aggregate avg (sum/count, IEEE)', () => {
  const scores = (...rows: Array<[string, number]>): WorkingMemoryOperation[] =>
    rows.map(([who, score], i) => ({
      op: 'assert_fact' as const, id: `s${i + 1}`, predicate: 'score', args: { who, score },
    }))

  it('averages numeric facts (exact when divisible)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'avg exact' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...scores(['a', 10], ['b', 20], ['c', 30]),
      { op: 'derive_aggregate', id: 'agg_avg', kind: 'avg', source: { predicate: 'score', valueArg: 'score' }, into: { predicate: 'mean', valueArg: 'value' } },
    ])
    const means = result.workingMemory.facts.filter((f) => f.atom.predicate === 'mean')
    assert.equal(means.length, 1, JSON.stringify(result.workingMemory.facts.map((f) => f.atom)))
    assert.equal(means[0]!.atom.args?.value, 20)
    assert.ok(means[0]!.derived)
  })

  it('a single fact averages to that value', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'avg one' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...scores(['solo', 42]),
      { op: 'derive_aggregate', id: 'agg_avg', kind: 'avg', source: { predicate: 'score', valueArg: 'score' }, into: { predicate: 'mean', valueArg: 'value' } },
    ])
    assert.equal(result.workingMemory.facts.find((f) => f.atom.predicate === 'mean')?.atom.args?.value, 42)
  })

  it('a non-divisible mean is an IEEE float (declared contract, not a failure)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'avg float' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      ...scores(['a', 10], ['b', 11]),
      { op: 'derive_aggregate', id: 'agg_avg', kind: 'avg', source: { predicate: 'score', valueArg: 'score' }, into: { predicate: 'mean', valueArg: 'value' } },
    ])
    assert.equal(result.workingMemory.facts.find((f) => f.atom.predicate === 'mean')?.atom.args?.value, 10.5)
  })

  it('avg composes with where + group_by (mean per group of a subset)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'avg group' })
    const result = applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'a1', predicate: 'score', args: { team: 'x', kind: 'final', score: 10 } },
      { op: 'assert_fact', id: 'a2', predicate: 'score', args: { team: 'x', kind: 'final', score: 30 } },
      { op: 'assert_fact', id: 'a3', predicate: 'score', args: { team: 'y', kind: 'final', score: 8 } },
      { op: 'assert_fact', id: 'a4', predicate: 'score', args: { team: 'y', kind: 'draft', score: 99 } },
      { op: 'derive_aggregate', id: 'agg', kind: 'avg', source: { predicate: 'score', valueArg: 'score' }, into: { predicate: 'team_mean', valueArg: 'value' }, where: { arg: 'kind', equals: 'final' }, group_by: 'team' },
    ])
    const means = result.workingMemory.facts.filter((f) => f.atom.predicate === 'team_mean')
    const byTeam = new Map(means.map((m) => [m.atom.args?.team, m.atom.args?.value]))
    assert.equal(byTeam.get('x'), 20)
    assert.equal(byTeam.get('y'), 8, 'y draft excluded by where; y final mean is 8')
  })

  it('avg without valueArg is a teaching error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'avg no arg' })
    assert.throws(
      () => applyWorkingMemoryOperations(store, space.id, [
        ...scores(['a', 1]),
        { op: 'derive_aggregate', id: 'bad', kind: 'avg', source: { predicate: 'score' }, into: { predicate: 'mean', valueArg: 'value' } },
      ]),
      /kind "avg" needs source\.valueArg/,
    )
  })
})

describe('committed_derivation: ② domain-pack hard enforcement — the domain owns a head definition', () => {
  const seedCommitted = (store: MemorySpaceStore, id: string): void => {
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'assert_fact', id: 'CD_safe', predicate: 'committed_derivation', args: { predicate: 'safe' } }] as WorkingMemoryOperation[],
      { source: 'system', createdBy: 'system' },
    )
  }
  const weakSafeRule: WorkingMemoryOperation = {
    op: 'add_axiom',
    id: 'AX_weak_safe',
    label: 'weak safe',
    when: [{ predicate: 'has_password', args: { node: '?n' } }],
    then: [{ predicate: 'safe', args: { node: '?n' } }],
  } as WorkingMemoryOperation
  const hasNode = (store: MemorySpaceStore, id: string, nodeId: string): boolean => store.listNodes(id).some((n) => n.id === nodeId)

  it('R1: a MODEL add_axiom deriving a committed head (safe) is blocked, whole batch rolled back', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r1' })
    seedCommitted(store, id)
    assert.throws(() => applyWorkingMemoryOperations(store, id, [weakSafeRule], { source: 'model' }), GoalpostMovingError)
    assert.ok(!hasNode(store, id, 'AX_weak_safe'), 'the competing weak rule was NOT added (whole-batch rollback)')
  })

  it('R2: a TRUSTED (system) batch may define the committed head — the domain owns it', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r2' })
    seedCommitted(store, id)
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'add_axiom', id: 'AX_strong_safe', label: 'strong safe', when: [{ predicate: 'has_2fa', args: { node: '?n' } }, { predicate: 'encrypted', args: { node: '?n' } }], then: [{ predicate: 'safe', args: { node: '?n' } }] }] as WorkingMemoryOperation[],
      { source: 'system', createdBy: 'system' },
    )
    assert.ok(hasNode(store, id, 'AX_strong_safe'), 'the domain may define the head it owns')
  })

  it('R3: a MODEL add_axiom deriving a NON-committed head is allowed (only committed heads are owned)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r3' })
    seedCommitted(store, id)
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'add_axiom', id: 'AX_other', label: 'other', when: [{ predicate: 'foo', args: { node: '?n' } }], then: [{ predicate: 'other', args: { node: '?n' } }] }] as WorkingMemoryOperation[],
      { source: 'model' },
    )
    assert.ok(hasNode(store, id, 'AX_other'), 'a rule deriving a non-committed head is fine')
  })

  it('R4: a MODEL retract of the committed_derivation marker is blocked (cannot lift the lock)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r4' })
    seedCommitted(store, id)
    assert.throws(() => applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'CD_safe' }] as WorkingMemoryOperation[], { source: 'model' }), GoalpostMovingError)
    assert.ok(hasNode(store, id, 'CD_safe'), 'the committed_derivation marker survives the blocked retract')
  })

  it('R5: a MODEL-forged committed_derivation does NOT count (trusted-only) — it cannot lock a head', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r5' })
    // a model asserts committed_derivation(safe) itself — not trusted, must not become a real lock
    applyWorkingMemoryOperations(store, id, [{ op: 'assert_fact', id: 'CD_forge', predicate: 'committed_derivation', args: { predicate: 'safe' } }] as WorkingMemoryOperation[], { source: 'model' })
    // so a model add_axiom deriving safe is NOT blocked by the forged marker
    applyWorkingMemoryOperations(store, id, [weakSafeRule], { source: 'model' })
    assert.ok(hasNode(store, id, 'AX_weak_safe'), 'a forged (model-sourced) marker does not lock the head — committedHeads is trusted-only')
  })

  it('R6: opt-in inert — with NO committed_derivation, a model may add_axiom freely (zero regression)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r6' })
    applyWorkingMemoryOperations(store, id, [weakSafeRule], { source: 'model' })
    assert.ok(hasNode(store, id, 'AX_weak_safe'), 'no committed_derivation on the board → no restriction')
  })

  it('R7: a MODEL may READ a committed head in a rule body — only PRODUCING it is blocked', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r7' })
    seedCommitted(store, id)
    // body reads safe, head is a DIFFERENT predicate → allowed (read-not-produce)
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'add_axiom', id: 'AX_uses_safe', label: 'uses safe', when: [{ predicate: 'safe', args: { node: '?n' } }], then: [{ predicate: 'deploy_ok', args: { node: '?n' } }] }] as WorkingMemoryOperation[],
      { source: 'model' },
    )
    assert.ok(hasNode(store, id, 'AX_uses_safe'), 'reading the committed head in a rule body is fine')
  })

  // ---- 2b: the domain's defining rule is protected from model deletion ----
  // ---- 2d: a trusted derivation amendment consumed in-batch authorizes a one-shot definition change ----
  const seedDomainRule = (store: MemorySpaceStore, id: string): void => {
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'add_axiom', id: 'AX_domain_safe', label: 'domain safe', when: [{ predicate: 'has_2fa', args: { node: '?n' } }, { predicate: 'encrypted', args: { node: '?n' } }], then: [{ predicate: 'safe', args: { node: '?n' } }] }] as WorkingMemoryOperation[],
      { source: 'system', createdBy: 'system' },
    )
  }
  const seedAmendment = (store: MemorySpaceStore, id: string, predicate: string, nodeId: string, model = false): void => {
    applyWorkingMemoryOperations(
      store,
      id,
      [{ op: 'assert_fact', id: nodeId, predicate: 'amendment_result', args: { kind: 'derivation', predicate } }] as WorkingMemoryOperation[],
      model ? { source: 'model' } : { source: 'system', createdBy: 'system' },
    )
  }

  it('R8 (2b): a MODEL retract of a TRUSTED rule defining the committed head is blocked', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r8' })
    seedCommitted(store, id)
    seedDomainRule(store, id)
    assert.throws(() => applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AX_domain_safe' }] as WorkingMemoryOperation[], { source: 'model' }), GoalpostMovingError)
    assert.ok(hasNode(store, id, 'AX_domain_safe'), 'the domain definition rule survives the blocked retract')
  })

  it('R9 (2b): a MODEL retract of a trusted rule deriving a NON-committed head is allowed', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r9' })
    seedCommitted(store, id)
    applyWorkingMemoryOperations(store, id, [{ op: 'add_axiom', id: 'AX_other', label: 'other', when: [{ predicate: 'foo', args: { node: '?n' } }], then: [{ predicate: 'other', args: { node: '?n' } }] }] as WorkingMemoryOperation[], { source: 'system', createdBy: 'system' })
    applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AX_other' }] as WorkingMemoryOperation[], { source: 'model' })
    assert.ok(!hasNode(store, id, 'AX_other'), 'a rule deriving a non-committed head can be retracted')
  })

  it('R10 (2b): opt-in inert — with NO committed_derivation, a model may retract any rule', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r10' })
    seedDomainRule(store, id) // trusted rule deriving safe, but NO committed_derivation marker
    applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AX_domain_safe' }] as WorkingMemoryOperation[], { source: 'model' })
    assert.ok(!hasNode(store, id, 'AX_domain_safe'), 'no marker → no rule protection (zero regression)')
  })

  it('R11 (2d): consuming a trusted derivation amendment lets the model ADD a rule deriving the head (one-shot)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r11' })
    seedCommitted(store, id)
    seedAmendment(store, id, 'safe', 'AM_safe')
    applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AM_safe' }, weakSafeRule] as WorkingMemoryOperation[], { source: 'model' })
    assert.ok(hasNode(store, id, 'AX_weak_safe'), 'the authorized (amended) definition change landed')
    assert.ok(!hasNode(store, id, 'AM_safe'), 'the amendment was consumed (one-shot)')
  })

  it('R12 (2d): consuming the amendment also lets the model RETRACT the domain rule (revise the definition)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r12' })
    seedCommitted(store, id)
    seedDomainRule(store, id)
    seedAmendment(store, id, 'safe', 'AM_safe')
    applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AM_safe' }, { op: 'retract_node', nodeId: 'AX_domain_safe' }] as WorkingMemoryOperation[], { source: 'model' })
    assert.ok(!hasNode(store, id, 'AX_domain_safe'), 'the authorized definition retraction landed')
  })

  it('R13 (2d): the amendment must be CONSUMED in the same batch — present-but-unconsumed still blocks', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r13' })
    seedCommitted(store, id)
    seedAmendment(store, id, 'safe', 'AM_safe')
    // add weak safe WITHOUT retracting AM_safe → not one-shot consumed → blocked
    assert.throws(() => applyWorkingMemoryOperations(store, id, [weakSafeRule] as WorkingMemoryOperation[], { source: 'model' }), GoalpostMovingError)
    assert.ok(!hasNode(store, id, 'AX_weak_safe'), 'an unconsumed amendment does not authorize the change')
  })

  it('R14 (2d): an amendment for a DIFFERENT predicate does not unlock this head', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r14' })
    seedCommitted(store, id)
    seedAmendment(store, id, 'other', 'AM_other') // amendment is for 'other', not 'safe'
    assert.throws(() => applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AM_other' }, weakSafeRule] as WorkingMemoryOperation[], { source: 'model' }), GoalpostMovingError)
    assert.ok(!hasNode(store, id, 'AX_weak_safe'), 'an amendment for another predicate does not authorize redefining safe')
  })

  it('R15 (2d): a MODEL-forged amendment does not count (trusted-only)', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'cd-r15' })
    seedCommitted(store, id)
    seedAmendment(store, id, 'safe', 'AM_forge', true) // model-sourced amendment
    assert.throws(() => applyWorkingMemoryOperations(store, id, [{ op: 'retract_node', nodeId: 'AM_forge' }, weakSafeRule] as WorkingMemoryOperation[], { source: 'model' }), GoalpostMovingError)
    assert.ok(!hasNode(store, id, 'AX_weak_safe'), 'a model-forged amendment cannot authorize the change')
  })
})
