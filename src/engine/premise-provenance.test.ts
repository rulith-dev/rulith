import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import {
  groundingOf,
  weakestPremises,
  type ProvenanceFact,
} from './premise-provenance.js'

/** Build a board, return its facts (LogicContextFact[] satisfies ProvenanceFact). */
function boardFacts(ops: WorkingMemoryOperation[]): ProvenanceFact[] {
  const store = new MemorySpaceStore()
  const { id } = store.createSpace({ title: 'prov' })
  applyWorkingMemoryOperations(store, id, ops, { source: 'system' })
  return getLogicContext(store, id).facts
}

const idOf = (facts: ProvenanceFact[], predicate: string, derived: boolean): string =>
  facts.find((f) => f.atom.predicate === predicate && f.derived === derived)!.nodeId

// A rule: fraud >= 100000 -> severity high. The verdict is SOUND; its only
// ground premise is the asserted fraud amount.
const SEV_RULE: WorkingMemoryOperation = {
  op: 'add_axiom',
  id: 'AX_SEV',
  label: 'large fraud is high severity',
  when: [
    { predicate: 'fraud_amount', args: { case: '?c', amount: '?a' } },
    { predicate: 'gte', args: { left: '?a', right: 100000 } },
  ],
  then: [{ predicate: 'severity', args: { case: '?c', level: 'high' } }],
} as WorkingMemoryOperation

describe('premise-provenance: a verdict is only as trustworthy as its weakest ground fact', () => {
  it('a derived verdict resting on an ASSERTED input is flagged ungrounded, naming the input to verify', () => {
    const facts = boardFacts([
      SEV_RULE,
      { op: 'assert_fact', id: 'F1', predicate: 'fraud_amount', args: { case: 'c1', amount: 1000000 } },
    ] as WorkingMemoryOperation[])

    const severity = idOf(facts, 'severity', true)
    const g = groundingOf(facts, severity)

    assert.equal(g.weakestTier, 'asserted', 'the verdict rests on an asserted input')
    assert.ok(g.ungrounded, 'nothing trusted underneath — pure agent word')
    assert.deepEqual(g.assertedPremises, ['F1'], 'the asserted fraud_amount is the premise to verify')
    assert.deepEqual(g.attestedPremises, [])
  })

  it('declaring the same input ATTESTED (machine-vouched predicate) lifts the verdict to fully grounded', () => {
    const facts = boardFacts([
      SEV_RULE,
      { op: 'assert_fact', id: 'F1', predicate: 'fraud_amount', args: { case: 'c1', amount: 1000000 } },
    ] as WorkingMemoryOperation[])

    const severity = idOf(facts, 'severity', true)
    const g = groundingOf(facts, severity, { attestedPredicates: ['fraud_amount'] })

    assert.equal(g.weakestTier, 'attested')
    assert.ok(!g.ungrounded, 'rests on a trusted channel now')
    assert.deepEqual(g.attestedPremises, ['F1'])
    assert.deepEqual(g.assertedPremises, [])
  })

  it('walks a DEEP chain past derived intermediates to the real ground premise', () => {
    const facts = boardFacts([
      {
        op: 'add_axiom',
        id: 'AX_FLAG',
        label: 'large fraud is flagged',
        when: [
          { predicate: 'fraud_amount', args: { case: '?c', amount: '?a' } },
          { predicate: 'gte', args: { left: '?a', right: 100000 } },
        ],
        then: [{ predicate: 'flagged', args: { case: '?c' } }],
      },
      {
        op: 'add_axiom',
        id: 'AX_SEV2',
        label: 'flagged is high severity',
        when: [{ predicate: 'flagged', args: { case: '?c' } }],
        then: [{ predicate: 'severity', args: { case: '?c', level: 'high' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'fraud_amount', args: { case: 'c1', amount: 1000000 } },
    ] as WorkingMemoryOperation[])

    const severity = idOf(facts, 'severity', true)
    const g = groundingOf(facts, severity)

    // flagged (a derived intermediate) is an inference step, not a premise —
    // the rollup descends through it to the asserted fraud_amount.
    assert.deepEqual(g.assertedPremises, ['F1'])
    assert.equal(g.weakestTier, 'asserted')
  })

  it('a MIX of asserted + attested premises is capped at the weakest (asserted), but not ungrounded', () => {
    const facts = boardFacts([
      {
        op: 'add_axiom',
        id: 'AX_SEV3',
        label: 'large filed fraud is high severity',
        when: [
          { predicate: 'fraud_amount', args: { case: '?c', amount: '?a' } },
          { predicate: 'report_filed', args: { case: '?c' } },
          { predicate: 'gte', args: { left: '?a', right: 100000 } },
        ],
        then: [{ predicate: 'severity', args: { case: '?c', level: 'high' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'fraud_amount', args: { case: 'c1', amount: 1000000 } },
      { op: 'assert_fact', id: 'R1', predicate: 'report_filed', args: { case: 'c1' } },
    ] as WorkingMemoryOperation[])

    const severity = idOf(facts, 'severity', true)
    const g = groundingOf(facts, severity, { attestedPredicates: ['report_filed'] })

    assert.equal(g.weakestTier, 'asserted', 'one asserted input caps the verdict')
    assert.ok(!g.ungrounded, 'but there IS trusted evidence under it')
    assert.deepEqual(g.assertedPremises, ['F1'])
    assert.deepEqual(g.attestedPremises, ['R1'])
  })

  it('a ground fact from a trusted CHANNEL (createdBy tool/system) is attested without naming the predicate', () => {
    const trusted: ProvenanceFact[] = [
      { nodeId: 'g1', atom: { predicate: 'weight' }, evidenceRefs: [], derived: false, createdBy: 'tool' },
      { nodeId: 'v1', atom: { predicate: 'over_limit' }, evidenceRefs: ['RULE', 'g1'], derived: true },
    ]
    assert.equal(groundingOf(trusted, 'v1').weakestTier, 'attested')
    assert.ok(!groundingOf(trusted, 'v1').ungrounded)

    // same shape, but the weight is the model's free word -> asserted, ungrounded
    const freeWord: ProvenanceFact[] = [
      { nodeId: 'g1', atom: { predicate: 'weight' }, evidenceRefs: [], derived: false, createdBy: 'agent' },
      { nodeId: 'v1', atom: { predicate: 'over_limit' }, evidenceRefs: ['RULE', 'g1'], derived: true },
    ]
    assert.equal(groundingOf(freeWord, 'v1').weakestTier, 'asserted')
    assert.ok(groundingOf(freeWord, 'v1').ungrounded)
  })

  it('weakestPremises ranks the most-in-need-of-verification verdicts first', () => {
    const facts: ProvenanceFact[] = [
      { nodeId: 'a_in', atom: { predicate: 'x' }, evidenceRefs: [], derived: false, createdBy: 'tool' },
      { nodeId: 'a_out', atom: { predicate: 'verdictA' }, evidenceRefs: ['a_in'], derived: true },
      { nodeId: 'b_in', atom: { predicate: 'y' }, evidenceRefs: [], derived: false, createdBy: 'agent' },
      { nodeId: 'b_out', atom: { predicate: 'verdictB' }, evidenceRefs: ['b_in'], derived: true },
    ]
    const ranked = weakestPremises(facts, ['a_out', 'b_out'])
    assert.deepEqual(ranked.map((g) => g.factId), ['b_out', 'a_out'], 'asserted-grounded verdict first')
  })

  it('cycles in the evidence chain do not loop', () => {
    const cyclic: ProvenanceFact[] = [
      { nodeId: 'p', atom: { predicate: 'p' }, evidenceRefs: ['q'], derived: true },
      { nodeId: 'q', atom: { predicate: 'q' }, evidenceRefs: ['p'], derived: true },
    ]
    // both derived, cycle, no ground leaf — terminates, attested (vacuous), not ungrounded
    const g = groundingOf(cyclic, 'p')
    assert.equal(g.weakestTier, 'attested')
    assert.deepEqual(g.assertedPremises, [])
  })
})
