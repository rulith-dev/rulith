/**
 * Trust-lattice grounding (integration-fabric first increment): a conclusion's
 * floor = the weakest tier among its ground premises, across heterogeneous
 * engine outputs (verified DB / approximate numeric / uncertain ML). Legacy
 * attested/asserted behaviour is covered by premise-provenance.test.ts; here we
 * guard the typed lattice + zero-regression of the default path.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { groundingOf, type ProvenanceFact } from './premise-provenance.js'

const ground = (nodeId: string, predicate: string, trustTier: ProvenanceFact['trustTier']): ProvenanceFact => ({
  nodeId,
  atom: { predicate },
  evidenceRefs: [],
  derived: false,
  trustTier,
  createdBy: 'tool',
})
const derived = (nodeId: string, refs: string[]): ProvenanceFact => ({
  nodeId,
  atom: { predicate: 'verdict' },
  evidenceRefs: refs,
  derived: true,
})

test('floor = weakest tier; ML (uncertain) drags a verified+approximate verdict down', () => {
  const facts = [
    ground('db', 'income', 'verified'),
    ground('num', 'ratio', 'approximate'),
    ground('ml', 'risk', 'uncertain'),
    derived('v', ['db', 'num', 'ml']),
  ]
  const g = groundingOf(facts, 'v')
  assert.equal(g.weakestTier, 'uncertain')
  assert.deepEqual(g.weakestPremiseIds, ['ml'])
  assert.equal(g.ungrounded, false)
  assert.deepEqual(g.premisesByTier.verified, ['db'])
})

test('drop the ML input → floor rises to the numeric approximation', () => {
  const facts = [ground('db', 'income', 'verified'), ground('num', 'ratio', 'approximate'), derived('v', ['db', 'num'])]
  assert.equal(groundingOf(facts, 'v').weakestTier, 'approximate')
})

test('verified-only derivation → floor is verified', () => {
  const facts = [ground('db', 'income', 'verified'), derived('v', ['db'])]
  assert.equal(groundingOf(facts, 'v').weakestTier, 'verified')
})

test('zero regression: facts without trustTier keep attested/asserted floor', () => {
  const facts: ProvenanceFact[] = [
    { nodeId: 'tool', atom: { predicate: 'w' }, evidenceRefs: [], derived: false, createdBy: 'tool' },
    { nodeId: 'bare', atom: { predicate: 'c' }, evidenceRefs: [], derived: false },
    { nodeId: 'v', atom: { predicate: 'verdict' }, evidenceRefs: ['tool', 'bare'], derived: true },
  ]
  const g = groundingOf(facts, 'v')
  assert.equal(g.weakestTier, 'asserted')
  assert.deepEqual(g.attestedPremises, ['tool'])
  assert.deepEqual(g.assertedPremises, ['bare'])
  assert.equal(g.ungrounded, false) // has a trusted premise
})
