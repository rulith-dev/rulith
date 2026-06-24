import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { detectPredicateConflicts, detectFunctionalConflicts } from './conflict.js'

describe('conflict kernel', () => {
  it('detects positive and explicit negative facts with the same atom', () => {
    const conflicts = detectPredicateConflicts([
      { id: 'F1', atom: { predicate: 'at', args: { object: 'car', location: 'home' } } },
      { id: 'F2', atom: { predicate: 'at', args: { location: 'home', object: 'car' }, negated: true } },
      { id: 'F3', atom: { predicate: 'at', args: { object: 'car', location: 'shop' }, negated: true } },
    ])

    assert.deepEqual(conflicts, [
      {
        atom: { predicate: 'at', args: { object: 'car', location: 'home' } },
        positiveFactId: 'F1',
        negativeFactId: 'F2',
      },
    ])
  })
})

describe('functional-dependency conflicts (kernel)', () => {
  it('flags facts that share the key but disagree on the rest (derived vs asserted cost — arith p10)', () => {
    const conflicts = detectFunctionalConflicts(
      [
        { id: 'A_sensor', atom: { predicate: 'cost', args: { item: 'sensor', total: 388752151850 } } },
        { id: 'D_sensor', atom: { predicate: 'cost', args: { item: 'sensor', total: 3886450604850 } } },
        { id: 'A_manifold', atom: { predicate: 'cost', args: { item: 'manifold', total: 4106912068412 } } },
        { id: 'D_manifold', atom: { predicate: 'cost', args: { item: 'manifold', total: 4107101510012 } } },
      ],
      [{ predicate: 'cost', key: ['item'] }],
    )
    assert.equal(conflicts.length, 2, JSON.stringify(conflicts))
    const sensor = conflicts.find((c) => c.key.item === 'sensor')
    assert.ok(sensor, 'a conflict for item=sensor')
    assert.deepEqual(sensor!.factIds, ['A_sensor', 'D_sensor'])
  })

  it('does NOT flag identical facts (same key AND same value — a re-derivation of the same total)', () => {
    const conflicts = detectFunctionalConflicts(
      [
        { id: 'A', atom: { predicate: 'cost', args: { item: 'x', total: 5 } } },
        { id: 'B', atom: { predicate: 'cost', args: { item: 'x', total: 5 } } },
      ],
      [{ predicate: 'cost', key: ['item'] }],
    )
    assert.deepEqual(conflicts, [])
  })

  it('is inert with no declared dependency (kernel adjudicates, domain declares)', () => {
    const conflicts = detectFunctionalConflicts(
      [
        { id: 'A', atom: { predicate: 'cost', args: { item: 'x', total: 5 } } },
        { id: 'B', atom: { predicate: 'cost', args: { item: 'x', total: 9 } } },
      ],
      [],
    )
    assert.deepEqual(conflicts, [])
  })
})
