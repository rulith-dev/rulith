import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import type { LogicContextFact } from './logic-context.js'
import { diffFacts, formatDelta, factKind } from './board-diff.js'

function fact(
  nodeId: string,
  predicate: string,
  args: Record<string, unknown>,
  opts: { derived?: boolean; effect?: boolean } = {},
): LogicContextFact {
  return {
    nodeId,
    atom: { predicate, args } as LogicContextFact['atom'],
    status: 'verified',
    confidence: 1,
    evidenceRefs: [],
    createdBy: 'agent',
    derived: opts.derived ?? false,
    effect: opts.effect,
  }
}

describe('board-diff: turn-over-turn delta', () => {
  it('detects added facts (nodeId absent last turn) and marks them highlight', () => {
    const prev = [fact('L0', 'line', { item: 'valve' })]
    const curr = [fact('L0', 'line', { item: 'valve' }), fact('d1', 'cost', { item: 'valve', total: 10 }, { derived: true })]
    const d = diffFacts(prev, curr)
    assert.equal(d.added.length, 1)
    assert.equal(d.added[0]!.nodeId, 'd1')
    assert.ok(d.highlight.has('d1'))
    assert.equal(d.changed.length, 0)
    assert.deepEqual(d.removedIds, [])
  })

  it('detects a changed fact (same nodeId, different args)', () => {
    const d = diffFacts([fact('a1', 'qty', { n: 1 })], [fact('a1', 'qty', { n: 2 })])
    assert.equal(d.added.length, 0)
    assert.equal(d.changed.length, 1)
    assert.ok(d.highlight.has('a1'))
  })

  it('detects removed facts (consumed/retracted)', () => {
    const d = diffFacts([fact('x', 'gold', { amt: 5 })], [])
    assert.deepEqual(d.removedIds, ['x'])
    assert.equal(d.highlight.size, 0)
  })

  it('first turn (prev undefined): everything is added', () => {
    const d = diffFacts(undefined, [fact('L0', 'line', { item: 'valve' })])
    assert.equal(d.added.length, 1)
  })

  it('formatDelta: EMPTY delta warns the op produced nothing (the empty-board failure)', () => {
    const same = [fact('L0', 'line', { item: 'valve' })]
    const s = formatDelta(diffFacts(same, same))
    assert.match(s, /NOTHING CHANGED/)
    assert.match(s, /retry/i)
  })

  it('formatDelta: calls out DERIVED new facts vs bare assertions', () => {
    const derivedAdd = diffFacts([], [fact('d1', 'cost', { item: 'valve', total: 10 }, { derived: true })])
    assert.match(formatDelta(derivedAdd), /DERIVED/)
    const assertedAdd = diffFacts([], [fact('a1', 'line', { item: 'valve' })])
    assert.match(formatDelta(assertedAdd), /only assertions/i)
  })

  it('formatDelta: firstTurn suppresses the delta line', () => {
    const d = diffFacts(undefined, [fact('L0', 'line', { item: 'x' })])
    assert.equal(formatDelta(d, { firstTurn: true }), '')
  })

  it('factKind maps derived / effect / asserted', () => {
    assert.equal(factKind(fact('a', 'p', {}, { derived: true })), 'derived')
    assert.equal(factKind(fact('b', 'p', {}, { effect: true })), 'effect')
    assert.equal(factKind(fact('c', 'p', {})), 'asserted')
  })
})
