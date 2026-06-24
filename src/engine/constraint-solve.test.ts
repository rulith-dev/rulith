import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import { solveConstraintsOnBoard } from './constraint-solve.js'

/**
 * constraint-solve — the board-integrated DRIVER for solveFiniteDomain: the
 * external caller that turns the bare search into a usable board capability.
 * Canonical CSP = graph coloring (variables = nodes, domain = colors, conflict
 * when adjacent nodes share a color). The search proposes colorings; the closure
 * adjudicates each candidate via the conflict rule; the certified coloring is
 * committed back and the closure re-verifies it on the real board.
 */

/** A triangle a-b-c: 3-colorable, NOT 2-colorable. */
function coloringBoard(): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const { id } = store.createSpace({ title: 'coloring' })
  const ops: WorkingMemoryOperation[] = [
    { op: 'assert_fact', id: 'E_ab', predicate: 'edge', args: { a: 'a', b: 'b' } },
    { op: 'assert_fact', id: 'E_bc', predicate: 'edge', args: { a: 'b', b: 'c' } },
    { op: 'assert_fact', id: 'E_ac', predicate: 'edge', args: { a: 'a', b: 'c' } },
    // Two adjacent nodes with the same color -> a DERIVED conflict.
    {
      op: 'add_axiom', id: 'AX_COLOR', label: 'adjacent nodes must differ',
      when: [
        { predicate: 'edge', args: { a: '?x', b: '?y' } },
        { predicate: 'assignment', args: { var: '?x', value: '?c' } },
        { predicate: 'assignment', args: { var: '?y', value: '?c' } },
      ],
      then: [{ predicate: 'conflict', args: { x: '?x', y: '?y', color: '?c' } }],
    },
  ]
  applyWorkingMemoryOperations(store, id, ops, { source: 'system' })
  return { store, spaceId: id }
}

const derivedConflicts = (store: MemorySpaceStore, spaceId: string): number =>
  getLogicContext(store, spaceId).facts.filter((f) => f.atom.predicate === 'conflict' && f.derived).length

describe('solveConstraintsOnBoard: search proposes, board closure adjudicates + certifies', () => {
  it('finds a proper 3-coloring, commits it, and the closure re-verifies zero conflict', () => {
    const { store, spaceId } = coloringBoard()
    const r = solveConstraintsOnBoard(store, spaceId, {
      variables: ['a', 'b', 'c'].map((n) => ({ name: n, domain: ['red', 'green', 'blue'] })),
    })

    assert.equal(r.sat, true)
    // The committed board must derive ZERO conflict (closure certifies the answer).
    assert.equal(derivedConflicts(store, spaceId), 0)

    // The assignment is on the board, and adjacent nodes genuinely differ.
    const ctx = getLogicContext(store, spaceId)
    const color = (n: string): unknown =>
      ctx.facts.find((f) => f.atom.predicate === 'assignment' && f.atom.args?.var === n)?.atom.args?.value
    assert.ok(color('a') && color('b') && color('c'), 'all three nodes are colored on the board')
    assert.notEqual(color('a'), color('b'))
    assert.notEqual(color('b'), color('c'))
    assert.notEqual(color('a'), color('c'))
  })

  it('reports unsat for a triangle with only 2 colors, committing nothing', () => {
    const { store, spaceId } = coloringBoard()
    const r = solveConstraintsOnBoard(store, spaceId, {
      variables: ['a', 'b', 'c'].map((n) => ({ name: n, domain: ['red', 'green'] })),
    })

    assert.equal(r.sat, false)
    if (!r.sat) assert.equal(r.reason, 'unsat')
    // Non-vacuous: an unsat solve must NOT leave a partial/garbage assignment behind.
    const assignmentFacts = getLogicContext(store, spaceId).facts.filter(
      (f) => f.atom.predicate === 'assignment',
    )
    assert.equal(assignmentFacts.length, 0, 'nothing committed on unsat')
  })

  it('reports unsat when a zero-variable board already derives a conflict', () => {
    const store = new MemorySpaceStore()
    const { id: spaceId } = store.createSpace({ title: 'base-conflict' })
    applyWorkingMemoryOperations(store, spaceId, [
      { op: 'assert_fact', id: 'BAD', predicate: 'bad', args: {} },
      {
        op: 'add_axiom', id: 'AX_BAD', label: 'bad means conflict',
        when: [{ predicate: 'bad', args: {} }],
        then: [{ predicate: 'conflict', args: { reason: 'base' } }],
      },
    ], { source: 'system' })

    const r = solveConstraintsOnBoard(store, spaceId, { variables: [] })
    assert.equal(r.sat, false)
    if (!r.sat) assert.equal(r.reason, 'unsat')
  })
})
