import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { BoardSolver, verifySolverCertificate, type SolverCertificate } from './solver.js'

/** A triangle a-b-c: 3-colorable, NOT 2-colorable (adjacent nodes must differ). */
function coloringBoard(): { store: MemorySpaceStore; spaceId: string } {
  const store = new MemorySpaceStore()
  const { id } = store.createSpace({ title: 'coloring' })
  applyWorkingMemoryOperations(store, id, [
    { op: 'assert_fact', id: 'E_ab', predicate: 'edge', args: { a: 'a', b: 'b' } },
    { op: 'assert_fact', id: 'E_bc', predicate: 'edge', args: { a: 'b', b: 'c' } },
    { op: 'assert_fact', id: 'E_ac', predicate: 'edge', args: { a: 'a', b: 'c' } },
    {
      op: 'add_axiom', id: 'AX_COLOR', label: 'adjacent nodes must differ',
      when: [
        { predicate: 'edge', args: { a: '?x', b: '?y' } },
        { predicate: 'assignment', args: { var: '?x', value: '?c' } },
        { predicate: 'assignment', args: { var: '?y', value: '?c' } },
      ],
      then: [{ predicate: 'conflict', args: { x: '?x', y: '?y' } }],
    },
  ] as WorkingMemoryOperation[], { source: 'system' })
  return { store, spaceId: id }
}

const solver = new BoardSolver()

describe('Solver — delegation seam (➕): certificate or honest gap, core re-verifies', () => {
  it('plan: an already-satisfied goal returns an (empty) plan certificate the core accepts', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'plan-sat' })
    applyWorkingMemoryOperations(store, id, [
      { op: 'assert_fact', id: 'F', predicate: 'target', args: {} },
      { op: 'declare_goal', id: 'G', label: 'reach target', desired: [{ predicate: 'target', args: {} }] },
    ] as WorkingMemoryOperation[], { source: 'system' })

    const r = solver.solve(store, id, { kind: 'plan' })
    assert.equal(r.ok, true)
    if (r.ok) {
      assert.equal(r.certificate.kind, 'plan')
      assert.equal(verifySolverCertificate(store, id, r.certificate).verified, true)
    }
  })

  it('plan: an unreachable goal is an honest gap (ok:false), not a false claim', () => {
    const store = new MemorySpaceStore()
    const { id } = store.createSpace({ title: 'plan-unreach' })
    applyWorkingMemoryOperations(store, id, [
      { op: 'declare_goal', id: 'G', label: 'reach target', desired: [{ predicate: 'target', args: {} }] },
    ] as WorkingMemoryOperation[], { source: 'system' })

    const r = solver.solve(store, id, { kind: 'plan' })
    assert.equal(r.ok, false)
    if (!r.ok) assert.equal(r.reason, 'unsat')
  })

  it('constraint: a 3-colorable triangle returns an assignment certificate the closure re-verifies', () => {
    const { store, spaceId } = coloringBoard()
    const r = solver.solve(store, spaceId, {
      kind: 'constraint',
      spec: { variables: ['a', 'b', 'c'].map((n) => ({ name: n, domain: ['red', 'green', 'blue'] })) },
    })
    assert.equal(r.ok, true)
    if (r.ok) {
      assert.equal(r.certificate.kind, 'assignment')
      assert.equal(verifySolverCertificate(store, spaceId, r.certificate).verified, true)
    }
  })

  it('constraint: a triangle with only 2 colors is an honest gap (unsat)', () => {
    const { store, spaceId } = coloringBoard()
    const r = solver.solve(store, spaceId, {
      kind: 'constraint',
      spec: { variables: ['a', 'b', 'c'].map((n) => ({ name: n, domain: ['red', 'green'] })) },
    })
    assert.equal(r.ok, false)
    if (!r.ok) assert.equal(r.reason, 'unsat')
  })

  it('命门: a FORGED certificate is REJECTED by the core re-verification (a backend cannot lie)', () => {
    const { store, spaceId } = coloringBoard()
    // a and b given the SAME color on a triangle -> the closure derives a conflict.
    const forged: SolverCertificate = {
      kind: 'assignment',
      label: 'forged',
      facts: [
        { predicate: 'assignment', args: { var: 'a', value: 'red' } },
        { predicate: 'assignment', args: { var: 'b', value: 'red' } },
        { predicate: 'assignment', args: { var: 'c', value: 'green' } },
      ],
    }
    assert.equal(verifySolverCertificate(store, spaceId, forged).verified, false)
  })
})
