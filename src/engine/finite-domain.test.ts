import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import { solveFiniteDomain, type FdVariable, type FdValue } from './finite-domain.js'

/**
 * finite-domain — 慢轨 #2, first slice: the board as a constraint SOLVER, not
 * just a checker (constraint-agent already checks). The solver is a bounded
 * backtracking search that PROPOSES assignments; the board (closure) ADJUDICATES
 * each candidate. A buggy/incomplete solver can only return a rejected proposal
 * or honestly report "no solution" — it can never certify a wrong answer, because
 * the final assignment is verified by the exact closure. propose/adjudicate keeps
 * the core safe.
 */

describe('solveFiniteDomain: bounded search proposes, an adjudicator decides', () => {
  it('finds a satisfying assignment with a pure consistency check', () => {
    const vars: FdVariable[] = [
      { name: 'x', domain: [1, 2, 3, 4, 5] },
      { name: 'y', domain: [1, 2, 3, 4, 5] },
    ]
    // x + y = 7 AND x < y. Partial assignments are not yet falsifiable.
    const isConsistent = (p: Readonly<Record<string, FdValue>>): boolean => {
      if (typeof p.x === 'number' && typeof p.y === 'number') {
        return p.x + p.y === 7 && p.x < p.y
      }
      return true
    }
    const r = solveFiniteDomain(vars, isConsistent)
    assert.equal(r.sat, true)
    if (r.sat) {
      // Non-vacuous: the returned assignment must actually satisfy the constraint.
      assert.equal(Number(r.assignment.x) + Number(r.assignment.y), 7)
      assert.ok(Number(r.assignment.x) < Number(r.assignment.y))
    }
  })

  it('reports unsat (not a fabricated answer) when no assignment works', () => {
    const vars: FdVariable[] = [
      { name: 'x', domain: [1, 2, 3, 4, 5] },
      { name: 'y', domain: [1, 2, 3, 4, 5] },
    ]
    const r = solveFiniteDomain(vars, (p) =>
      !(typeof p.x === 'number' && typeof p.y === 'number') || p.x + p.y === 100,
    )
    assert.equal(r.sat, false)
    if (!r.sat) assert.equal(r.reason, 'unsat')
  })

  it('fails visibly on budget exhaustion rather than hanging', () => {
    const vars: FdVariable[] = [
      { name: 'x', domain: [1, 2, 3, 4, 5] },
      { name: 'y', domain: [1, 2, 3, 4, 5] },
    ]
    const r = solveFiniteDomain(vars, () => true, { maxNodes: 1 })
    // Non-vacuous: with a 1-node budget the search cannot complete 2 vars.
    assert.equal(r.sat, false)
    if (!r.sat) assert.equal(r.reason, 'budget')
  })

  it('adjudicates the empty assignment instead of auto-certifying zero-variable SAT', () => {
    const r = solveFiniteDomain([], () => false)
    assert.equal(r.sat, false)
    if (!r.sat) assert.equal(r.reason, 'unsat')
  })

  it('rejects duplicate variable names instead of silently overwriting assignments', () => {
    assert.throws(
      () => solveFiniteDomain(
        [{ name: 'x', domain: [1] }, { name: 'x', domain: [2] }],
        () => true,
      ),
      /duplicate variable name "x"/,
    )
  })

  it('board-adjudicated: the solver finds a schedule the CLOSURE certifies conflict-free', () => {
    // Same conflict rule as constraint-agent: two different meetings in one
    // room+slot collide, DERIVED by the closure.
    const conflictRule: WorkingMemoryOperation = {
      op: 'add_axiom', id: 'AX_CONFLICT', label: 'room+slot clash',
      when: [
        { predicate: 'meeting', args: { id: '?a', room: '?r', slot: '?s' } },
        { predicate: 'meeting', args: { id: '?b', room: '?r', slot: '?s' } },
        { predicate: 'neq', args: { left: '?a', right: '?b' } },
      ],
      then: [{ predicate: 'conflict', args: { room: '?r', slot: '?s', a: '?a', b: '?b' } }],
    } as WorkingMemoryOperation

    // Adjudicator = the board closure: lay the assignment as meeting facts in a
    // FRESH space alongside the rule, run closure, accept iff zero DERIVED conflict.
    const boardConsistent = (partial: Readonly<Record<string, FdValue>>): boolean => {
      const store = new MemorySpaceStore()
      const { id } = store.createSpace({ title: 'fd-check' })
      const ops: WorkingMemoryOperation[] = [conflictRule]
      for (const [name, slot] of Object.entries(partial)) {
        ops.push({ op: 'assert_fact', id: name, predicate: 'meeting', args: { id: name, room: 'r1', slot: String(slot) } })
      }
      applyWorkingMemoryOperations(store, id, ops, { source: 'system' })
      return !getLogicContext(store, id).facts.some((f) => f.atom.predicate === 'conflict' && f.derived)
    }

    // 2 meetings, 2 slots, same room -> SAT (one per slot).
    const sat = solveFiniteDomain(
      [
        { name: 'M1', domain: ['slot1', 'slot2'] },
        { name: 'M2', domain: ['slot1', 'slot2'] },
      ],
      boardConsistent,
    )
    assert.equal(sat.sat, true)
    if (sat.sat) {
      // The board must independently certify the proposed assignment conflict-free.
      assert.equal(boardConsistent(sat.assignment), true, 'closure certifies the solution')
      assert.notEqual(sat.assignment.M1, sat.assignment.M2, 'the two meetings took different slots')
    }

    // 3 meetings, 2 slots, same room -> pigeonhole UNSAT, driven by the closure.
    const unsat = solveFiniteDomain(
      [
        { name: 'M1', domain: ['slot1', 'slot2'] },
        { name: 'M2', domain: ['slot1', 'slot2'] },
        { name: 'M3', domain: ['slot1', 'slot2'] },
      ],
      boardConsistent,
    )
    assert.equal(unsat.sat, false)
    if (!unsat.sat) assert.equal(unsat.reason, 'unsat')
  })
})
