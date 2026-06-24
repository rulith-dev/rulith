import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations } from './working-memory.js'
import { explain, formatExplanation } from './explain.js'

/**
 * explain: derivation-trace ("why is this fact true?")
 *
 * Non-vacuity proof for each test:
 * (a) 2-level chain — without explain() the tree markers/parents would be absent;
 *     the test asserts derived marker + non-empty parents at level 1, and
 *     asserted marker + empty parents at level 2. Reversing the marker
 *     assignment in explain.ts would break both assertions.
 * (b) Asserted leaf — without markerOf() defaulting to 'asserted', the node
 *     would wrongly claim 'derived'. The test checks marker==='asserted' and
 *     parents.length===0 on a plain assert_fact node.
 * (c) Unknown id — without the guard throw, the call would crash with an
 *     opaque store error instead of the teaching message. The test checks
 *     both throw AND the message text.
 */

describe('explain: derivation trace', () => {
  /**
   * (a) 2-level derivation: asserted obs → axiom → derived intermediate →
   *     axiom2 → derived finding.
   *
   * Board:
   *   OBS:  observation(line:7)          [asserted]
   *   AX1:  IF observation(?l) THEN detected(line:?l)
   *   AX2:  IF detected(line:?l) THEN finding(kind:issue, line:?l)
   *
   * After closure:
   *   derived:observation(line:7)... wait — closure only derives IDB.
   *   derived: detected(line:7)    [derived by AX1, parents=[OBS]]
   *   derived: finding(...)        [derived by AX2, parents=[detected node]]
   *
   * explain(finding node) must return:
   *   { marker:'derived', ruleId:'AX2', parents:[
   *     { marker:'derived', ruleId:'AX1', parents:[
   *       { marker:'asserted', ruleId:null, parents:[] }
   *     ]}
   *   ]}
   */
  it('(a) 2-level derivation traces to asserted leaves with correct markers', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'explain 2-level' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'OBS', predicate: 'observation', args: { line: 7 } },
      {
        op: 'add_axiom',
        id: 'AX1',
        label: 'observation to detected',
        when: [{ predicate: 'observation', args: { line: '?l' } }],
        then: [{ predicate: 'detected', args: { line: '?l' } }],
      },
      {
        op: 'add_axiom',
        id: 'AX2',
        label: 'detected to finding',
        when: [{ predicate: 'detected', args: { line: '?l' } }],
        then: [{ predicate: 'finding', args: { kind: 'issue', line: '?l' } }],
      },
    ])

    // Find the derived finding node id.
    const nodes = store.listNodes(space.id)
    const findingNode = nodes.find(
      (n) =>
        n.type === 'fact' &&
        n.semantic?.kind === 'predicate' &&
        n.semantic.predicate === 'finding',
    )
    assert.ok(findingNode, `expected a derived finding fact, got ids: ${nodes.map((n) => n.id).join(', ')}`)

    const tree = explain(store, space.id, findingNode.id)

    // Top-level: derived by AX2.
    assert.equal(tree.marker, 'derived', 'top node must be derived')
    assert.equal(tree.ruleId, 'AX2', 'top node must cite AX2')
    assert.equal(tree.ruleLabel, 'detected to finding')
    assert.equal(tree.parents.length, 1, 'finding has one parent (detected)')

    // Level 1: derived detected node, derived by AX1.
    const detectedNode = tree.parents[0]
    assert.equal(detectedNode.marker, 'derived', 'detected must be derived')
    assert.equal(detectedNode.ruleId, 'AX1', 'detected must cite AX1')
    assert.equal(detectedNode.ruleLabel, 'observation to detected')
    assert.equal(detectedNode.parents.length, 1, 'detected has one parent (OBS)')

    // Level 2: asserted OBS leaf.
    const obsNode = detectedNode.parents[0]
    assert.equal(obsNode.marker, 'asserted', 'OBS must be asserted')
    assert.equal(obsNode.nodeId, 'OBS')
    assert.equal(obsNode.ruleId, null, 'asserted leaf has no ruleId')
    assert.deepEqual(obsNode.parents, [], 'asserted leaf has no parents')

    // formatExplanation must mention all three node ids and markers.
    const text = formatExplanation(tree)
    assert.match(text, /derived/, 'formatted text must mention derived')
    assert.match(text, /asserted/, 'formatted text must mention asserted')
    assert.match(text, /AX2/, 'formatted text must mention AX2')
    assert.match(text, /AX1/, 'formatted text must mention AX1')
    // OBS id appears somewhere in the indented subtree.
    assert.match(text, /OBS/, 'formatted text must mention OBS')
  })

  /**
   * (b) Asserted leaf: a bare assert_fact returns a single leaf with
   *     marker='asserted', no parents, no ruleId.
   *
   * Non-vacuous: if the marker defaulted to 'derived' or parents were
   * populated from the (empty) evidenceRefs the assertions would fail.
   */
  it('(b) an asserted fact returns a single leaf with marker asserted', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'explain leaf' })

    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'F_BARE', predicate: 'temperature', args: { celsius: 42 } },
    ])

    const tree = explain(store, space.id, 'F_BARE')

    assert.equal(tree.nodeId, 'F_BARE')
    assert.equal(tree.marker, 'asserted')
    assert.equal(tree.ruleId, null)
    assert.deepEqual(tree.parents, [])
    assert.equal(tree.cycleDetected, undefined)

    // formatExplanation for a leaf must be a single line (no newline inside).
    const text = formatExplanation(tree)
    assert.ok(!text.includes('\n'), `leaf explanation must be one line, got: ${text}`)
    assert.match(text, /asserted/)
    assert.match(text, /F_BARE/)
  })

  /**
   * (c) Unknown id throws a teaching error.
   *
   * Non-vacuous: without the guard in explain.ts the call would propagate
   * whatever error the store throws (which may be opaque and carry no
   * teaching text). The test checks both the throw and the message.
   */
  it('(c) unknown fact id throws a teaching error', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'explain unknown' })

    assert.throws(
      () => explain(store, space.id, 'DOES_NOT_EXIST'),
      (err: unknown) => {
        assert.ok(err instanceof Error)
        assert.match(
          err.message,
          /unknown node id.*DOES_NOT_EXIST/i,
          `teaching error must name the missing id, got: ${(err as Error).message}`,
        )
        // Must tell the model where to look.
        assert.match(
          err.message,
          /logic_context|board/i,
          `teaching error must point at the board, got: ${(err as Error).message}`,
        )
        return true
      },
    )
  })

  /**
   * Bonus (d): non-fact node id throws a teaching error naming the actual type.
   */
  it('(d) passing a non-fact node id throws a teaching error naming the type', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'explain non-fact' })

    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX_RULE',
        label: 'some rule',
        when: [{ predicate: 'p', args: {} }],
        then: [{ predicate: 'q', args: {} }],
      },
    ])

    assert.throws(
      () => explain(store, space.id, 'AX_RULE'),
      (err: unknown) => {
        assert.ok(err instanceof Error)
        assert.match(
          err.message,
          /axiom/i,
          `error must mention the actual type "axiom", got: ${(err as Error).message}`,
        )
        return true
      },
    )
  })
})
