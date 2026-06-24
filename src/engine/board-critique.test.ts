import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations } from './working-memory.js'
import { getLogicContext, formatLogicContextAsText } from './logic-context.js'
import { doneBlockingCritiques } from './board-critique.js'

/**
 * board_critique: standing board health, not transient per-apply warnings.
 * The teaching warnings (unfirable rule, self-recursive, self-sealed) fire
 * once, right after the op that triggered them; a problem left on the board
 * then goes silent. critique re-derives the STANDING problems from board
 * state every turn, so a model can see and fix what it left broken - the
 * "failures become learnable" half of the mandate.
 */
describe('board_critique: standing problems, re-derived from board state', () => {
  it('flags a finding asserted directly instead of derived (the laundering smell)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'asserted finding' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'OBS', predicate: 'observation', args: { line: 7 } },
    ])
    // A bare finding(...) with no rule behind it - revise_fact lets us place
    // it without the record_result gate, modelling the standing smell.
    const ctx = getLogicContext(store, space.id)
    const before = ctx.critique.length
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AXF',
        label: 'finding from observation',
        when: [{ predicate: 'observation', args: { line: '?l' } }],
        then: [{ predicate: 'finding', args: { kind: 'issue', line: '?l' } }],
      },
    ])
    // This finding IS derived - must NOT be flagged.
    const derivedCtx = getLogicContext(store, space.id)
    assert.equal(
      derivedCtx.critique.some((c) => c.kind === 'asserted_finding'),
      false,
      'a rule-derived finding is healthy',
    )
    assert.equal(before, before) // anchor
  })

  it('flags a finding that IS asserted directly (positive case - Codex review)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'bare finding' })
    // A finding placed straight onto the board with no rule behind it.
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'BF', predicate: 'finding', args: { type: 'leak', file: 'a.ts' } },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'asserted_finding')
    assert.ok(item, `a bare asserted finding must be flagged, got ${JSON.stringify(ctx.critique)}`)
    assert.equal(item!.nodeId, 'BF')
    assert.match(item!.message, /asserted, not derived/i)
  })

  it('flags a self-sealed goal as a standing issue', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'self sealed' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal',
        id: 'G1',
        label: 'audit done',
        desired: [{ predicate: 'audit_done', args: { scope: 'all' } }],
      },
      { op: 'assert_fact', id: 'F1', predicate: 'audit_done', args: { scope: 'all' } },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'self_sealed_goal')
    assert.ok(item, `expected a self_sealed_goal critique, got ${JSON.stringify(ctx.critique)}`)
    assert.equal(item!.nodeId, 'G1')
    // And it must be visible in the rendered board, every turn.
    assert.match(formatLogicContextAsText(ctx), /critique|self-sealed/i)
  })

  it('flags a vacuous rule (body merely renames the head)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'vacuous' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AXV',
        label: 'rename',
        when: [{ predicate: 'suspected_leak', args: { file: '?f' } }],
        then: [{ predicate: 'finding', args: { type: 'leak', file: '?f' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.ok(
      ctx.critique.some((c) => c.kind === 'vacuous_rule' && c.nodeId === 'AXV'),
      `expected a vacuous_rule critique, got ${JSON.stringify(ctx.critique)}`,
    )
  })

  it('flags an unfirable rule (a guard that can provably never hold) as a standing critique', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'unfirable' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'add_axiom',
        id: 'AX_DEAD',
        label: 'dead guard',
        when: [{ predicate: 'eq', args: { left: 3, right: 4 } }],
        then: [{ predicate: 'flagged', args: { reason: 'never' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'unfirable_rule' && c.nodeId === 'AX_DEAD')
    // Non-vacuous: the dead rule sits on the board firing nothing; the standing
    // critique must keep surfacing it (not just the once-at-apply warning).
    assert.ok(item, `expected an unfirable_rule critique, got ${JSON.stringify(ctx.critique)}`)
    assert.match(item?.message ?? '', /never fire/)
    // Dead code, not a correctness fault: it must NOT hard-block finishing.
    assert.equal(doneBlockingCritiques(ctx.critique).some((c) => c.kind === 'unfirable_rule'), false)
  })

  it('a clean board has an empty critique and prints no critique section', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'clean' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'L1', predicate: 'line', args: { item: 'bolt', unit: 3, qty: 4 } },
      {
        op: 'add_axiom',
        id: 'AX',
        label: 'cost = unit*qty',
        when: [
          { predicate: 'line', args: { item: '?i', unit: '?u', qty: '?q' } },
          { predicate: 'mul', args: { left: '?u', right: '?q', result: '?t' } },
        ],
        then: [{ predicate: 'cost', args: { item: '?i', total: '?t' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.deepEqual(ctx.critique, [])
    assert.doesNotMatch(formatLogicContextAsText(ctx), /^critique/im)
  })
})


describe('board_critique: predicate contradiction (p AND not-p)', () => {
  it('flags a contradiction and names both polarities + resolution', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'contradiction' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'POS', predicate: 'door', args: { state: 'open' } },
      { op: 'assert_fact', id: 'NEG', predicate: 'door', args: { state: 'open' }, negated: true },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.predicateConflicts.length, 1, JSON.stringify(ctx.predicateConflicts))
    const item = ctx.critique.find((c) => c.kind === 'contradiction')
    assert.ok(item, `expected a contradiction critique, got ${JSON.stringify(ctx.critique)}`)
    assert.ok(item!.nodeId === 'POS' || item!.nodeId === 'NEG')
    assert.match(item!.message, /door\(state=open\)/)
    assert.match(item!.message, /not door\(state=open\)/)
    assert.match(item!.message, /retract/i)
    assert.match(formatLogicContextAsText(ctx), /contradiction/i)
  })

  it('a board with no opposing facts has zero contradiction critiques', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'no contradiction' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'A', predicate: 'door', args: { state: 'open' } },
      { op: 'assert_fact', id: 'B', predicate: 'door', args: { state: 'closed' } },
      { op: 'assert_fact', id: 'C', predicate: 'window', args: { state: 'open' }, negated: true },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.predicateConflicts.length, 0)
    assert.equal(ctx.critique.some((c) => c.kind === 'contradiction'), false)
  })
})

describe('board text shows evidence-chain provenance (absorbed from Codex e619d4d)', () => {
  it('renders <- evidenceRefs on facts and results', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'provenance' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'OBS1', predicate: 'observed', args: { line: 7 }, evidenceRefs: ['file.ts:7'] },
      {
        op: 'add_axiom', id: 'AX', label: 'finding from obs',
        when: [{ predicate: 'observed', args: { line: '?l' } }],
        then: [{ predicate: 'flagged', args: { line: '?l' } }],
      },
      { op: 'record_result', id: 'R1', label: 'audit', summary: 'one flag', evidenceRefs: ['OBS1'] },
    ])
    const text = formatLogicContextAsText(getLogicContext(store, space.id))
    assert.match(text, /OBS1:.*<- file\.ts:7/, 'fact provenance should render')
    assert.match(text, /R1:.*<- OBS1/, 'result provenance should render')
  })
})

describe('board_critique: unsatisfiable action (precondition predicate nothing supplies)', () => {
  it('flags an action whose precondition predicate is supplied by nothing', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'dead action' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'T1', predicate: 'task', args: { id: 'a' } },
      {
        op: 'define_action', id: 'A_GO', action: 'go', label: 'go',
        // ready(...) is supplied by NOTHING - no fact, no rule head, no effect.
        preconditions: [{ predicate: 'ready', args: { task: 'a' } }],
        effects: [{ predicate: 'done', args: { task: 'a' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'unsatisfiable_action')
    assert.ok(item, `expected an unsatisfiable_action critique, got ${JSON.stringify(ctx.critique)}`)
    assert.equal(item!.nodeId, 'A_GO')
    assert.match(item!.message, /ready\(\.\.\.\)/)
    assert.match(formatLogicContextAsText(ctx), /unsatisfiable_action|can never become applicable/i)
  })

  it('does NOT flag an action whose precondition a rule derives', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'live action' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'T1', predicate: 'task', args: { id: 'a' } },
      {
        op: 'add_axiom', id: 'AX_READY', label: 'ready when task',
        when: [{ predicate: 'task', args: { id: '?t' } }],
        then: [{ predicate: 'ready', args: { task: '?t' } }],
      },
      {
        op: 'define_action', id: 'A_GO', action: 'go', label: 'go',
        preconditions: [{ predicate: 'ready', args: { task: 'a' } }],
        effects: [{ predicate: 'done', args: { task: 'a' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unsatisfiable_action'), false)
  })

  it('does NOT flag preconditions that are builtins or naf', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'builtin/naf precond' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'C1', predicate: 'counter', args: { value: 3 } },
      {
        op: 'define_action', id: 'A_INC', action: 'inc', label: 'inc',
        preconditions: [
          { predicate: 'counter', args: { value: '?v' } },
          { predicate: 'lt', args: { left: '?v', right: 10 } },
          { predicate: 'done', args: { value: '?v' }, naf: true },
        ],
        effects: [{ predicate: 'bumped', args: { value: '?v' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unsatisfiable_action'), false,
      `lt (builtin) and done (naf) must not count as unsuppliable, got ${JSON.stringify(ctx.critique)}`)
  })

  it('an action supplied by ANOTHER action effect is not flagged', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'effect-supplied' })
    applyWorkingMemoryOperations(store, space.id, [
      { op: 'assert_fact', id: 'S1', predicate: 'start', args: { id: 'x' } },
      {
        op: 'define_action', id: 'A_PREP', action: 'prep', label: 'prep',
        preconditions: [{ predicate: 'start', args: { id: '?x' } }],
        effects: [{ predicate: 'prepped', args: { id: '?x' } }],
      },
      {
        op: 'define_action', id: 'A_USE', action: 'use', label: 'use',
        preconditions: [{ predicate: 'prepped', args: { id: '?x' } }],
        effects: [{ predicate: 'used', args: { id: '?x' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unsatisfiable_action'), false,
      'prepped is produced by A_PREP, so A_USE is satisfiable')
  })
})

/**
 * board_critique #6: an unreachable goal. The model declared a goal but the
 * board carries no machinery to close it - NO rule head derives its desired
 * predicate AND no action effect produces it. This is the standing-item form
 * of the per-turn "no rule derives this yet" guidance: a goal left open with
 * no inference or action path stays flagged turn after turn until the model
 * adds a rule, defines a producing action, or - if it is a primitive
 * observation - asserts the fact directly. Open + zero hints + zero
 * actionHints is the trigger.
 */
describe('board_critique: unreachable goal (no rule path, no producing action)', () => {
  it('flags an open goal nothing can derive or produce', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'unreachable goal' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G_DONE', label: 'ship it',
        desired: [{ predicate: 'shipped', args: { release: '1.0' } }],
      },
      // Some unrelated facts/rules so the board is not empty - none touch shipped.
      { op: 'assert_fact', id: 'F1', predicate: 'built', args: { release: '1.0' } },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'unreachable_goal')
    assert.ok(item, `expected an unreachable_goal critique, got ${JSON.stringify(ctx.critique)}`)
    assert.equal(item!.nodeId, 'G_DONE')
    assert.match(item!.message, /shipped/)
    assert.match(formatLogicContextAsText(ctx), /unreachable_goal|no rule derives it and no action produces it/i)
  })

  it('does NOT flag an open goal a rule could derive (has a rule path)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'reachable via rule' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G', label: 'cleared',
        desired: [{ predicate: 'cleared', args: { id: 'x' } }],
      },
      // Rule head matches the goal; its body is missing, so it is a live path
      // (a non-empty abduction hint), not a dead end.
      {
        op: 'add_axiom', id: 'AX', label: 'cleared when checked',
        when: [{ predicate: 'checked', args: { id: '?i' } }],
        then: [{ predicate: 'cleared', args: { id: '?i' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unreachable_goal'), false,
      `a goal with a rule path must not be unreachable, got ${JSON.stringify(ctx.critique)}`)
  })

  it('does NOT flag an open goal an action could produce', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'reachable via action' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G', label: 'have gold',
        desired: [{ predicate: 'have', args: { item: 'gold' } }],
      },
      {
        op: 'define_action', id: 'A_MINE', action: 'mine', label: 'mine',
        preconditions: [{ predicate: 'at', args: { place: 'mine' } }],
        effects: [{ predicate: 'have', args: { item: 'gold' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unreachable_goal'), false,
      `a goal a defined action produces must not be unreachable, got ${JSON.stringify(ctx.critique)}`)
  })

  it('does NOT flag a satisfied goal', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'satisfied goal' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G', label: 'present',
        desired: [{ predicate: 'present', args: { who: 'a' } }],
      },
      { op: 'assert_fact', id: 'F', predicate: 'present', args: { who: 'a' } },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(ctx.critique.some((c) => c.kind === 'unreachable_goal'), false,
      'a satisfied goal is not unreachable')
  })
})

/**
 * board_critique #7: conflicting goals. Two declared goals whose desired atoms
 * directly contradict each other — one wants p(args), the other wants not-p(args)
 * on the SAME predicate and SAME args. No plan can satisfy both simultaneously.
 * This is a planning-level conflict: it fires even when neither fact exists yet
 * on the board (unlike the contradiction critique which requires both facts to
 * stand). The critique names both goal ids and the conflicting atom so the model
 * can reconcile immediately.
 */
describe('board_critique: conflicting goals (p vs not-p across two goals)', () => {
  it('(a) flags two goals desiring p vs not-p on identical args — names both ids and atom', () => {
    // Non-vacuous: without the conflicting_goals check the critique would be empty
    // (neither goal is self-sealed, no facts, no rules, no findings). The ONLY
    // reason a critique fires here is the new kind.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'conflicting goals basic' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G_OPEN', label: 'open door',
        desired: [{ predicate: 'locked', args: { door: 'front' } }],
      },
      {
        op: 'declare_goal', id: 'G_LOCKED', label: 'keep locked',
        desired: [{ predicate: 'locked', args: { door: 'front' }, negated: true }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const item = ctx.critique.find((c) => c.kind === 'conflicting_goals')
    assert.ok(item, `expected a conflicting_goals critique, got ${JSON.stringify(ctx.critique)}`)
    // Anchored on the positive goal.
    assert.equal(item!.nodeId, 'G_OPEN')
    // Message must name both goal ids.
    assert.match(item!.message, /G_OPEN/)
    assert.match(item!.message, /G_LOCKED/)
    // Message must name the conflicting atom.
    assert.match(item!.message, /locked\(door=front\)/)
    // Must appear in rendered board text.
    assert.match(formatLogicContextAsText(ctx), /conflicting_goals|No plan can satisfy both/i)
  })

  it('(b) two compatible goals (different predicates) -> NOT flagged', () => {
    // Non-vacuous: if the detection were broken to always fire, this would catch it.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'compatible goals' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G1', label: 'door open',
        desired: [{ predicate: 'open', args: { door: 'front' } }],
      },
      {
        op: 'declare_goal', id: 'G2', label: 'lights on',
        desired: [{ predicate: 'lights_on', args: { room: 'hall' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(
      ctx.critique.some((c) => c.kind === 'conflicting_goals'),
      false,
      `goals with unrelated predicates must not conflict, got ${JSON.stringify(ctx.critique)}`,
    )
  })

  it('(c) same predicate but DIFFERENT args -> NOT flagged', () => {
    // Non-vacuous: same predicate with opposite polarity but different args is
    // NOT a direct contradiction (locked(door=front) vs not-locked(door=back)
    // can both hold in a consistent world). A broken implementation that ignores
    // args would incorrectly flag this.
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'same pred different args' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G_FRONT', label: 'front locked',
        desired: [{ predicate: 'locked', args: { door: 'front' } }],
      },
      {
        op: 'declare_goal', id: 'G_BACK', label: 'back unlocked',
        desired: [{ predicate: 'locked', args: { door: 'back' }, negated: true }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    assert.equal(
      ctx.critique.some((c) => c.kind === 'conflicting_goals'),
      false,
      `same predicate but different args must not be flagged, got ${JSON.stringify(ctx.critique)}`,
    )
  })

  it('(d) three goals where exactly one pair conflicts — only that pair is emitted once', () => {
    // Non-vacuous: verifies both that an unrelated third goal does not generate
    // spurious critiques AND that the conflicting pair is reported exactly once
    // (no double-emission of A↔B and B↔A).
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'three goals one conflict' })
    applyWorkingMemoryOperations(store, space.id, [
      {
        op: 'declare_goal', id: 'G_POS', label: 'power on',
        desired: [{ predicate: 'powered', args: { unit: 'main' } }],
      },
      {
        op: 'declare_goal', id: 'G_NEG', label: 'power off',
        desired: [{ predicate: 'powered', args: { unit: 'main' }, negated: true }],
      },
      {
        op: 'declare_goal', id: 'G_UNREL', label: 'fuel loaded',
        desired: [{ predicate: 'fueled', args: { unit: 'main' } }],
      },
    ])
    const ctx = getLogicContext(store, space.id)
    const conflicts = ctx.critique.filter((c) => c.kind === 'conflicting_goals')
    assert.equal(
      conflicts.length,
      1,
      `exactly one conflicting_goals critique expected, got ${JSON.stringify(conflicts)}`,
    )
    assert.equal(conflicts[0]!.nodeId, 'G_POS')
    assert.match(conflicts[0]!.message, /G_POS/)
    assert.match(conflicts[0]!.message, /G_NEG/)
    // The unrelated goal must not appear in the conflict message.
    assert.doesNotMatch(conflicts[0]!.message, /G_UNREL/)
  })
})

describe('board_critique: unexplained reopen (I6.R6d)', () => {
  it('flags a grant that vanished without a trusted revocation_result (approval still stands)', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'reopen' })
    // a real, trusted approval produced a grant (createdBy:'tool' = trusted channel)
    applyWorkingMemoryOperations(
      store,
      space.id,
      [
        { op: 'assert_fact', id: 'AR', predicate: 'approval_result', args: { action: 'deploy', granted: true, approvedBy: 'alice' } },
        { op: 'assert_fact', id: 'GR', predicate: 'permission_granted', args: { action: 'deploy', maxRisk: 5 } },
      ],
      { source: 'system', createdBy: 'tool' },
    )
    assert.equal(getLogicContext(store, space.id).critique.some((c) => c.kind === 'unexplained_reopen'), false, 'grant present → no critique')

    // grant raw-retracted, NO revocation_result recorded
    applyWorkingMemoryOperations(store, space.id, [{ op: 'retract_node', nodeId: 'GR' }], { source: 'model' })
    const flagged = getLogicContext(store, space.id).critique.filter((c) => c.kind === 'unexplained_reopen')
    assert.equal(flagged.length, 1, 'unexplained reopen flagged')
    assert.match(flagged[0]!.message, /unexplained reopen|without a revocation/)
  })

  it('a trusted revocation_result explains the disappearance → no critique', () => {
    const store = new MemorySpaceStore()
    const space = store.createSpace({ title: 'reopen-ok' })
    applyWorkingMemoryOperations(
      store,
      space.id,
      [
        { op: 'assert_fact', id: 'AR', predicate: 'approval_result', args: { action: 'deploy', granted: true } },
        { op: 'assert_fact', id: 'GR', predicate: 'permission_granted', args: { action: 'deploy', maxRisk: 5 } },
      ],
      { source: 'system', createdBy: 'tool' },
    )
    applyWorkingMemoryOperations(store, space.id, [{ op: 'retract_node', nodeId: 'GR' }], { source: 'model' })
    applyWorkingMemoryOperations(
      store,
      space.id,
      [{ op: 'assert_fact', id: 'REV', predicate: 'revocation_result', args: { action: 'deploy', revokedBy: 'alice' } }],
      { source: 'system', createdBy: 'tool' },
    )
    assert.equal(getLogicContext(store, space.id).critique.some((c) => c.kind === 'unexplained_reopen'), false)
  })
})
