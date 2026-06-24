import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { RuleDefinition } from '../kernel/predicate.js'
import { compileBoardToSql, CompileBoardError } from './compile-board-sql.js'

// ── a simple 2-layer chain: grandparent from parent ─────────────────────────────
// parent(p, c)  [base]
// L1: ancestor1(a, d) :- parent(a, d)          (rename — non-recursive)
// L2: grandparent(g, c) :- parent(g, p), parent(p, c)
//
// (Both layer-1 here: grandparent reads only parent. To force a 2-layer chain we
//  derive an intermediate predicate first.)

const CHAIN_RULES: RuleDefinition[] = [
  {
    id: 'R_HAS_CHILD',
    when: [{ predicate: 'parent', args: { p: '?x', c: '?y' } }],
    then: [{ predicate: 'has_child', args: { who: '?x' } }],
  },
  {
    id: 'R_GRANDPARENT',
    when: [
      { predicate: 'parent', args: { p: '?g', c: '?m' } },
      { predicate: 'has_child', args: { who: '?m' } },
      { predicate: 'parent', args: { p: '?m', c: '?gc' } },
    ],
    then: [{ predicate: 'grandparent', args: { g: '?g', gc: '?gc' } }],
  },
]

test('2-layer chain produces two views in dependency order', () => {
  const { sql, viewsSql, layers, baseTables, views } = compileBoardToSql(CHAIN_RULES)
  // base table = parent
  assert.deepEqual(
    baseTables.map((t) => t.predicate),
    ['parent'],
    'parent is the only base table',
  )
  // two derived views
  assert.deepEqual(
    views.map((v) => v.predicate).sort(),
    ['grandparent', 'has_child'],
  )
  // has_child must be in a STRICTLY lower layer than grandparent (it reads it)
  const hcLayer = views.find((v) => v.predicate === 'has_child')!.layer
  const gpLayer = views.find((v) => v.predicate === 'grandparent')!.layer
  assert.ok(hcLayer < gpLayer, `has_child (L${hcLayer}) must precede grandparent (L${gpLayer})`)

  // The view for has_child must be CREATEd before the one referencing it.
  const hcIdx = viewsSql.indexOf('CREATE VIEW "has_child"')
  const gpIdx = viewsSql.indexOf('CREATE VIEW "grandparent"')
  assert.ok(hcIdx >= 0 && gpIdx >= 0, 'both views present')
  assert.ok(hcIdx < gpIdx, 'has_child view defined before grandparent view (no forward ref)')

  // base table DDL present
  assert.ok(sql.includes('CREATE TABLE "parent"'), 'parent base table created')
  assert.equal(layers.length, 2, 'two dependency layers')
})

// ── NAF rule → NOT EXISTS anti-join ──────────────────────────────────────────────

const NAF_RULES: RuleDefinition[] = [
  {
    id: 'R_MATCHED',
    when: [
      { predicate: 'employee', args: { id: '?e', dept: '?d' } },
      { predicate: 'department', args: { id: '?d' } },
    ],
    then: [{ predicate: 'placed', args: { emp: '?e' } }],
  },
  {
    id: 'R_ORPHAN',
    when: [
      { predicate: 'employee', args: { id: '?e', dept: '?d' } },
      { predicate: 'placed', args: { emp: '?e' }, naf: true },
    ],
    then: [{ predicate: 'orphan', args: { emp: '?e' } }],
  },
]

test('naf rule compiles to NOT EXISTS anti-join', () => {
  const { viewsSql } = compileBoardToSql(NAF_RULES)
  assert.ok(viewsSql.includes('NOT EXISTS'), 'naf becomes NOT EXISTS')
  // the anti-join binds emp against the bound ?e column
  assert.match(
    viewsSql,
    /NOT EXISTS \(SELECT 1 FROM "placed" AS n0 WHERE .*"emp" = /,
    'NOT EXISTS references placed with bound emp condition',
  )
})

// ── comparison builtin (gte) → WHERE >= ──────────────────────────────────────────

const GTE_RULES: RuleDefinition[] = [
  {
    id: 'R_BIG',
    when: [
      { predicate: 'order', args: { id: '?o', amount: '?a' } },
      { predicate: 'gte', args: { left: '?a', right: 200 } },
    ],
    then: [{ predicate: 'big_order', args: { id: '?o' } }],
  },
]

test('gte builtin compiles to a WHERE >= comparison', () => {
  const { viewsSql } = compileBoardToSql(GTE_RULES)
  assert.match(viewsSql, />=\s*200/, 'gte 200 becomes >= 200 in WHERE')
  assert.ok(viewsSql.includes('CREATE VIEW "big_order"'), 'big_order view present')
})

test('between / contains builtins map to BETWEEN and INSTR', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'R_MID',
      when: [
        { predicate: 'item', args: { id: '?i', score: '?s', name: '?n' } },
        { predicate: 'between', args: { value: '?s', low: 10, high: 20 } },
        { predicate: 'contains', args: { left: '?n', right: 'pro' } },
      ],
      then: [{ predicate: 'mid', args: { id: '?i' } }],
    },
  ]
  const { viewsSql } = compileBoardToSql(rules)
  assert.match(viewsSql, /BETWEEN\s+10\s+AND\s+20/, 'between → BETWEEN low AND high')
  assert.match(viewsSql, /INSTR\(.*,\s*'pro'\)\s*>\s*0/, 'contains → INSTR(...) > 0')
})

// ── multi-rule predicate → UNION ──────────────────────────────────────────────────

test('a predicate derived by two rules becomes a UNION of selects', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'R_A',
      when: [{ predicate: 'src_a', args: { id: '?x' } }],
      then: [{ predicate: 'flagged', args: { id: '?x' } }],
    },
    {
      id: 'R_B',
      when: [{ predicate: 'src_b', args: { id: '?x' } }],
      then: [{ predicate: 'flagged', args: { id: '?x' } }],
    },
  ]
  const { viewsSql } = compileBoardToSql(rules)
  assert.ok(viewsSql.includes('UNION'), 'two arms unioned')
  assert.ok(viewsSql.includes('-- rule R_A') && viewsSql.includes('-- rule R_B'), 'both rule arms emitted')
})

// ── fail-visible: unsupported constructs throw CompileBoardError ───────────────────

test('arithmetic builtin in body throws CompileBoardError', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'R_ARITH',
      when: [
        { predicate: 'order', args: { id: '?o', amount: '?a' } },
        { predicate: 'add', args: { left: '?a', right: 1, result: '?r' } },
      ],
      then: [{ predicate: 'bumped', args: { id: '?o', total: '?r' } }],
    },
  ]
  assert.throws(() => compileBoardToSql(rules), CompileBoardError, 'arithmetic producer rejected')
  assert.throws(() => compileBoardToSql(rules), /add/, 'error names the builtin')
})

test('recursive predicate throws CompileBoardError', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'R_REACH_BASE',
      when: [{ predicate: 'edge', args: { from: '?a', to: '?b' } }],
      then: [{ predicate: 'reach', args: { from: '?a', to: '?b' } }],
    },
    {
      id: 'R_REACH_STEP',
      when: [
        { predicate: 'reach', args: { from: '?a', to: '?b' } },
        { predicate: 'edge', args: { from: '?b', to: '?c' } },
      ],
      then: [{ predicate: 'reach', args: { from: '?a', to: '?c' } }],
    },
  ]
  assert.throws(() => compileBoardToSql(rules), CompileBoardError, 'recursion rejected')
  assert.throws(() => compileBoardToSql(rules), /recursive/, 'error mentions recursion')
})

test('strong-negated body literal throws CompileBoardError', () => {
  const rules: RuleDefinition[] = [
    {
      id: 'R_NEG',
      when: [{ predicate: 'x', args: { id: '?i' }, negated: true } as any],
      then: [{ predicate: 'y', args: { id: '?i' } }],
    },
  ]
  assert.throws(() => compileBoardToSql(rules), CompileBoardError)
})
