/**
 * compile-board-aggregate.test.ts — TRIPLE AGREEMENT for the AGGREGATION layer.
 *
 * compile-board-agreement.test.ts proved board ≡ JS pipeline ≡ SQL views for the
 * relational + NAF + comparison fragment. This file extends the SAME faithfulness
 * proof to AGGREGATES (derive_aggregate): sum / count / sum-with-group_by /
 * sum-with-where / min / max.
 *
 * THE THREE PATHS (each must produce the SAME aggregate-result fact SET, by atomKey):
 *   (1) BOARD: assert base facts + the rules, APPLY the derive_aggregate op (the
 *       engine expands it to its sanctioned chain rule + closes), read the
 *       [derived] target facts back from getLogicContext.
 *   (2) JS:    compileBoard(rules, { aggregates }).pipeline(baseFacts) — the
 *       aggregate is a reduce scheduled after its source layer.
 *   (3) SQL:   compileBoardToSql(rules, { aggregates }) → GROUP BY view, executed
 *       in node:sqlite.
 *
 * The aggregate spec (compile-level) maps field-for-field to derive_aggregate:
 *   target ← into.predicate, outArg ← into.valueArg, source ← source.predicate,
 *   valueArg ← source.valueArg, kind, where {arg,equals}, groupBy ← group_by.
 *
 * If board / JS / SQL disagree on any case, that is a real compiler bug — the
 * assertion stays sharp. avg is OUT OF SCOPE (fail-visibly) — it folds through
 * IEEE div whose float result does not round-trip through atomKey; we assert the
 * fail-visible instead.
 *
 * (Sandbox: run via the compiled route — tsx fails here. See CLAUDE.md.)
 */
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { atomKey } from '../kernel/predicate.js'
import type { RuleDefinition } from '../kernel/predicate.js'
import type { SemanticScalar } from '../model/types.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { applyWorkingMemoryOperations, type WorkingMemoryOperation } from './working-memory.js'
import { getLogicContext } from './logic-context.js'
import {
  compileBoard,
  CompileBoardError,
  type AggregateSpec,
  type Fact,
} from './compile-board.js'
import { compileBoardToSql } from './compile-board-sql.js'

// ── the derive_aggregate op that an AggregateSpec mirrors ─────────────────────────
type AggOp = Extract<WorkingMemoryOperation, { op: 'derive_aggregate' }>

function specToOp(spec: AggregateSpec, id: string): AggOp {
  const op: AggOp = {
    op: 'derive_aggregate',
    id,
    kind: spec.kind ?? 'sum',
    source: { predicate: spec.source, valueArg: spec.valueArg },
    into: { predicate: spec.target, valueArg: spec.outArg },
  }
  if (spec.where != null) op.where = { arg: spec.where.arg, equals: spec.where.equals as string | number | boolean }
  if (spec.groupBy != null) op.group_by = spec.groupBy
  return op
}

// ── (1) BOARD path ───────────────────────────────────────────────────────────────
function boardAggregateSet(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  aggregates: AggregateSpec[],
): Set<string> {
  const store = new MemorySpaceStore()
  const { id: spaceId } = store.createSpace({ title: 'aggregate triple-agreement' })
  // Stage 1: assert base facts + rules, so the closure materialises any DERIVED
  // source predicate before the aggregate op runs. (Since the same-batch deferral,
  // facts + rules + aggregate may also be sent in ONE batch — the aggregate expands
  // after the batch's first closure. The two-stage split here is still valid and
  // keeps this triple-agreement harness simple; it is no longer a requirement.)
  const setupOps: WorkingMemoryOperation[] = [
    ...baseFacts.map(
      (f, i) =>
        ({ op: 'assert_fact', id: `base_${i}`, predicate: f.predicate, args: f.args } as WorkingMemoryOperation),
    ),
    ...rules.map(
      (r) => ({ op: 'add_axiom', id: r.id, label: r.id, when: r.when, then: r.then } as WorkingMemoryOperation),
    ),
  ]
  applyWorkingMemoryOperations(store, spaceId, setupOps)
  // Stage 2: the aggregate ops, now that all sources are materialised.
  applyWorkingMemoryOperations(
    store,
    spaceId,
    aggregates.map((spec, i) => specToOp(spec, `agg_${i}_${spec.target}`)),
  )
  const ctx = getLogicContext(store, spaceId)
  const targets = new Set(aggregates.map((a) => a.target))
  const out = new Set<string>()
  for (const fact of ctx.facts) {
    if (fact.derived && targets.has(fact.atom.predicate)) {
      out.add(atomKey({ predicate: fact.atom.predicate, args: fact.atom.args }))
    }
  }
  return out
}

// ── (2) JS pipeline path ───────────────────────────────────────────────────────
function jsAggregateSet(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  aggregates: AggregateSpec[],
): Set<string> {
  const compiled = compileBoard(rules, { aggregates })
  const targets = new Set(aggregates.map((a) => a.target))
  const out = new Set<string>()
  for (const f of compiled.pipeline(baseFacts)) {
    if (targets.has(f.predicate)) out.add(atomKey({ predicate: f.predicate, args: f.args }))
  }
  return out
}

// ── (3) SQL backend in node:sqlite ────────────────────────────────────────────────
type SqliteModule = typeof import('node:sqlite')

function detectSqlite(): SqliteModule | undefined {
  try {
    const req = createRequire(import.meta.url)
    const sqlite = req('node:sqlite') as SqliteModule
    if (sqlite?.DatabaseSync) return sqlite
  } catch {
    // unavailable
  }
  return undefined
}

function normalizeForSqlite(value: SemanticScalar | undefined): string | number | null {
  if (value === undefined) return null
  if (typeof value === 'boolean') return value ? 1 : 0
  return value
}

function sqlAggregateSet(
  sqlite: SqliteModule,
  rules: RuleDefinition[],
  baseFacts: Fact[],
  aggregates: AggregateSpec[],
): Set<string> {
  const compiled = compileBoardToSql(rules, { aggregates })
  const db = new sqlite.DatabaseSync(':memory:')
  try {
    for (const t of compiled.baseTables) {
      const cols = t.columns.length > 0 ? t.columns.map((c) => `"${c}"`).join(', ') : '"_"'
      db.exec(`CREATE TABLE "${t.predicate}" (${cols});`)
    }
    for (const t of compiled.baseTables) {
      const rows = baseFacts.filter((f) => f.predicate === t.predicate)
      if (rows.length === 0) continue
      const colNames = t.columns
      if (colNames.length === 0) {
        // Zero discovered columns (e.g. COUNT(*) over a base predicate whose args
        // are never referenced): the table is just a "_" placeholder column. We
        // still need ROW PRESENCE so COUNT(*) counts them — insert a default row
        // per fact. A real host loading facts does the same (row existence is the
        // datum).
        const stmt = db.prepare(`INSERT INTO "${t.predicate}" DEFAULT VALUES`)
        for (let i = 0; i < rows.length; i++) stmt.run()
        continue
      }
      const placeholders = colNames.map(() => '?').join(', ')
      const stmt = db.prepare(
        `INSERT INTO "${t.predicate}" (${colNames.map((c) => `"${c}"`).join(', ')}) VALUES (${placeholders})`,
      )
      for (const row of rows) {
        stmt.run(...colNames.map((c) => normalizeForSqlite(row.args[c])))
      }
    }
    db.exec(compiled.viewsSql)
    const targets = new Set(aggregates.map((a) => a.target))
    const out = new Set<string>()
    for (const v of compiled.views) {
      if (!targets.has(v.predicate)) continue
      const stmt = db.prepare(`SELECT * FROM "${v.predicate}"`)
      for (const row of stmt.all() as Record<string, unknown>[]) {
        const args: Record<string, SemanticScalar> = {}
        for (const col of v.columns) {
          const value = row[col]
          if (value === null || value === undefined) continue
          args[col] = value as SemanticScalar
        }
        out.add(atomKey({ predicate: v.predicate, args }))
      }
    }
    return out
  } finally {
    db.close()
  }
}

const SQLITE = detectSqlite()

// ── triple-agreement helper for aggregates ────────────────────────────────────────
function assertAggregateAgreement(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  aggregates: AggregateSpec[],
  label: string,
): Set<string> {
  const provenance =
    `\n  label: ${label}` +
    `\n  rules: ${JSON.stringify(rules)}` +
    `\n  baseFacts: ${JSON.stringify(baseFacts)}` +
    `\n  aggregates: ${JSON.stringify(aggregates)}`

  const board = boardAggregateSet(rules, baseFacts, aggregates)
  const js = jsAggregateSet(rules, baseFacts, aggregates)
  assert.deepEqual(
    [...js].sort(),
    [...board].sort(),
    `AGGREGATE-AGREEMENT FAILURE (board != JS reduce) — real compiler bug.${provenance}`,
  )
  if (SQLITE) {
    const sql = sqlAggregateSet(SQLITE, rules, baseFacts, aggregates)
    assert.deepEqual(
      [...sql].sort(),
      [...board].sort(),
      `AGGREGATE-AGREEMENT FAILURE (board != SQL GROUP BY) — real compiler bug.${provenance}`,
    )
  }
  return board
}

// ════════════════════════════════════════════════════════════════════════════════
// SETUP: an order/customer domain. No rules needed for the pure-aggregate cases,
// but we also exercise an aggregate over a DERIVED predicate (rules feeding agg).
// ════════════════════════════════════════════════════════════════════════════════

const ORDERS: Fact[] = [
  { predicate: 'order', args: { id: 'O1', customer: 'A', region: 'east', amount: 100 } },
  { predicate: 'order', args: { id: 'O2', customer: 'A', region: 'west', amount: 250 } },
  { predicate: 'order', args: { id: 'O3', customer: 'B', region: 'east', amount: 40 } },
  { predicate: 'order', args: { id: 'O4', customer: 'B', region: 'east', amount: 60 } },
  { predicate: 'order', args: { id: 'O5', customer: 'C', region: 'west', amount: 0 } },
]

test('SQL leg availability (reported, never fails the suite)', () => {
  console.log(
    SQLITE
      ? '[agg-agreement] node:sqlite AVAILABLE — SQL GROUP BY leg WILL run (full triple agreement).'
      : '[agg-agreement] node:sqlite UNAVAILABLE — SQL leg skipped; board==JS only.',
  )
})

test('plain SUM over all orders — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [{ target: 'grand_total', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'sum' }],
    'plain sum',
  )
  // 100+250+40+60+0 = 450
  assert.deepEqual([...set], [atomKey({ predicate: 'grand_total', args: { value: 450 } })])
})

test('COUNT of orders — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [{ target: 'order_count', outArg: 'n', source: 'order', kind: 'count' }],
    'count',
  )
  assert.deepEqual([...set], [atomKey({ predicate: 'order_count', args: { n: 5 } })])
})

test('SUM with group_by region — triple agreement (one fact per bucket)', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [
      {
        target: 'region_total',
        outArg: 'value',
        source: 'order',
        valueArg: 'amount',
        kind: 'sum',
        groupBy: 'region',
      },
    ],
    'sum group_by region',
  )
  // east: 100+40+60=200 ; west: 250+0=250
  assert.deepEqual(
    [...set].sort(),
    [
      atomKey({ predicate: 'region_total', args: { value: 200, region: 'east' } }),
      atomKey({ predicate: 'region_total', args: { value: 250, region: 'west' } }),
    ].sort(),
  )
})

test('SUM with where filter region=east — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [
      {
        target: 'east_total',
        outArg: 'value',
        source: 'order',
        valueArg: 'amount',
        kind: 'sum',
        where: { arg: 'region', equals: 'east' },
      },
    ],
    'sum where region=east',
  )
  // east: 100+40+60 = 200
  assert.deepEqual([...set], [atomKey({ predicate: 'east_total', args: { value: 200 } })])
})

test('COUNT with group_by customer — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [{ target: 'cust_orders', outArg: 'n', source: 'order', kind: 'count', groupBy: 'customer' }],
    'count group_by customer',
  )
  // A:2, B:2, C:1
  assert.deepEqual(
    [...set].sort(),
    [
      atomKey({ predicate: 'cust_orders', args: { n: 2, customer: 'A' } }),
      atomKey({ predicate: 'cust_orders', args: { n: 2, customer: 'B' } }),
      atomKey({ predicate: 'cust_orders', args: { n: 1, customer: 'C' } }),
    ].sort(),
  )
})

test('MIN and MAX over amounts — triple agreement', () => {
  const minSet = assertAggregateAgreement(
    [],
    ORDERS,
    [{ target: 'min_amount', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'min' }],
    'min',
  )
  assert.deepEqual([...minSet], [atomKey({ predicate: 'min_amount', args: { value: 0 } })])

  const maxSet = assertAggregateAgreement(
    [],
    ORDERS,
    [{ target: 'max_amount', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'max' }],
    'max',
  )
  assert.deepEqual([...maxSet], [atomKey({ predicate: 'max_amount', args: { value: 250 } })])
})

test('MIN/MAX with group_by region — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [
      {
        target: 'region_max',
        outArg: 'value',
        source: 'order',
        valueArg: 'amount',
        kind: 'max',
        groupBy: 'region',
      },
    ],
    'max group_by region',
  )
  // east max 100, west max 250
  assert.deepEqual(
    [...set].sort(),
    [
      atomKey({ predicate: 'region_max', args: { value: 100, region: 'east' } }),
      atomKey({ predicate: 'region_max', args: { value: 250, region: 'west' } }),
    ].sort(),
  )
})

test('aggregate over a DERIVED predicate (rules feed the aggregate) — triple agreement', () => {
  // Rule: big_order(id, amount) :- order(id, amount), gte(amount, 100).
  // Then SUM big_order.amount. The aggregate is scheduled AFTER big_order's layer.
  const rules: RuleDefinition[] = [
    {
      id: 'AX_BIG',
      when: [
        { predicate: 'order', args: { id: '?o', amount: '?a' } },
        { predicate: 'gte', args: { left: '?a', right: 100 } },
      ],
      then: [{ predicate: 'big_order', args: { id: '?o', amount: '?a' } }],
    },
  ]
  const set = assertAggregateAgreement(
    rules,
    ORDERS,
    [{ target: 'big_total', outArg: 'value', source: 'big_order', valueArg: 'amount', kind: 'sum' }],
    'sum over derived big_order',
  )
  // big_order: O1(100), O2(250) -> 350
  assert.deepEqual([...set], [atomKey({ predicate: 'big_total', args: { value: 350 } })])
})

test('two aggregates with distinct targets in one program — triple agreement', () => {
  const set = assertAggregateAgreement(
    [],
    ORDERS,
    [
      { target: 'grand_total', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'sum' },
      { target: 'order_count', outArg: 'n', source: 'order', kind: 'count' },
    ],
    'two aggregates',
  )
  assert.deepEqual(
    [...set].sort(),
    [
      atomKey({ predicate: 'grand_total', args: { value: 450 } }),
      atomKey({ predicate: 'order_count', args: { n: 5 } }),
    ].sort(),
  )
})

// ════════════════════════════════════════════════════════════════════════════════
// FAIL-VISIBLE BOUNDARIES (honest scope).
// ════════════════════════════════════════════════════════════════════════════════

test('avg is fail-visible in BOTH compile backends (out of exact-integer scope)', () => {
  const spec: AggregateSpec = {
    target: 'avg_amount',
    outArg: 'value',
    source: 'order',
    valueArg: 'amount',
    kind: 'avg' as AggregateSpec['kind'],
  }
  assert.throws(() => compileBoard([], { aggregates: [spec] }), CompileBoardError)
  assert.throws(() => compileBoardToSql([], { aggregates: [spec] }), CompileBoardError)
})

test('sum needs valueArg — fail-visible', () => {
  const spec = { target: 'bad', outArg: 'value', source: 'order', kind: 'sum' } as AggregateSpec
  assert.throws(() => compileBoard([], { aggregates: [spec] }), CompileBoardError)
  assert.throws(() => compileBoardToSql([], { aggregates: [spec] }), CompileBoardError)
})

test('aggregate target colliding with a rule head — fail-visible', () => {
  const rules: RuleDefinition[] = [
    { id: 'R', when: [{ predicate: 'order', args: { id: '?o' } }], then: [{ predicate: 'total', args: { id: '?o' } }] },
  ]
  const spec: AggregateSpec = { target: 'total', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'sum' }
  assert.throws(() => compileBoard(rules, { aggregates: [spec] }), CompileBoardError)
  assert.throws(() => compileBoardToSql(rules, { aggregates: [spec] }), CompileBoardError)
})

test('two aggregates targeting the same predicate — fail-visible', () => {
  const specs: AggregateSpec[] = [
    { target: 'dup', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'sum' },
    { target: 'dup', outArg: 'n', source: 'order', kind: 'count' },
  ]
  assert.throws(() => compileBoard([], { aggregates: specs }), CompileBoardError)
  assert.throws(() => compileBoardToSql([], { aggregates: specs }), CompileBoardError)
})

test('non-numeric value for sum — fail-visible at pipeline run', () => {
  const facts: Fact[] = [{ predicate: 'order', args: { id: 'O1', amount: 'oops' } }]
  const compiled = compileBoard([], {
    aggregates: [{ target: 't', outArg: 'value', source: 'order', valueArg: 'amount', kind: 'sum' }],
  })
  assert.throws(() => compiled.pipeline(facts), CompileBoardError)
})

// ════════════════════════════════════════════════════════════════════════════════
// EMITTED SQL — sanity that GROUP BY + HAVING guard are present (the artifact).
// ════════════════════════════════════════════════════════════════════════════════

test('emitted SQL for sum-with-group_by uses SUM(...) ... GROUP BY ... HAVING COUNT(*) > 0', () => {
  const compiled = compileBoardToSql([], {
    aggregates: [
      {
        target: 'region_total',
        outArg: 'value',
        source: 'order',
        valueArg: 'amount',
        kind: 'sum',
        groupBy: 'region',
      },
    ],
  })
  console.log('═══ emitted SQL (sum group_by region) ═══')
  console.log(compiled.sql)
  console.log('═════════════════════════════════════════')
  assert.match(compiled.viewsSql, /CREATE VIEW "region_total"/)
  assert.match(compiled.viewsSql, /SUM\("amount"\)/)
  assert.match(compiled.viewsSql, /GROUP BY "region"/)
  assert.match(compiled.viewsSql, /HAVING COUNT\(\*\) > 0/)
})
