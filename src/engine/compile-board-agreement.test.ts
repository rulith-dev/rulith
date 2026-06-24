/**
 * compile-board-agreement.test.ts — the strongest faithfulness artifact for the
 * two compile backends.
 *
 * THE PROPERTY (triple agreement)
 * -------------------------------
 * For many rule programs + base-fact sets, ALL THREE paths must agree on the
 * derived-fact SET (compared by atomKey):
 *   (1) the BOARD closure (ground truth) — MemorySpaceStore + applyWorkingMemory
 *       Operations(assert_fact base facts + add_axiom rules) + getLogicContext.
 *   (2) compileBoard(...).pipeline(baseFacts) — the JS pure-function backend.
 *   (3) compileBoardToSql(...) executed in node:sqlite — the SQL view backend.
 *
 * compile-board-agent.ts / compile-board-sql-agent.ts each proved this on the
 * SINGLE omni rule pack. This test GENERALIZES it: hand-picked packs (omni +
 * expense) with edge fact sets, SYNTHETIC structured shapes (diamond, NAF
 * anti-join, comparison guard, 3-layer chain), and a SEEDED bounded random FUZZ
 * over a tiny acyclic predicate alphabet.
 *
 * If board / JS / SQL ever DISAGREE on any case, that is a real compiler bug —
 * the assertion stays sharp (no weakening). Red is informative: it prints the
 * exact rules + facts (+ seed for the fuzz) so the failure reproduces.
 *
 * SQL leg: if node:sqlite is unavailable, we LOG a skip note and assert only
 * board==js (we never fail the suite for a missing engine; we DO assert the SQL
 * leg whenever the engine is present).
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
import { compileBoard, CompileBoardError, type Fact } from './compile-board.js'
import { compileBoardToSql } from './compile-board-sql.js'

// ── (1) BOARD closure path (ground truth) ───────────────────────────────────────

function boardDerivedSet(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  spinePredicates: string[],
): Set<string> {
  const store = new MemorySpaceStore()
  const { id: spaceId } = store.createSpace({ title: 'triple-agreement board path' })
  const ops: WorkingMemoryOperation[] = [
    ...baseFacts.map(
      (f, i) =>
        ({ op: 'assert_fact', id: `base_${i}`, predicate: f.predicate, args: f.args } as WorkingMemoryOperation),
    ),
    ...rules.map(
      (r) => ({ op: 'add_axiom', id: r.id, label: r.id, when: r.when, then: r.then } as WorkingMemoryOperation),
    ),
  ]
  applyWorkingMemoryOperations(store, spaceId, ops)
  const ctx = getLogicContext(store, spaceId)
  const want = new Set(spinePredicates)
  const out = new Set<string>()
  for (const fact of ctx.facts) {
    if (fact.derived && want.has(fact.atom.predicate)) {
      out.add(atomKey({ predicate: fact.atom.predicate, args: fact.atom.args }))
    }
  }
  return out
}

// ── (2) JS pipeline path ──────────────────────────────────────────────────────

function jsDerivedSet(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  spinePredicates: string[],
): Set<string> {
  const compiled = compileBoard(rules)
  const want = new Set(spinePredicates)
  const out = new Set<string>()
  for (const f of compiled.pipeline(baseFacts)) {
    if (want.has(f.predicate)) out.add(atomKey({ predicate: f.predicate, args: f.args }))
  }
  return out
}

// ── (3) SQL backend executed in node:sqlite ─────────────────────────────────────
//
// Reuses the EXACT execution approach of src/examples/compile-board-sql-agent.ts:
// detect node:sqlite via createRequire, CREATE TABLE per base predicate, INSERT
// the base facts, exec the view chain (already in dependency order), SELECT each
// spine view, collect tuples by atomKey.

type SqliteModule = typeof import('node:sqlite')

function detectSqlite(): SqliteModule | undefined {
  try {
    const req = createRequire(import.meta.url)
    const sqlite = req('node:sqlite') as SqliteModule
    if (sqlite?.DatabaseSync) return sqlite
  } catch {
    // node:sqlite not available on this runtime
  }
  return undefined
}

/** node:sqlite accepts string|number|bigint|null|Uint8Array. Map board scalars. */
function normalizeForSqlite(value: SemanticScalar | undefined): string | number | null {
  if (value === undefined) return null
  if (typeof value === 'boolean') return value ? 1 : 0
  return value
}

function sqlDerivedSet(
  sqlite: SqliteModule,
  rules: RuleDefinition[],
  baseFacts: Fact[],
  spinePredicates: string[],
): Set<string> {
  const compiled = compileBoardToSql(rules)
  const db = new sqlite.DatabaseSync(':memory:')
  try {
    // 1. Base tables.
    for (const t of compiled.baseTables) {
      const cols = t.columns.length > 0 ? t.columns.map((c) => `"${c}"`).join(', ') : '"_"'
      db.exec(`CREATE TABLE "${t.predicate}" (${cols});`)
    }
    // 2. Insert base facts into their tables.
    for (const t of compiled.baseTables) {
      const rows = baseFacts.filter((f) => f.predicate === t.predicate)
      if (rows.length === 0) continue
      const colNames = t.columns
      const placeholders = colNames.map(() => '?').join(', ')
      const stmt = db.prepare(
        `INSERT INTO "${t.predicate}" (${colNames.map((c) => `"${c}"`).join(', ')}) VALUES (${placeholders})`,
      )
      for (const row of rows) {
        const vals = colNames.map((c) => normalizeForSqlite(row.args[c]))
        stmt.run(...vals)
      }
    }
    // 3. Create the views (already in dependency order).
    db.exec(compiled.viewsSql)
    // 4. Query each spine predicate, collect tuples by atomKey.
    const want = new Set(spinePredicates)
    const out = new Set<string>()
    for (const v of compiled.views) {
      if (!want.has(v.predicate)) continue
      const stmt = db.prepare(`SELECT * FROM "${v.predicate}"`)
      for (const row of stmt.all() as Record<string, unknown>[]) {
        const args: Record<string, SemanticScalar> = {}
        for (const col of v.columns) {
          const value = row[col]
          if (value === null || value === undefined) continue // NULL = col not bound by this rule
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
const SQLITE_AVAILABLE = SQLITE !== undefined

// ── the reusable triple-agreement helper ─────────────────────────────────────────

/**
 * Run all three paths on the same (rules, baseFacts) and assert the derived-fact
 * SETS over spinePredicates are pairwise equal: board==js, and board==sql when an
 * engine is present. The failure message prints the exact rules + facts so any
 * disagreement reproduces. `label` and optional `seed` annotate the report.
 */
function assertTripleAgreement(
  rules: RuleDefinition[],
  baseFacts: Fact[],
  spinePredicates: string[],
  label = '(unnamed)',
  seed?: number,
): void {
  const provenance =
    `\n  label: ${label}` +
    (seed !== undefined ? `\n  seed: ${seed}` : '') +
    `\n  rules: ${JSON.stringify(rules)}` +
    `\n  baseFacts: ${JSON.stringify(baseFacts)}` +
    `\n  spine: ${JSON.stringify(spinePredicates)}`

  const board = boardDerivedSet(rules, baseFacts, spinePredicates)
  const js = jsDerivedSet(rules, baseFacts, spinePredicates)
  assert.deepEqual(
    [...js].sort(),
    [...board].sort(),
    `TRIPLE-AGREEMENT FAILURE (board != JS pipeline) — real compiler bug.${provenance}`,
  )

  if (SQLITE) {
    const sql = sqlDerivedSet(SQLITE, rules, baseFacts, spinePredicates)
    assert.deepEqual(
      [...sql].sort(),
      [...board].sort(),
      `TRIPLE-AGREEMENT FAILURE (board != SQL view backend) — real compiler bug.${provenance}`,
    )
  }
}

// Heads of a rule pack — the natural full spine (all derivable predicates).
function allHeads(rules: RuleDefinition[]): string[] {
  const s = new Set<string>()
  for (const r of rules) for (const h of r.then ?? []) s.add(h.predicate)
  return [...s].sort()
}

// ════════════════════════════════════════════════════════════════════════════════
// HAND-PICKED PACK 1: the omni order-processing spine (6 rules).
// ════════════════════════════════════════════════════════════════════════════════

const EXPRESS_THRESHOLD = 200

const OMNI_RULES: RuleDefinition[] = [
  {
    id: 'AX_HAS_CUSTOMER',
    when: [
      { predicate: 'order', args: { id: '?o', customer: '?c' } },
      { predicate: 'customer', args: { id: '?c' } },
    ],
    then: [{ predicate: 'has_customer', args: { order: '?o' } }],
  },
  {
    id: 'AX_BAD_CUSTOMER',
    when: [
      { predicate: 'order', args: { id: '?o', customer: '?c', amount: '?a' } },
      { predicate: 'has_customer', args: { order: '?o' }, naf: true },
    ],
    then: [{ predicate: 'invalid_order', args: { id: '?o', reason: 'missing_customer' } }],
  },
  {
    id: 'AX_BAD_AMOUNT',
    when: [
      { predicate: 'order', args: { id: '?o', amount: '?a' } },
      { predicate: 'lte', args: { left: '?a', right: 0 } },
    ],
    then: [{ predicate: 'invalid_order', args: { id: '?o', reason: 'nonpositive_amount' } }],
  },
  {
    id: 'AX_HAS_INVALID',
    when: [{ predicate: 'invalid_order', args: { id: '?o', reason: '?r' } }],
    then: [{ predicate: 'has_invalid', args: { order: '?o' } }],
  },
  {
    id: 'AX_VALID',
    when: [
      { predicate: 'order', args: { id: '?o', customer: '?c', amount: '?a' } },
      { predicate: 'has_invalid', args: { order: '?o' }, naf: true },
    ],
    then: [{ predicate: 'valid_order', args: { id: '?o' } }],
  },
  {
    id: 'AX_EXPRESS',
    when: [
      { predicate: 'valid_order', args: { id: '?o' } },
      { predicate: 'order', args: { id: '?o', amount: '?a' } },
      { predicate: 'gte', args: { left: '?a', right: EXPRESS_THRESHOLD } },
    ],
    then: [{ predicate: 'eligible_for_express', args: { id: '?o' } }],
  },
]

const OMNI_SPINE = [
  'has_customer',
  'invalid_order',
  'has_invalid',
  'valid_order',
  'eligible_for_express',
]

const OMNI_FACT_SETS: { name: string; facts: Fact[] }[] = [
  { name: 'zero orders', facts: [{ predicate: 'customer', args: { id: 'CUST-A' } }] },
  { name: 'empty board', facts: [] },
  {
    name: 'all valid (one express)',
    facts: [
      { predicate: 'order', args: { id: 'O1', customer: 'CUST-A', amount: 250 } },
      { predicate: 'order', args: { id: 'O2', customer: 'CUST-A', amount: 50 } },
      { predicate: 'customer', args: { id: 'CUST-A' } },
    ],
  },
  {
    name: 'all invalid (missing customer)',
    facts: [
      { predicate: 'order', args: { id: 'O1', customer: 'CUST-Z', amount: 250 } },
      { predicate: 'order', args: { id: 'O2', customer: 'CUST-Y', amount: 50 } },
    ],
  },
  {
    name: 'amount<=0 path + mixed',
    facts: [
      { predicate: 'order', args: { id: 'O-GOOD', customer: 'CUST-A', amount: 250 } },
      { predicate: 'order', args: { id: 'O-ZERO', customer: 'CUST-A', amount: 0 } },
      { predicate: 'order', args: { id: 'O-NEG', customer: 'CUST-A', amount: -5 } },
      { predicate: 'order', args: { id: 'O-NOCUST', customer: 'CUST-Z', amount: 99 } },
      { predicate: 'customer', args: { id: 'CUST-A' } },
    ],
  },
]

// ════════════════════════════════════════════════════════════════════════════════
// HAND-PICKED PACK 2: the expense-approval derivation spine (7 rules, 4 layers).
// ════════════════════════════════════════════════════════════════════════════════

const RECEIPT_FLOOR = 50
const AUTO_CEILING = 500

const EXPENSE_RULES: RuleDefinition[] = [
  {
    id: 'AX_OVER_LIMIT',
    when: [
      { predicate: 'expense', args: { id: '?e', amount: '?a', category: '?c' } },
      { predicate: 'policy', args: { category: '?c', limit: '?l' } },
      { predicate: 'gt', args: { left: '?a', right: '?l' } },
    ],
    then: [{ predicate: 'over_limit', args: { id: '?e' } }],
  },
  {
    id: 'AX_HAS_RECEIPT',
    when: [
      { predicate: 'expense', args: { id: '?e' } },
      { predicate: 'receipt', args: { expense: '?e' } },
    ],
    then: [{ predicate: 'has_receipt', args: { id: '?e' } }],
  },
  {
    id: 'AX_WITHIN_LIMIT',
    when: [
      { predicate: 'expense', args: { id: '?e' } },
      { predicate: 'over_limit', args: { id: '?e' }, naf: true },
    ],
    then: [{ predicate: 'within_limit', args: { id: '?e' } }],
  },
  {
    id: 'AX_NEEDS_RECEIPT',
    when: [
      { predicate: 'expense', args: { id: '?e', amount: '?a' } },
      { predicate: 'gte', args: { left: '?a', right: RECEIPT_FLOOR } },
    ],
    then: [{ predicate: 'needs_receipt', args: { id: '?e' } }],
  },
  {
    id: 'AX_MISSING_RECEIPT',
    when: [
      { predicate: 'needs_receipt', args: { id: '?e' } },
      { predicate: 'has_receipt', args: { id: '?e' }, naf: true },
    ],
    then: [{ predicate: 'missing_receipt', args: { id: '?e' } }],
  },
  {
    id: 'AX_COMPLIANT',
    when: [
      { predicate: 'within_limit', args: { id: '?e' } },
      { predicate: 'missing_receipt', args: { id: '?e' }, naf: true },
    ],
    then: [{ predicate: 'compliant', args: { id: '?e' } }],
  },
  {
    id: 'AX_AUTO_APPROVABLE',
    when: [
      { predicate: 'compliant', args: { id: '?e' } },
      { predicate: 'expense', args: { id: '?e', amount: '?a' } },
      { predicate: 'lt', args: { left: '?a', right: AUTO_CEILING } },
    ],
    then: [{ predicate: 'auto_approvable', args: { id: '?e' } }],
  },
]

const EXPENSE_SPINE = [
  'over_limit',
  'has_receipt',
  'within_limit',
  'needs_receipt',
  'missing_receipt',
  'compliant',
  'auto_approvable',
]

const EXPENSE_FACT_SETS: { name: string; facts: Fact[] }[] = [
  { name: 'no expenses', facts: [{ predicate: 'policy', args: { category: 'travel', limit: 800 } }] },
  {
    name: 'three distinct fates',
    facts: [
      { predicate: 'expense', args: { id: 'E-OK', employee: 'EMP-1', amount: 120, category: 'travel' } },
      { predicate: 'expense', args: { id: 'E-BIG', employee: 'EMP-2', amount: 900, category: 'travel' } },
      { predicate: 'expense', args: { id: 'E-NORC', employee: 'EMP-3', amount: 300, category: 'meals' } },
      { predicate: 'policy', args: { category: 'travel', limit: 800 } },
      { predicate: 'policy', args: { category: 'meals', limit: 400 } },
      { predicate: 'receipt', args: { expense: 'E-OK' } },
    ],
  },
  {
    name: 'all compliant, all small (auto-approve all)',
    facts: [
      { predicate: 'expense', args: { id: 'E-1', employee: 'X', amount: 30, category: 'travel' } },
      { predicate: 'expense', args: { id: 'E-2', employee: 'Y', amount: 49, category: 'meals' } },
      { predicate: 'policy', args: { category: 'travel', limit: 800 } },
      { predicate: 'policy', args: { category: 'meals', limit: 400 } },
    ],
  },
  {
    name: 'expense without a matching policy (no over_limit join)',
    facts: [
      { predicate: 'expense', args: { id: 'E-X', employee: 'Z', amount: 700, category: 'unknown' } },
      { predicate: 'receipt', args: { expense: 'E-X' } },
      { predicate: 'policy', args: { category: 'travel', limit: 800 } },
    ],
  },
]

// ════════════════════════════════════════════════════════════════════════════════
// SYNTHETIC STRUCTURED CASES.
// ════════════════════════════════════════════════════════════════════════════════

// (i) DIAMOND: d <- b, c ; b <- a ; c <- a. One join at the apex.
const DIAMOND_RULES: RuleDefinition[] = [
  { id: 'B', when: [{ predicate: 'a', args: { x: '?x' } }], then: [{ predicate: 'b', args: { x: '?x' } }] },
  { id: 'C', when: [{ predicate: 'a', args: { x: '?x' } }], then: [{ predicate: 'c', args: { x: '?x' } }] },
  {
    id: 'D',
    when: [
      { predicate: 'b', args: { x: '?x' } },
      { predicate: 'c', args: { x: '?x' } },
    ],
    then: [{ predicate: 'd', args: { x: '?x' } }],
  },
]

// (ii) NAF anti-join: valid <- item, NOT flagged.
const NAF_RULES: RuleDefinition[] = [
  {
    id: 'VALID',
    when: [
      { predicate: 'item', args: { id: '?i' } },
      { predicate: 'flagged', args: { id: '?i' }, naf: true },
    ],
    then: [{ predicate: 'valid', args: { id: '?i' } }],
  },
]

// (iii) comparison-builtin rule: hot <- reading, gte(t,100) ; cold <- reading, lt(t,0).
const CMP_RULES: RuleDefinition[] = [
  {
    id: 'HOT',
    when: [
      { predicate: 'reading', args: { sensor: '?s', t: '?t' } },
      { predicate: 'gte', args: { left: '?t', right: 100 } },
    ],
    then: [{ predicate: 'hot', args: { sensor: '?s' } }],
  },
  {
    id: 'COLD',
    when: [
      { predicate: 'reading', args: { sensor: '?s', t: '?t' } },
      { predicate: 'lt', args: { left: '?t', right: 0 } },
    ],
    then: [{ predicate: 'cold', args: { sensor: '?s' } }],
  },
]

// (iv) 3-layer chain: l1 <- base ; l2 <- l1 ; l3 <- l2.
const CHAIN_RULES: RuleDefinition[] = [
  { id: 'L1', when: [{ predicate: 'base', args: { x: '?x' } }], then: [{ predicate: 'l1', args: { x: '?x' } }] },
  { id: 'L2', when: [{ predicate: 'l1', args: { x: '?x' } }], then: [{ predicate: 'l2', args: { x: '?x' } }] },
  { id: 'L3', when: [{ predicate: 'l2', args: { x: '?x' } }], then: [{ predicate: 'l3', args: { x: '?x' } }] },
]

const SYNTHETIC_CASES: {
  name: string
  rules: RuleDefinition[]
  spine: string[]
  factSets: { name: string; facts: Fact[] }[]
}[] = [
  {
    name: 'diamond',
    rules: DIAMOND_RULES,
    spine: ['b', 'c', 'd'],
    factSets: [
      { name: 'empty', facts: [] },
      { name: 'two values', facts: [{ predicate: 'a', args: { x: 1 } }, { predicate: 'a', args: { x: 2 } }] },
    ],
  },
  {
    name: 'naf anti-join',
    rules: NAF_RULES,
    spine: ['valid'],
    factSets: [
      { name: 'empty', facts: [] },
      {
        name: 'naf does NOT trigger (no flags) — all valid',
        facts: [{ predicate: 'item', args: { id: 'A' } }, { predicate: 'item', args: { id: 'B' } }],
      },
      {
        name: 'naf triggers (B flagged) — only A valid',
        facts: [
          { predicate: 'item', args: { id: 'A' } },
          { predicate: 'item', args: { id: 'B' } },
          { predicate: 'flagged', args: { id: 'B' } },
        ],
      },
    ],
  },
  {
    name: 'comparison guard',
    rules: CMP_RULES,
    spine: ['hot', 'cold'],
    factSets: [
      { name: 'empty', facts: [] },
      {
        name: 'boundary values (100 hot, 0 not cold, -1 cold, 99 neither)',
        facts: [
          { predicate: 'reading', args: { sensor: 'S1', t: 100 } },
          { predicate: 'reading', args: { sensor: 'S2', t: 0 } },
          { predicate: 'reading', args: { sensor: 'S3', t: -1 } },
          { predicate: 'reading', args: { sensor: 'S4', t: 99 } },
        ],
      },
    ],
  },
  {
    name: '3-layer chain',
    rules: CHAIN_RULES,
    spine: ['l1', 'l2', 'l3'],
    factSets: [
      { name: 'empty', facts: [] },
      { name: 'two values', facts: [{ predicate: 'base', args: { x: 'p' } }, { predicate: 'base', args: { x: 'q' } }] },
    ],
  },
]

// ════════════════════════════════════════════════════════════════════════════════
// SEEDED BOUNDED RANDOM FUZZ.
// ════════════════════════════════════════════════════════════════════════════════

/** mulberry32 — a tiny deterministic PRNG. Same seed → same stream. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Generate one random ACYCLIC rule program over a tiny alphabet:
 *   - base predicates b0..b2 (read-only, never derived), each over arg key "x".
 *   - derived predicates p0..p4: rule for pK may only reference base preds and
 *     LOWER-indexed derived preds (p0..p(K-1)). Referencing strictly lower
 *     indices enforces acyclicity, so deriveStages never hits recursion.
 *   - each rule body: 1-3 literals; each positive literal over arg key "x" (so
 *     shared "?x" forms joins), optionally NAF; optionally a single comparison
 *     guard gte/lt over the bound ?x against a small integer.
 *   - the head of pK projects { x: '?x' }; a body MUST positively bind ?x or the
 *     rule would be unsafe — we always include >=1 positive literal binding ?x.
 * The top predicate p(K-1) (highest defined) is the goal/spine.
 *
 * Base facts: random tuples over b0..b2 with small integer "x".
 */
function generateProgram(rng: () => number): {
  rules: RuleDefinition[]
  baseFacts: Fact[]
  spine: string[]
} {
  const basePreds = ['b0', 'b1', 'b2']
  const numDerived = 2 + Math.floor(rng() * 3) // 2..4 derived predicates
  const derivedPreds = Array.from({ length: numDerived }, (_, i) => `p${i}`)

  const pick = <T>(arr: T[]): T => arr[Math.floor(rng() * arr.length)]

  const rules: RuleDefinition[] = []
  for (let k = 0; k < numDerived; k++) {
    const head = derivedPreds[k]
    const lowerPool = [...basePreds, ...derivedPreds.slice(0, k)] // only lower indices
    const nLiterals = 1 + Math.floor(rng() * 3) // 1..3 literals
    const when: RuleDefinition['when'] = []
    // Always one positive binder on ?x (range restriction + safety).
    when.push({ predicate: pick(lowerPool), args: { x: '?x' } })
    for (let l = 1; l < nLiterals; l++) {
      const naf = rng() < 0.35
      // NAF over a lower pred, joined on ?x.
      when.push({ predicate: pick(lowerPool), args: { x: '?x' }, ...(naf ? { naf: true } : {}) })
    }
    // Optional single comparison guard on ?x.
    if (rng() < 0.4) {
      const op = rng() < 0.5 ? 'gte' : 'lt'
      const bound = Math.floor(rng() * 5) // 0..4
      when.push({ predicate: op, args: { left: '?x', right: bound } })
    }
    rules.push({ id: `R_${head}`, when, then: [{ predicate: head, args: { x: '?x' } }] })
  }

  // Random base facts over b0..b2 with small integer x (0..4). A few rows each.
  const baseFacts: Fact[] = []
  const nFacts = Math.floor(rng() * 8) // 0..7 base facts (includes the empty-set case)
  for (let i = 0; i < nFacts; i++) {
    baseFacts.push({ predicate: pick(basePreds), args: { x: Math.floor(rng() * 5) } })
  }

  const spine = [derivedPreds[numDerived - 1]] // top predicate as the goal
  return { rules, baseFacts, spine }
}

// ════════════════════════════════════════════════════════════════════════════════
// THE TESTS.
// ════════════════════════════════════════════════════════════════════════════════

test('SQL leg availability (reported, never fails the suite)', () => {
  if (SQLITE_AVAILABLE) {
    console.log('[agreement] node:sqlite AVAILABLE — SQL backend leg WILL run (full triple agreement).')
  } else {
    console.log(
      '[agreement] node:sqlite UNAVAILABLE — SQL leg SKIPPED; asserting board==JS only ' +
        '(honest degrade, suite does not fail for a missing engine).',
    )
  }
})

test('hand-picked pack: omni order-processing spine — triple agreement on every fact set', () => {
  let nonVacuous = 0
  for (const { name, facts } of OMNI_FACT_SETS) {
    assertTripleAgreement(OMNI_RULES, facts, OMNI_SPINE, `omni / ${name}`)
    if (boardDerivedSet(OMNI_RULES, facts, OMNI_SPINE).size > 0) nonVacuous++
  }
  assert.ok(nonVacuous >= 3, `expected several non-vacuous omni fact sets, got ${nonVacuous}`)
})

test('hand-picked pack: expense-approval spine — triple agreement on every fact set', () => {
  let nonVacuous = 0
  for (const { name, facts } of EXPENSE_FACT_SETS) {
    assertTripleAgreement(EXPENSE_RULES, facts, EXPENSE_SPINE, `expense / ${name}`)
    if (boardDerivedSet(EXPENSE_RULES, facts, EXPENSE_SPINE).size > 0) nonVacuous++
  }
  assert.ok(nonVacuous >= 2, `expected several non-vacuous expense fact sets, got ${nonVacuous}`)
})

test('synthetic structured cases (diamond / NAF anti-join / comparison guard / 3-layer chain)', () => {
  for (const c of SYNTHETIC_CASES) {
    for (const { name, facts } of c.factSets) {
      assertTripleAgreement(c.rules, facts, c.spine, `synthetic ${c.name} / ${name}`)
    }
  }
  // Non-vacuous spot checks: the NAF case must actually differ between the two
  // trigger/no-trigger fact sets, and the diamond apex must join.
  const nafTrigger = SYNTHETIC_CASES.find((c) => c.name === 'naf anti-join')!
  const noFlag = boardDerivedSet(NAF_RULES, nafTrigger.factSets[1].facts, ['valid'])
  const withFlag = boardDerivedSet(NAF_RULES, nafTrigger.factSets[2].facts, ['valid'])
  assert.equal(noFlag.size, 2, 'NAF no-trigger: both items valid')
  assert.equal(withFlag.size, 1, 'NAF trigger: only the unflagged item valid')
})

test('seeded bounded random fuzz: ~30 acyclic programs — triple agreement on accepted ones', () => {
  const BASE_SEED = 0x5eed1234
  const N = 30
  let accepted = 0
  let rejected = 0
  let nonVacuous = 0

  for (let i = 0; i < N; i++) {
    const seed = (BASE_SEED + i * 0x9e3779b1) >>> 0
    const rng = mulberry32(seed)
    const { rules, baseFacts, spine } = generateProgram(rng)

    // Skip programs outside the compilable fragment (CompileBoardError is expected
    // for out-of-fragment shapes; the fuzz only asserts agreement on ACCEPTED ones).
    try {
      compileBoard(rules) // throws CompileBoardError if out of fragment
    } catch (err) {
      if (err instanceof CompileBoardError) {
        rejected++
        continue
      }
      throw err // any other error is a real bug — surface it with the seed.
    }

    assertTripleAgreement(rules, baseFacts, spine, `fuzz #${i}`, seed)
    accepted++
    if (boardDerivedSet(rules, baseFacts, spine).size > 0) nonVacuous++
  }

  console.log(
    `[fuzz] base seed 0x${BASE_SEED.toString(16)}: ${N} programs — ${accepted} accepted ` +
      `(${nonVacuous} non-vacuous), ${rejected} rejected as out-of-fragment.`,
  )
  // The generator stays inside the fragment by construction, so essentially all
  // should be accepted; require a healthy majority accepted AND some that actually
  // derive facts (else the fuzz is vacuous).
  assert.ok(accepted >= 20, `expected most fuzz programs accepted, got ${accepted}/${N}`)
  assert.ok(nonVacuous >= 1, `fuzz must exercise at least one non-vacuous derivation, got ${nonVacuous}`)
})
