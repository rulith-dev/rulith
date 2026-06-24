import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { createRequire } from 'node:module'
import type { RuleDefinition } from '../kernel/predicate.js'
import { compileBoardToSql, inferColumnTypes } from './compile-board-sql.js'
import type { Fact } from './compile-board.js'

describe('compileBoardToSql — postgres dialect (production-server shape)', () => {
  it('default dialect is unchanged: TYPELESS base tables and INSTR for contains', () => {
    const rules: RuleDefinition[] = [
      {
        id: 'R',
        when: [
          { predicate: 'doc', args: { body: '?b' } },
          { predicate: 'contains', args: { left: '?b', right: 'needle' } },
        ],
        then: [{ predicate: 'hit', args: { body: '?b' } }],
      },
    ]
    const std = compileBoardToSql(rules)
    assert.match(std.baseTablesSql, /CREATE TABLE "doc" \("body"\);/) // typeless
    assert.match(std.viewsSql, /INSTR\(/)
    assert.doesNotMatch(std.viewsSql, /STRPOS\(/)
  })

  it('postgres: TYPED base tables from columnTypes (numeric typed numeric, others text)', () => {
    const rules: RuleDefinition[] = [
      {
        id: 'R',
        when: [
          { predicate: 'order', args: { id: '?o', amount: '?a' } },
          { predicate: 'gte', args: { left: '?a', right: 100 } },
        ],
        then: [{ predicate: 'big', args: { id: '?o' } }],
      },
    ]
    const pg = compileBoardToSql(rules, {
      dialect: 'postgres',
      columnTypes: { order: { id: 'text', amount: 'numeric' } },
    })
    // columns are emitted sorted: amount, id
    assert.match(pg.baseTablesSql, /CREATE TABLE "order" \("amount" NUMERIC, "id" TEXT\);/)
  })

  it('postgres: a column absent from columnTypes defaults to TEXT', () => {
    const rules: RuleDefinition[] = [
      { id: 'R', when: [{ predicate: 'p', args: { a: '?a' } }], then: [{ predicate: 'q', args: { a: '?a' } }] },
    ]
    const pg = compileBoardToSql(rules, { dialect: 'postgres' }) // no columnTypes at all
    assert.match(pg.baseTablesSql, /CREATE TABLE "p" \("a" TEXT\);/)
  })

  it('contains compiles to STRPOS in postgres (vs INSTR in standard)', () => {
    const rules: RuleDefinition[] = [
      {
        id: 'R',
        when: [
          { predicate: 'doc', args: { body: '?b' } },
          { predicate: 'contains', args: { left: '?b', right: 'needle' } },
        ],
        then: [{ predicate: 'hit', args: { body: '?b' } }],
      },
    ]
    assert.match(compileBoardToSql(rules, { dialect: 'postgres' }).viewsSql, /STRPOS\(t0\."body", 'needle'\) > 0/)
    assert.doesNotMatch(compileBoardToSql(rules, { dialect: 'postgres' }).viewsSql, /INSTR\(/)
  })

  it('boolean constants: TRUE/FALSE in postgres, 0/1 in standard', () => {
    const rules: RuleDefinition[] = [
      {
        id: 'R',
        when: [
          { predicate: 'flag', args: { on: '?x' } },
          { predicate: 'eq', args: { left: '?x', right: true } },
        ],
        then: [{ predicate: 'lit', args: { on: '?x' } }],
      },
    ]
    assert.match(compileBoardToSql(rules, { dialect: 'postgres' }).viewsSql, /= TRUE\)/)
    assert.match(compileBoardToSql(rules).viewsSql, /= 1\)/)
  })

  describe('inferColumnTypes', () => {
    it('numeric only when every value is a number; mixed/empty → text; all-boolean → boolean', () => {
      const facts: Fact[] = [
        { predicate: 'order', args: { id: 'O1', amount: 250 } },
        { predicate: 'order', args: { id: 'O2', amount: 0 } },
        { predicate: 'flag', args: { on: true } },
        { predicate: 'mix', args: { v: 1 } },
        { predicate: 'mix', args: { v: 'x' } }, // same column, different types → text
      ]
      const t = inferColumnTypes(facts)
      assert.equal(t.order.amount, 'numeric')
      assert.equal(t.order.id, 'text')
      assert.equal(t.flag.on, 'boolean')
      assert.equal(t.mix.v, 'text')
    })
  })

  it('EXECUTION: the typed postgres schema + views run through node:sqlite and are correct', () => {
    // node:sqlite can execute the postgres-shape SQL whenever there is no STRPOS
    // (Postgres-only). A typed NUMERIC column must compare numerically, not as text.
    let sqlite: typeof import('node:sqlite')
    try {
      sqlite = createRequire(import.meta.url)('node:sqlite') as typeof import('node:sqlite')
      if (!sqlite?.DatabaseSync) return
    } catch {
      return
    }
    const rules: RuleDefinition[] = [
      {
        id: 'R1',
        when: [
          { predicate: 'order', args: { id: '?o', amount: '?a' } },
          { predicate: 'gte', args: { left: '?a', right: 100 } },
        ],
        then: [{ predicate: 'big_order', args: { id: '?o' } }],
      },
    ]
    const facts: Fact[] = [
      { predicate: 'order', args: { id: 'O1', amount: 150 } },
      { predicate: 'order', args: { id: 'O2', amount: 50 } },
    ]
    const pg = compileBoardToSql(rules, { dialect: 'postgres', columnTypes: inferColumnTypes(facts) })
    assert.match(pg.baseTablesSql, /"amount" NUMERIC/)

    const db = new sqlite.DatabaseSync(':memory:')
    db.exec(pg.baseTablesSql)
    const stmt = db.prepare('INSERT INTO "order" ("amount", "id") VALUES (?, ?)')
    stmt.run(150, 'O1')
    stmt.run(50, 'O2')
    db.exec(pg.viewsSql)
    // node:sqlite returns null-prototype row objects; project to plain objects.
    const rows = (db.prepare('SELECT * FROM "big_order"').all() as Record<string, unknown>[]).map((r) => ({
      id: r.id,
    }))
    db.close()
    // Only O1 (150 >= 100). If the column were text, '50' >= '100' would be true
    // lexically and O2 would wrongly appear — so this also proves the typing.
    assert.deepEqual(rows, [{ id: 'O1' }])
  })
})
