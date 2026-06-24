import type { PredicateAtom, SemanticScalar } from '../model/types.js'
import type { RuleDefinition } from '../kernel/predicate.js'
import { isVariable } from '../kernel/predicate.js'
import { COMPARISON_BUILTINS, isBuiltinPredicate } from '../kernel/builtins.js'
import { deriveStageLayers } from './derive-stages.js'
import {
  CompileBoardError,
  assertCompilable,
  assertAggregateSchedule,
  layerSchedule,
  type Fact,
  type AggregateSpec,
} from './compile-board.js'

/**
 * compile-board-sql — a SECOND backend for the SAME board rule program.
 *
 * THE POINT
 * ---------
 * compile-board.ts proved a board rule program (stratified-NAF Datalog) compiles
 * to an ordinary pure-FUNCTION pipeline ordered by deriveStages. This file proves
 * the thesis is not tied to ONE target language: the very same program, the same
 * dependency layering, the same fragment boundaries, compiles to a **SQL VIEW
 * sequence**. The kernel's rule program is a declarative dataflow program; SQL is
 * a declarative dataflow language; so the translation is direct:
 *
 *   - Each BASE predicate (read but never derived) = an input TABLE. Its arg keys
 *     are the columns.
 *   - Each DERIVED predicate = a VIEW. Layers are emitted in deriveStages order,
 *     so every view only references tables + views already defined (no forward
 *     reference, mirroring the per-layer pipeline).
 *   - A positive body literal = a relational source in the FROM list; a shared
 *     variable across literals = an equality JOIN condition; a constant arg = a
 *     literal equality in WHERE.
 *   - A `naf` body literal = `WHERE NOT EXISTS (SELECT 1 FROM <pred> WHERE ...)`
 *     — the anti-join.
 *   - A comparison built-in (eq/neq/lt/lte/gt/gte/between/contains) = a WHERE
 *     condition on bound terms.
 *   - The rule HEAD = the SELECT projection (head arg key = output column;
 *     value = the bound source column or a constant literal).
 *   - A predicate derived by MULTIPLE rules = a UNION of per-rule SELECTs in the
 *     view.
 *
 * SAME FRAGMENT, SAME FAIL-VISIBLE BOUNDARIES
 * -------------------------------------------
 * We reuse compile-board's assertCompilable, so the unsupported constructs throw
 * the SAME CompileBoardError naming rule + construct: arithmetic/value-producing
 * built-ins, derive_aggregate recipes, recursion (a fixpoint is not a straight
 * view chain), strong negation in heads or negated body literals, naf-over-builtin.
 *
 * FAITHFULNESS PROOF
 * ------------------
 * examples/compile-board-sql-agent.ts loads base facts as INSERTs, runs the
 * generated views in a SQLite engine, and asserts the final per-spine-predicate
 * SELECT results EQUAL the board closure's derived facts (same predicate+args
 * tuples). If no SQLite is available it degrades honestly to a structural check.
 */

/**
 * Target SQL dialect.
 *  - 'standard' (default): double-quoted identifiers, TYPELESS base tables (SQLite's
 *    dynamic typing), `INSTR` for `contains`, integer 0/1 for booleans. This is the
 *    historical behaviour and what the node:sqlite roundtrip proves.
 *  - 'postgres': the production-server shape. TYPED base tables (a statically-typed
 *    DB demands a type per column — see columnTypes), `STRPOS` for `contains`, native
 *    TRUE/FALSE booleans. Everything else (NOT EXISTS, BETWEEN, GROUP BY/HAVING,
 *    UNION, joins, double-quoted identifiers) is already standard SQL and unchanged.
 */
export type SqlDialect = 'standard' | 'postgres'

/** The SQL type of a base-table column, for the typed (postgres) dialect. */
export type SqlColumnType = 'numeric' | 'text' | 'boolean'

export type CompileBoardSqlOptions = {
  /** Goal predicate to compile toward. Default: the whole program. */
  goalPredicate?: string
  /** Target dialect. Default 'standard' (byte-identical to the historical output). */
  dialect?: SqlDialect
  /**
   * For dialect:'postgres', per-predicate column SQL types. A statically-typed DB
   * needs every column typed; numeric columns MUST be typed numeric so comparison
   * guards (lt/gt/between) compare as numbers, not text. Columns absent here default
   * to 'text'. Build it from the base facts with `inferColumnTypes(facts)`. Ignored
   * for the 'standard' dialect (SQLite types dynamically).
   */
  columnTypes?: Record<string, Record<string, SqlColumnType>>
  /**
   * Aggregate specs — the SQL mirror of derive_aggregate. Each compiles to a
   * GROUP BY aggregate view scheduled after its source predicate's layer.
   */
  aggregates?: AggregateSpec[]
}

/** SQL type name per dialect for a typed column (postgres). */
const PG_TYPE_NAME: Record<SqlColumnType, string> = {
  numeric: 'NUMERIC',
  text: 'TEXT',
  boolean: 'BOOLEAN',
}

/**
 * Infer a per-predicate, per-column SQL type from a set of base facts: a column
 * whose every observed value is a number → 'numeric'; every value a boolean →
 * 'boolean'; otherwise 'text' (the safe default, including mixed/empty). This is
 * how a host turns the board's dynamically-typed facts into a typed schema for a
 * statically-typed production DB. Pass the result as `columnTypes`.
 */
export function inferColumnTypes(facts: Fact[]): Record<string, Record<string, SqlColumnType>> {
  const seen = new Map<string, Map<string, Set<string>>>()
  for (const f of facts) {
    const byCol = seen.get(f.predicate) ?? new Map<string, Set<string>>()
    for (const [key, value] of Object.entries(f.args ?? {})) {
      const types = byCol.get(key) ?? new Set<string>()
      types.add(typeof value)
      byCol.set(key, types)
    }
    seen.set(f.predicate, byCol)
  }
  const out: Record<string, Record<string, SqlColumnType>> = {}
  for (const [predicate, byCol] of seen) {
    out[predicate] = {}
    for (const [col, types] of byCol) {
      out[predicate][col] =
        types.size === 1 && types.has('number')
          ? 'numeric'
          : types.size === 1 && types.has('boolean')
            ? 'boolean'
            : 'text'
    }
  }
  return out
}

export type CompiledBoardSql = {
  /** The full SQL script: CREATE TABLE for each base predicate (with the columns
   *  we discovered), one CREATE VIEW per derived predicate in dependency order,
   *  then a SELECT per spine (derived) predicate. This is the "board → SQL"
   *  artifact. */
  sql: string
  /** The base-table DDL only (CREATE TABLE ...), for hosts that build their own
   *  inputs and only want the view chain. */
  baseTablesSql: string
  /** The view chain only (CREATE VIEW ... in dependency order). */
  viewsSql: string
  /** The final per-predicate SELECTs (one per derived/spine predicate). */
  selectsSql: string
  /** deriveStages note + the layer order actually compiled. */
  note: string
  /** Per-layer predicate grouping actually compiled, ascending. */
  layers: { layer: number; predicates: string[] }[]
  /** Discovered base predicates and their column sets (the input tables). */
  baseTables: { predicate: string; columns: string[] }[]
  /** Derived predicates and their column sets (the views), in dependency order. */
  views: { predicate: string; columns: string[]; layer: number }[]
}

export { CompileBoardError } from './compile-board.js'

// ── identifier / literal quoting (standard SQL, sqlite-compatible) ──────────────

/** Quote an identifier (table/view/column name). Standard double-quote, with
 *  embedded-quote doubling. */
function ident(name: string): string {
  return '"' + String(name).replace(/"/g, '""') + '"'
}

/** Render a SQL literal for a SemanticScalar constant. */
function literal(value: SemanticScalar, dialect: SqlDialect = 'standard'): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new CompileBoardError(
        `cannot emit non-finite numeric literal ${String(value)} as SQL`,
      )
    }
    return String(value)
  }
  if (typeof value === 'boolean') {
    // 'standard'/SQLite has no native boolean → integer 0/1 (rounds-trips against
    // numeric columns). 'postgres' has native booleans → TRUE/FALSE so a BOOLEAN
    // column comparison is well-typed.
    if (dialect === 'postgres') return value ? 'TRUE' : 'FALSE'
    return value ? '1' : '0'
  }
  return "'" + value.replace(/'/g, "''") + "'"
}

// ── column discovery: every arg key a predicate ever uses, in stable order ──────

/**
 * Collect, per predicate, the union of all arg keys seen anywhere it appears
 * (as a body literal source OR a head). A predicate's columns are this union,
 * sorted for determinism. Base tables and views share the same column model so
 * a view can SELECT a base table's columns by name.
 */
function collectColumns(rules: RuleDefinition[]): Map<string, Set<string>> {
  const cols = new Map<string, Set<string>>()
  const add = (atom: PredicateAtom): void => {
    if (isBuiltinPredicate(atom.predicate)) return
    const set = cols.get(atom.predicate) ?? new Set<string>()
    for (const key of Object.keys(atom.args ?? {})) set.add(key)
    cols.set(atom.predicate, set)
  }
  for (const rule of rules) {
    for (const l of rule.when ?? []) add(l)
    for (const h of rule.then ?? []) add(h)
  }
  return cols
}

const sortedCols = (s: Set<string> | undefined): string[] => [...(s ?? [])].sort()

// ── per-rule SELECT compilation ─────────────────────────────────────────────────

/** Order a rule body the same way the kernel/JS-backend matcher does: positive
 *  (binding) literals first, then guards/naf. */
function orderBody(body: PredicateAtom[]): PredicateAtom[] {
  const isBinder = (l: PredicateAtom): boolean =>
    l.naf !== true && !isBuiltinPredicate(l.predicate)
  const binders = body.filter(isBinder)
  const rest = body.filter((l) => !isBinder(l))
  return [...binders, ...rest]
}

type BoundVar = { table: string; alias: string; column: string }

/**
 * Compile ONE rule to a single SELECT statement. Returns the SQL text (no
 * trailing semicolon). Throws CompileBoardError via the term resolver if a head
 * or guard term references an unbound variable (would be a malformed view).
 */
function compileRuleSelect(rule: RuleDefinition, dialect: SqlDialect = 'standard'): string {
  const body = orderBody(rule.when ?? [])
  const positives = body.filter((l) => l.naf !== true && !isBuiltinPredicate(l.predicate))
  const guards = body.filter((l) => isBuiltinPredicate(l.predicate))
  const nafs = body.filter((l) => l.naf === true)

  // First positive literal binding each variable wins; later occurrences become
  // join-equality conditions. Constants on positives become WHERE equalities.
  const varBind = new Map<string, BoundVar>()
  const fromParts: string[] = []
  const whereParts: string[] = []

  positives.forEach((lit, i) => {
    const alias = `t${i}`
    fromParts.push(`${ident(lit.predicate)} AS ${alias}`)
    for (const [key, value] of Object.entries(lit.args ?? {})) {
      const colExpr = `${alias}.${ident(key)}`
      if (isVariable(value)) {
        const name = value.slice(1)
        const bound = varBind.get(name)
        if (bound === undefined) {
          varBind.set(name, { table: lit.predicate, alias, column: key })
        } else {
          whereParts.push(`${colExpr} = ${bound.alias}.${ident(bound.column)}`)
        }
      } else {
        whereParts.push(`${colExpr} = ${literal(value, dialect)}`)
      }
    }
  })

  /** Resolve a term to its SQL expression (bound column or constant literal). */
  const term = (value: SemanticScalar | undefined): string => {
    if (value === undefined) {
      throw new CompileBoardError(
        `rule "${rule.id}": a guard/head term is undefined — cannot emit SQL.`,
      )
    }
    if (isVariable(value)) {
      const name = value.slice(1)
      const bound = varBind.get(name)
      if (bound === undefined) {
        throw new CompileBoardError(
          `rule "${rule.id}": variable ?${name} is used in a guard or head but never ` +
            `bound by a positive body literal — unsafe rule, cannot emit SQL.`,
        )
      }
      return `${bound.alias}.${ident(bound.column)}`
    }
    return literal(value, dialect)
  }

  for (const g of guards) whereParts.push(formatGuard(g, term, rule.id, dialect))

  for (const n of nafs) {
    // Anti-join: NOT EXISTS (SELECT 1 FROM <pred> AS nX WHERE <bound-arg matches>).
    const nAlias = `n${nafs.indexOf(n)}`
    const conds: string[] = []
    for (const [key, value] of Object.entries(n.args ?? {})) {
      const colExpr = `${nAlias}.${ident(key)}`
      if (isVariable(value)) {
        const name = value.slice(1)
        const bound = varBind.get(name)
        if (bound === undefined) {
          // A naf literal whose variable is not bound elsewhere: this matches the
          // existence of ANY tuple with that column present — but with no binding
          // it is a free anti-join. The kernel requires naf vars to be bound for
          // safety; reject to fail visibly rather than emit a wrong view.
          throw new CompileBoardError(
            `rule "${rule.id}": naf literal "${n.predicate}" references unbound ` +
              `variable ?${name} — naf must be range-restricted (bound by a positive ` +
              `literal first). Cannot emit a faithful anti-join.`,
          )
        }
        conds.push(`${colExpr} = ${bound.alias}.${ident(bound.column)}`)
      } else {
        conds.push(`${colExpr} = ${literal(value, dialect)}`)
      }
    }
    const inner = `SELECT 1 FROM ${ident(n.predicate)} AS ${nAlias}` +
      (conds.length > 0 ? ` WHERE ${conds.join(' AND ')}` : '')
    whereParts.push(`NOT EXISTS (${inner})`)
  }

  // Projection: one column per head arg. (A rule may have multiple heads; each
  // head predicate gets its own view, so we project a head at a time — here we
  // build the select for whichever head this call targets. To keep one SELECT
  // per (rule, head) we pass the head separately; see compilePredicateView.)
  // This function returns only the FROM/WHERE skeleton; projection is attached
  // by the caller which knows the target head.
  const fromClause = fromParts.length > 0 ? `\n  FROM ${fromParts.join(',\n       ')}` : ''
  const whereClause = whereParts.length > 0 ? `\n  WHERE ${whereParts.join('\n    AND ')}` : ''
  // Stash the resolver-bound projection-building closure on the returned object
  // would be awkward; instead the caller re-runs projection. To avoid duplicate
  // work we return a sentinel the caller never uses directly.
  return JSON.stringify({ fromClause, whereClause, varBind: serializeBind(varBind) })
}

function serializeBind(m: Map<string, BoundVar>): Record<string, BoundVar> {
  const o: Record<string, BoundVar> = {}
  for (const [k, v] of m) o[k] = v
  return o
}

/** Build the SELECT projection list for one head against a rule's bindings. */
function projectHead(
  head: PredicateAtom,
  columns: string[],
  varBind: Record<string, BoundVar>,
  ruleId: string,
  dialect: SqlDialect = 'standard',
): string {
  const headArgs = head.args ?? {}
  const parts = columns.map((col) => {
    const value = headArgs[col]
    if (value === undefined) {
      // The head does not bind this column. The view's column set is the union
      // across all rules; emit NULL so every UNION arm is shape-compatible.
      return `NULL AS ${ident(col)}`
    }
    if (isVariable(value)) {
      const name = value.slice(1)
      const bound = varBind[name]
      if (bound === undefined) {
        throw new CompileBoardError(
          `rule "${ruleId}": head variable ?${name} is not bound by any positive body ` +
            `literal — unsafe rule, cannot project SQL column.`,
        )
      }
      return `${bound.alias}.${ident(bound.column)} AS ${ident(col)}`
    }
    return `${literal(value, dialect)} AS ${ident(col)}`
  })
  return parts.join(', ')
}

/** Format a comparison built-in as a SQL WHERE fragment. */
function formatGuard(
  atom: PredicateAtom,
  term: (v: SemanticScalar | undefined) => string,
  ruleId: string,
  dialect: SqlDialect = 'standard',
): string {
  const a = atom.args ?? {}
  switch (atom.predicate) {
    case 'eq':
      return `(${term(a.left)} = ${term(a.right)})`
    case 'neq':
      return `(${term(a.left)} <> ${term(a.right)})`
    case 'lt':
      return `(${term(a.left)} < ${term(a.right)})`
    case 'lte':
      return `(${term(a.left)} <= ${term(a.right)})`
    case 'gt':
      return `(${term(a.left)} > ${term(a.right)})`
    case 'gte':
      return `(${term(a.left)} >= ${term(a.right)})`
    case 'between':
      return `(${term(a.value)} BETWEEN ${term(a.low)} AND ${term(a.high)})`
    case 'contains':
      // substring membership: left contains right → position of needle in haystack
      // > 0. SQLite/MySQL spell it INSTR(haystack, needle); Postgres spells it
      // STRPOS(haystack, needle). The ONE token that differs across the dialects.
      return dialect === 'postgres'
        ? `(STRPOS(${term(a.left)}, ${term(a.right)}) > 0)`
        : `(INSTR(${term(a.left)}, ${term(a.right)}) > 0)`
    default:
      throw new CompileBoardError(
        `rule "${ruleId}": comparison built-in "${atom.predicate}" has no SQL mapping.`,
      )
  }
}

/**
 * Build the CREATE VIEW statement for ONE derived predicate: UNION of one SELECT
 * per rule whose head is that predicate.
 */
function compilePredicateView(
  predicate: string,
  columns: string[],
  rules: RuleDefinition[],
  dialect: SqlDialect = 'standard',
): string {
  const arms: string[] = []
  for (const rule of rules) {
    for (const head of rule.then ?? []) {
      if (head.predicate !== predicate) continue
      const skeleton = JSON.parse(compileRuleSelect(rule, dialect)) as {
        fromClause: string
        whereClause: string
        varBind: Record<string, BoundVar>
      }
      const projection = projectHead(head, columns, skeleton.varBind, rule.id, dialect)
      arms.push(`  -- rule ${rule.id}\n  SELECT ${projection}${skeleton.fromClause}${skeleton.whereClause}`)
    }
  }
  // Multiple arms unioned. DISTINCT keeps view membership a SET (the closure is a
  // set; SQL bag semantics would over-count duplicate derivations).
  const body = arms.join('\n  UNION\n')
  const colList = columns.map(ident).join(', ')
  return `CREATE VIEW ${ident(predicate)} (${colList}) AS\n${body};`
}

/** SQL aggregate function name for a kind. */
const SQL_AGG: Record<string, string> = { sum: 'SUM', count: 'COUNT', min: 'MIN', max: 'MAX' }

/**
 * Compile ONE aggregate spec to a CREATE VIEW with a GROUP BY (or a single
 * implicit group). Mirrors computeAggregate's fold:
 *   sum  -> SUM(value)        count -> COUNT(*)
 *   min  -> MIN(value)        max   -> MAX(value)
 * where  -> WHERE arg = lit ; groupBy -> GROUP BY key (key carried into SELECT).
 * The HAVING COUNT(*) > 0 guard suppresses the all-NULL row a bare aggregate over
 * an empty source would otherwise produce, matching computeAggregate's "empty
 * pool → no fact".
 */
function compileAggregateView(spec: AggregateSpec, columns: string[], dialect: SqlDialect = 'standard'): string {
  const kind = spec.kind ?? 'sum'
  const fn = SQL_AGG[kind]
  const valueExpr = kind === 'count' ? '*' : ident(spec.valueArg!)
  const whereClause = spec.where != null ? `\n  WHERE ${ident(spec.where.arg)} = ${literal(spec.where.equals, dialect)}` : ''
  const groupClause = spec.groupBy != null ? `\n  GROUP BY ${ident(spec.groupBy)}` : ''
  // Projection: one column per the view's column set (outArg + optional groupBy);
  // any other discovered column (unlikely) projects NULL for UNION-shape parity.
  const selectParts = columns.map((col) => {
    if (col === spec.outArg) return `${fn}(${valueExpr}) AS ${ident(col)}`
    if (spec.groupBy != null && col === spec.groupBy) return `${ident(col)} AS ${ident(col)}`
    return `NULL AS ${ident(col)}`
  })
  const colList = columns.map(ident).join(', ')
  return (
    `CREATE VIEW ${ident(spec.target)} (${colList}) AS\n` +
    `  SELECT ${selectParts.join(', ')}\n` +
    `  FROM ${ident(spec.source)}${whereClause}${groupClause}\n` +
    `  HAVING COUNT(*) > 0;`
  )
}

/**
 * Compile the rule program to a SQL VIEW sequence ordered by deriveStages.
 * Mirrors compileBoard: same fragment, same fail-visible boundaries (reuses
 * assertCompilable), same layer schedule (reuses layerSchedule).
 */
export function compileBoardToSql(
  rules: RuleDefinition[],
  options: CompileBoardSqlOptions = {},
): CompiledBoardSql {
  assertCompilable(rules)
  const aggregates = options.aggregates ?? []
  assertAggregateSchedule(rules, aggregates)
  const dialect: SqlDialect = options.dialect ?? 'standard'
  const columnTypes = options.columnTypes ?? {}

  const { layers, rulesByLayer, aggregatesByLayer, heads } = layerSchedule(
    rules,
    options.goalPredicate,
    aggregates,
  )
  const { note } = deriveStageLayers(
    options.goalPredicate === undefined
      ? [
          ...rules,
          {
            id: '__compile_sink__',
            when: [...heads].map((p) => ({ predicate: p, args: {} })),
            then: [{ predicate: '__compile_all__', args: {} }],
          },
        ]
      : rules,
    options.goalPredicate ?? '__compile_all__',
  )

  const columns = collectColumns(rules)
  const aggTargets = new Set(aggregates.map((a) => a.target))
  // Aggregate targets are views with columns {outArg} (+ groupBy). Aggregate
  // sources/where/groupBy args must exist on the source's column set so the base
  // TABLE (when the source is a base predicate) carries them.
  for (const spec of aggregates) {
    const tcols = columns.get(spec.target) ?? new Set<string>()
    tcols.add(spec.outArg)
    if (spec.groupBy != null) tcols.add(spec.groupBy)
    columns.set(spec.target, tcols)
    const scols = columns.get(spec.source) ?? new Set<string>()
    if (spec.valueArg != null) scols.add(spec.valueArg)
    if (spec.where != null) scols.add(spec.where.arg)
    if (spec.groupBy != null) scols.add(spec.groupBy)
    columns.set(spec.source, scols)
  }

  // Base predicates = predicates read in some body but never a rule head, and
  // not built-ins; PLUS aggregate sources that are not themselves derived.
  // These become input TABLES.
  const baseTables: { predicate: string; columns: string[] }[] = []
  const basePreds = new Set<string>()
  for (const rule of rules) {
    for (const l of rule.when ?? []) {
      if (!heads.has(l.predicate) && !isBuiltinPredicate(l.predicate)) basePreds.add(l.predicate)
    }
  }
  for (const spec of aggregates) {
    if (!heads.has(spec.source)) basePreds.add(spec.source)
  }
  for (const p of [...basePreds].sort()) {
    baseTables.push({ predicate: p, columns: sortedCols(columns.get(p)) })
  }

  // Views = derived predicates (relational heads + aggregate targets), in
  // dependency-layer order.
  const views: { predicate: string; columns: string[]; layer: number }[] = []
  for (const g of layers) {
    for (const p of g.predicates) {
      views.push({ predicate: p, columns: sortedCols(columns.get(p)), layer: g.layer })
    }
  }

  // ── render ──
  const baseLines: string[] = []
  baseLines.push('-- Base input tables (predicates read but never derived).')
  baseLines.push('-- Arg keys become columns; a host inserts the board base facts here.')
  for (const t of baseTables) {
    // 'standard'/SQLite: typeless columns (dynamic typing). 'postgres': a type per
    // column (from columnTypes, default TEXT) — required by a statically-typed DB.
    const colDef = (c: string): string =>
      dialect === 'postgres'
        ? `${ident(c)} ${PG_TYPE_NAME[columnTypes[t.predicate]?.[c] ?? 'text']}`
        : ident(c)
    const cols =
      t.columns.length > 0
        ? t.columns.map(colDef).join(', ')
        : dialect === 'postgres'
          ? `${ident('_')} TEXT`
          : ident('_')
    baseLines.push(`CREATE TABLE ${ident(t.predicate)} (${cols});`)
  }
  const baseTablesSql = baseLines.join('\n')

  const viewLines: string[] = []
  viewLines.push('-- Derived predicates as views, one per dependency layer (deriveStages order).')
  viewLines.push('-- positive literal = join; naf = NOT EXISTS anti-join; comparison = WHERE; head = SELECT.')
  const aggByTarget = new Map(aggregates.map((a) => [a.target, a]))
  for (const g of layers) {
    viewLines.push(`-- ── layer ${g.layer}: ${g.predicates.join(', ')} ──`)
    for (const p of g.predicates) {
      const cols = sortedCols(columns.get(p))
      if (aggTargets.has(p)) {
        viewLines.push(compileAggregateView(aggByTarget.get(p)!, cols, dialect))
      } else {
        viewLines.push(compilePredicateView(p, cols, rulesByLayer.get(g.layer) ?? [], dialect))
      }
    }
  }
  const viewsSql = viewLines.join('\n')

  const selectLines: string[] = []
  selectLines.push('-- Final read: one SELECT per derived (spine) predicate.')
  for (const v of views) {
    selectLines.push(`SELECT * FROM ${ident(v.predicate)};`)
  }
  const selectsSql = selectLines.join('\n')

  const sql = [baseTablesSql, '', viewsSql, '', selectsSql].join('\n')

  return { sql, baseTablesSql, viewsSql, selectsSql, note, layers, baseTables, views }
}
