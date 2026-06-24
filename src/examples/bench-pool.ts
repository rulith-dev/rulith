/**
 * bench-pool — unified work-queue runner for the three P5 benches.
 *
 * Problem with static sharding (run-parallel.mjs): each shard owns a fixed
 * id-range, so when fast shards finish their slice they sit IDLE while a few
 * stragglers (hard problems, long board runs) drag on; and the three benches
 * run one-after-another. Concurrency collapses at every tail and the GPU coasts.
 *
 * Fix: put ALL tasks (3 benches × N) into ONE queue and run K workers that pull
 * the next task on completion. No tail (a freed worker immediately grabs the next
 * task, including from the NEXT bench) → the server stays pinned at K concurrent
 * requests until the very end. Each worker uses its OWN LlmClient(s) so the
 * per-arm consumeUsage() token windows never interleave across concurrent tasks.
 *
 * Usage:
 *   tsx src/examples/bench-pool.ts --selftest          # scripted, no LLM
 *   RULITH_BENCH_N=50 RULITH_BENCH_CONCURRENCY=25 \
 *   RULITH_BENCH_NOTE=<label> tsx src/examples/bench-pool.ts
 * Per-bench JSONL logs land under logs/ exactly like the standalone benches,
 * so bench-aggregate.ts reads them unchanged.
 */
import { strict as assert } from 'node:assert'
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { ChatModel } from '../agent/task-loop.js'
import type { LlmUsage } from '../agent/llm.js'
import type { BenchArm } from './bench-arms.js'
import { resolveArms } from './bench-arms.js'
import { LlmClient, modelBConfigFromEnv } from '../agent/llm.js'
import {
  mulberry32,
  generateProblem as genArithProblem,
  runArithProblemRow,
  emptyArithTally,
  type ArithTally,
} from './bench-arith.js'
import {
  generateProblem as genAuditProblem,
  runAuditProblemRow,
  emptyAuditTally,
  type AuditTally,
} from './bench-audit.js'
import { runCodingRepRow, newCodingTally } from './bench-coding-trust.js'

// A model the benches can drive: a ChatModel, optionally with token accounting.
// LlmClient satisfies it (has consumeUsage); scripted test models omit it.
export type BenchClient = ChatModel & { consumeUsage?: () => LlmUsage }
export type ClientFactory = (arm: BenchArm) => BenchClient

/**
 * Bounded dynamic work-queue: `limit` runners each pull the next item until the
 * queue drains. Concurrency never exceeds `limit`; every item runs exactly once;
 * fast runners naturally take more items (no fixed assignment = no tail idle).
 */
export async function mapPool<T>(
  items: readonly T[],
  limit: number,
  worker: (item: T, index: number) => Promise<void>,
): Promise<void> {
  const n = items.length
  if (n === 0) return
  let next = 0
  const lim = Math.max(1, Math.min(Math.floor(limit) || 1, n))
  const runner = async (): Promise<void> => {
    while (true) {
      const i = next
      next += 1 // single-threaded JS: read+increment is atomic, no double-take
      if (i >= n) return
      await worker(items[i]!, i)
    }
  }
  await Promise.all(Array.from({ length: lim }, () => runner()))
}

// ---------------------------------------------------------------- orchestrator

type BenchName = 'arith' | 'audit' | 'coding'
interface RowTask {
  bench: BenchName
  produce: () => Promise<Record<string, unknown>>
}

/** Run every task through ONE shared bounded pool; route each finished row to its bench sink. */
async function runPool(
  tasks: readonly RowTask[],
  conc: number,
  sink: Record<BenchName, (row: Record<string, unknown>) => void>,
): Promise<void> {
  await mapPool(tasks, conc, async (t) => {
    try {
      const row = await t.produce()
      sink[t.bench](row)
    } catch (e) {
      // A throwing task must NEVER kill its worker — that would permanently shrink
      // the pool (Running drifts down mid-run). Record a DNF row and keep pulling.
      sink[t.bench]({ error: String(e).slice(0, 200), dnf: true })
    }
  })
}

async function orchestrate(): Promise<void> {
  const N = Number(process.env.RULITH_BENCH_N ?? 50)
  const seed = Number(process.env.RULITH_BENCH_SEED ?? 7)
  const conc = Number(process.env.RULITH_BENCH_CONCURRENCY ?? 25)
  const maxTurns = Number(process.env.RULITH_BENCH_TURNS ?? 10)
  const note = process.env.RULITH_BENCH_NOTE
  const bConfig = modelBConfigFromEnv()
  const arms = resolveArms(process.env.RULITH_BENCH_ARM, bConfig !== undefined)
  const ORDER: readonly BenchArm[] = ['baseline', 'baseline_b', 'board', 'board_b']
  const active = ORDER.filter((a) => arms.has(a))
  // Which benches to include. Think cells should pass RULITH_BENCH_ONLY=arith,audit:
  // coding-trust always runs its board arm (ignores RULITH_BENCH_ARM), and board+think
  // accumulates a long prompt that + max_tokens overflows max-model-len → 400s. Skip it.
  const only = new Set(
    (process.env.RULITH_BENCH_ONLY ?? 'arith,audit,coding').split(',').map((s) => s.trim()).filter(Boolean),
  )
  const want = (b: BenchName): boolean => only.has(b)

  // Per-task FRESH clients: concurrent tasks must not share a consumeUsage() window.
  const makeClientAB: ClientFactory = (arm) => (arm.endsWith('_b') ? new LlmClient(bConfig) : new LlmClient())
  const makeClientA = (): BenchClient => new LlmClient()

  const arithTally = new Map<BenchArm, ArithTally>(active.map((a) => [a, emptyArithTally()]))
  const auditTally = new Map<BenchArm, AuditTally>(active.map((a) => [a, emptyAuditTally()]))
  const codingTally = newCodingTally()

  // run-pool sets RULITH_BENCH_LOGDIR=runs/<note> so concurrent runs (cloud + local) never
  // share a dir → no log mixing. Standalone use defaults to logs/.
  const logDir = process.env.RULITH_BENCH_LOGDIR ?? 'logs'
  mkdirSync(logDir, { recursive: true })
  const ts = Date.now()
  const openLog = (bench: string): ((entry: unknown) => void) => {
    const path = join(logDir, `bench-${bench}-${ts}.jsonl`)
    const write = (entry: unknown): void => appendFileSync(path, `${JSON.stringify(entry)}\n`)
    write({
      type: 'config',
      bench,
      runner: 'pool',
      startedAt: new Date().toISOString(),
      model: process.env.RULITH_LLM_MODEL ?? '(client default)',
      note,
      seed,
      n: N,
      concurrency: conc,
      arms: [...active].join(','),
    })
    return write
  }
  const noop = (): void => {}
  const writeArith = want('arith') ? openLog('arith') : noop
  const writeAudit = want('audit') ? openLog('audit') : noop
  const writeCoding = want('coding') ? openLog('coding-trust') : noop

  // Same seed → same problems as the standalone benches (comparable across cells).
  // Each bench has its own rng, so skipping one doesn't shift another's problem ids.
  const arithRng = mulberry32(seed)
  const auditRng = mulberry32(seed)
  const arithTasks: RowTask[] = !want('arith')
    ? []
    : Array.from({ length: N }, (_v, i) => genArithProblem(arithRng, i + 1)).map((p) => ({
        bench: 'arith' as BenchName,
        produce: () => runArithProblemRow(p, { active, maxTurns, tally: arithTally, makeClient: makeClientAB }),
      }))
  const auditTasks: RowTask[] = !want('audit')
    ? []
    : Array.from({ length: N }, (_v, i) => genAuditProblem(auditRng, i + 1)).map((p) => ({
        bench: 'audit' as BenchName,
        produce: () => runAuditProblemRow(p, { active, maxTurns, tally: auditTally, makeClient: makeClientAB }),
      }))
  const codingTasks: RowTask[] = !want('coding')
    ? []
    : Array.from({ length: N }, (_v, i) => ({
        bench: 'coding' as BenchName,
        produce: () => runCodingRepRow(i + 1, { maxTurns, tally: codingTally, makeClient: makeClientA }),
      }))

  // Round-robin interleave so long board runs spread evenly → smooth, no clustered tail.
  const tasks: RowTask[] = []
  for (let i = 0; i < N; i += 1) {
    if (arithTasks[i]) tasks.push(arithTasks[i]!)
    if (auditTasks[i]) tasks.push(auditTasks[i]!)
    if (codingTasks[i]) tasks.push(codingTasks[i]!)
  }

  const sink: Record<BenchName, (row: Record<string, unknown>) => void> = {
    arith: (row) => { writeArith(row); console.log(JSON.stringify({ b: 'arith', ...row })) },
    audit: (row) => { writeAudit(row); console.log(JSON.stringify({ b: 'audit', ...row })) },
    coding: (row) => { writeCoding(row); console.log(JSON.stringify({ b: 'coding', ...row })) },
  }

  console.error(`bench-pool: ${tasks.length} tasks [${[...only].join('+')}]×${N} across ${conc} workers → note=${note ?? '(none)'}`)
  const t0 = Date.now()
  await runPool(tasks, conc, sink)
  console.log(`\ndone in ${Math.round((Date.now() - t0) / 1000)}s  →  ${logDir}/bench-{arith,audit,coding-trust}-${ts}.jsonl`)
  console.log(`aggregate:  npx tsx src/examples/bench-aggregate.ts runs/`)
}

// ---------------------------------------------------------------- selftest

async function selftest(): Promise<void> {
  // (1) every item processed exactly once
  {
    const items = Array.from({ length: 150 }, (_v, i) => i)
    const seen: number[] = []
    await mapPool(items, 25, async (x) => {
      await new Promise((r) => setTimeout(r, Math.random() * 3))
      seen.push(x)
    })
    seen.sort((a, b) => a - b)
    assert.equal(seen.length, 150, 'all 150 processed')
    assert.deepEqual(seen, items, 'each item exactly once, none missed/duplicated')
  }

  // (2) concurrency never exceeds the limit; (3) it actually reaches the limit
  {
    const items = Array.from({ length: 60 }, (_v, i) => i)
    let inFlight = 0
    let maxInFlight = 0
    await mapPool(items, 8, async () => {
      inFlight += 1
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((r) => setTimeout(r, 2))
      inFlight -= 1
    })
    assert.ok(maxInFlight <= 8, `concurrency capped at 8, saw ${maxInFlight}`)
    assert.equal(maxInFlight, 8, `pool should saturate to 8 (n=60), saw ${maxInFlight}`)
  }

  // (4) dynamic pull: with a fast/slow mix the queue still covers every item and
  //     stays saturated (a freed worker immediately grabs the next — no tail).
  {
    const items = Array.from({ length: 100 }, (_v, i) => i)
    let inFlight = 0
    let maxInFlight = 0
    let done = 0
    await mapPool(items, 4, async (x) => {
      inFlight += 1
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((r) => setTimeout(r, x % 5 === 0 ? 6 : 1))
      inFlight -= 1
      done += 1
    })
    assert.equal(done, 100, 'dynamic run still covers every item')
    assert.equal(maxInFlight, 4, `fast/slow mix still saturates the pool, saw ${maxInFlight}`)
  }

  // (5) orchestrator plumbing: 3 benches × N in ONE pool, routed to per-bench sinks,
  //     and genuinely interleaved (cross-bench fill, not one bench at a time).
  {
    const N = 20
    const order: BenchName[] = []
    const counts: Record<BenchName, number> = { arith: 0, audit: 0, coding: 0 }
    const tasks: RowTask[] = []
    for (let i = 0; i < N; i += 1) {
      for (const b of ['arith', 'audit', 'coding'] as BenchName[]) {
        tasks.push({
          bench: b,
          produce: async () => {
            await new Promise((r) => setTimeout(r, Math.random() * 2))
            return { id: i + 1 }
          },
        })
      }
    }
    const sink: Record<BenchName, (row: Record<string, unknown>) => void> = {
      arith: () => { counts.arith += 1; order.push('arith') },
      audit: () => { counts.audit += 1; order.push('audit') },
      coding: () => { counts.coding += 1; order.push('coding') },
    }
    await runPool(tasks, 8, sink)
    assert.equal(counts.arith, N, 'every arith task routed')
    assert.equal(counts.audit, N, 'every audit task routed')
    assert.equal(counts.coding, N, 'every coding task routed')
    assert.equal(order.length, 3 * N, 'all 3×N tasks processed exactly once')
    assert.equal(
      new Set(order.slice(0, 24)).size,
      3,
      'all three benches run concurrently in the shared pool (cross-bench fill, not sequential)',
    )
  }

  // (6) resilience: a throwing task must NOT shrink the pool — workers stay saturated,
  //     every task still routed (throwers become DNF rows). Guards the "Running drifts down" failure.
  {
    const M = 60
    let inFlight = 0
    let maxInFlight = 0
    let routed = 0
    const tasks: RowTask[] = Array.from({ length: M }, (_v, i) => ({
      bench: 'arith' as BenchName,
      produce: async () => {
        inFlight += 1
        maxInFlight = Math.max(maxInFlight, inFlight)
        await new Promise((r) => setTimeout(r, 1))
        inFlight -= 1
        if (i % 3 === 0) throw new Error('boom') // 1/3 throw
        return { id: i }
      },
    }))
    const sink: Record<BenchName, (row: Record<string, unknown>) => void> = {
      arith: () => { routed += 1 },
      audit: () => {},
      coding: () => {},
    }
    await runPool(tasks, 8, sink)
    assert.equal(routed, M, 'every task routed despite 1/3 throwing (throwers → DNF rows, not lost)')
    assert.equal(maxInFlight, 8, `pool stays saturated despite throwing tasks, saw ${maxInFlight}`)
  }

  console.log(
    'bench-pool selftest PASSED — mapPool: all-once + cap + saturation + dynamic pull; orchestrator: routes + interleaves + throw-resilient.',
  )
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  if (process.argv.includes('--selftest')) {
    selftest().catch((e) => {
      console.error(e)
      process.exitCode = 1
    })
  } else {
    orchestrate().catch((e) => {
      console.error(e)
      process.exitCode = 1
    })
  }
}
