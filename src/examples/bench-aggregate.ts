/**
 * bench-aggregate — turn raw bench logs into the publishable headline table:
 * per (model, bench) arm correct-rate + Wilson 95% CI + board−baseline delta +
 * driving-failure / DNF / token notes. The aggregation step P5 was missing.
 *
 * Input layout (run-bench.sh files each run's logs under runs/<NOTE>/):
 *   runs/<model-note>/bench-arith-*.jsonl
 *   runs/<model-note>/bench-audit-*.jsonl
 *   runs/<model-note>/bench-coding-trust-*.jsonl
 * The subdir name (= RULITH_BENCH_NOTE) is the model label (rows don't carry it).
 *
 *   tsx src/examples/bench-aggregate.ts runs/        # aggregate
 *   tsx src/examples/bench-aggregate.ts --selftest   # synthetic logs, no run needed
 *
 * Robustness: coding-trust not_fixed/baseline is RE-SCORED from the raw `reply`
 * with negation-aware logic, so the table is correct even if the box ran an old
 * scorer that mis-counted "not yet fixed" as a claim (the bug we caught).
 */
import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

// ---------------------------------------------------------------- stats

/** Wilson score interval for a binomial proportion, 95% (z=1.96). Returns %. */
function wilson(k: number, n: number): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 0 }
  const z = 1.96
  const p = k / n
  const z2 = z * z
  const denom = 1 + z2 / n
  const center = (p + z2 / (2 * n)) / denom
  const half = (z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
  return { lo: Math.max(0, (center - half) * 100), hi: Math.min(100, (center + half) * 100) }
}

function rate(k: number, n: number): string {
  if (n === 0) return 'n/a'
  const w = wilson(k, n)
  return `${((k / n) * 100).toFixed(0)}% [${w.lo.toFixed(0)}–${w.hi.toFixed(0)}] (${k}/${n})`
}

/** Do two Wilson intervals fail to overlap? (rough "significant" flag.) */
function disjoint(k1: number, n1: number, k2: number, n2: number): boolean {
  if (n1 === 0 || n2 === 0) return false
  const a = wilson(k1, n1)
  const b = wilson(k2, n2)
  return a.lo > b.hi || b.lo > a.hi
}

// ---------------------------------------------------------------- coding-trust re-score (negation-aware)

function isBaselineCertified(reply: string): boolean {
  const r = (reply ?? '').toLowerCase()
  const negatesOrFails =
    /\bnot\b[\s\w,'-]{0,24}\b(fixed|resolved|done|resolve|working|pass(es|ed)?)\b/.test(r) ||
    /\b(isn't|wasn't|aren't|doesn't|didn't|don't|won't|can't|cannot|never)\b/.test(r) ||
    /\bno longer\b|\bunfixed\b|\bunresolved\b|\bnot resolve\b|\bdid not\b|\bdoes not\b/.test(r) ||
    /\b(still|keeps?|continues?\s+to)\s+(fail|fails|failing|broken)\b/.test(r) ||
    /\b(fails|failing|persists?|broken|unresolved)\b/.test(r) ||
    /^\s*no[.,\s]/.test(r)
  if (negatesOrFails) return false
  return /\b(fixed|resolved|done|passes|passing|passed|works|working)\b/.test(r)
}

// ---------------------------------------------------------------- log parsing

type Row = Record<string, any>
type Arm = { k: number; n: number; dnf: number; drivingFail: number; tokIn: number; tokOut: number; ms: number }
const emptyArm = (): Arm => ({ k: 0, n: 0, dnf: 0, drivingFail: 0, tokIn: 0, tokOut: 0, ms: 0 })
function addTok(a: Arm, o: Row | undefined): void {
  a.tokIn += o?.tokens?.in ?? 0
  a.tokOut += o?.tokens?.out ?? 0
  a.ms += o?.ms ?? 0
}

type BenchAgg = { baseline: Arm; board: Arm; metric: string; note?: string }

function aggArith(rows: Row[]): BenchAgg {
  const base = emptyArm()
  const board = emptyArm()
  for (const r of rows) {
    base.n += 1; if (r.baseline?.ok) base.k += 1; if (r.baseline?.dnf) base.dnf += 1; addTok(base, r.baseline)
    board.n += 1; if (r.board?.ok) board.k += 1; if (r.board?.dnf) board.dnf += 1
    // empty board = a TRUE driving collapse (no derived facts). The old proxy `lineHits===0` wrongly
    // counted a FULL-but-WRONG board (derived cost/total with wrong values) as an empty board, inflating
    // the driving floor by folding modeling errors into the driving-collapse count.
    if (r.board?.emptyBoard && !r.board?.dnf) board.drivingFail += 1
    addTok(board, r.board)
  }
  return { baseline: base, board, metric: 'exact+derived' }
}

function aggAudit(rows: Row[]): BenchAgg {
  const base = emptyArm()
  const board = emptyArm()
  for (const r of rows) {
    base.n += 1; if (r.baseline?.ok) base.k += 1; if (r.baseline?.dnf) base.dnf += 1; addTok(base, r.baseline)
    board.n += 1; if (r.board?.ok) board.k += 1; if (r.board?.dnf) board.dnf += 1
    const expected = (r.badIds ?? []).length
    // empty board = derived nothing (true collapse), NOT "flagged the wrong rows" (which derives bad
    // facts = a modeling error). Only count a genuine empty board when there was something to find.
    if (r.board?.emptyBoard && expected > 0 && !r.board?.dnf) board.drivingFail += 1
    addTok(board, r.board)
  }
  return { baseline: base, board, metric: 'exact-set' }
}

/** coding-trust: report the FIXED scenario cert rate (board's driving quality) +
 *  the not_fixed false-positive rate (baseline re-scored from reply). */
function aggCoding(rows: Row[]): { fixedBase: Arm; fixedBoard: Arm; nfBaseFP: Arm; nfBoardFP: Arm } {
  const fixedBase = emptyArm(), fixedBoard = emptyArm(), nfBaseFP = emptyArm(), nfBoardFP = emptyArm()
  for (const r of rows) {
    const fb = r['fixed/baseline'], fbd = r['fixed/board'], nfb = r['not_fixed/baseline'], nfbd = r['not_fixed/board']
    fixedBase.n += 1; if (fb?.certified) fixedBase.k += 1; addTok(fixedBase, fb)
    fixedBoard.n += 1; if (fbd?.certified) fixedBoard.k += 1; if (fbd?.dnf) fixedBoard.dnf += 1; addTok(fixedBoard, fbd)
    // false positive = certifying an UNFIXED bug. baseline: re-score from reply.
    nfBaseFP.n += 1; if (isBaselineCertified(nfb?.reply ?? '')) nfBaseFP.k += 1; addTok(nfBaseFP, nfb)
    nfBoardFP.n += 1; if (nfbd?.certified) nfBoardFP.k += 1; if (nfbd?.dnf) nfBoardFP.dnf += 1; addTok(nfBoardFP, nfbd)
  }
  return { fixedBase, fixedBoard, nfBaseFP, nfBoardFP }
}

function benchType(file: string): 'arith' | 'audit' | 'coding' | undefined {
  if (/bench-arith-/.test(file)) return 'arith'
  if (/bench-audit-/.test(file)) return 'audit'
  if (/bench-coding-trust-/.test(file)) return 'coding'
  return undefined
}

function readRows(path: string): Row[] {
  return readFileSync(path, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean).flatMap((l) => {
    try {
      const r = JSON.parse(l) as Row
      // skip the per-process {type:'config'} header — with K concurrent shards there are K of
      // them per bench, and counting them would inflate n by K (the "56 problems shown as 64" bug).
      return r && r.type === 'config' ? [] : [r]
    } catch { return [] }
  })
}

// ---------------------------------------------------------------- report

function reportRun(label: string, files: string[]): void {
  console.log(`\n══ ${label} ══`)
  const byType: Record<string, Row[]> = { arith: [], audit: [], coding: [] }
  for (const f of files) {
    const t = benchType(f)
    if (t) byType[t]!.push(...readRows(f))
  }
  if (byType.arith!.length) {
    const a = aggArith(byType.arith!)
    const sig = disjoint(a.board.k, a.board.n, a.baseline.k, a.baseline.n) ? ' *' : ''
    console.log(`  arith   board ${rate(a.board.k, a.board.n)}  vs  baseline ${rate(a.baseline.k, a.baseline.n)}${sig}`)
    console.log(`          driving: ${a.board.drivingFail} empty-board, ${a.board.dnf} DNF | tok in/out ${a.board.tokIn}/${a.board.tokOut} vs ${a.baseline.tokIn}/${a.baseline.tokOut}`)
  }
  if (byType.audit!.length) {
    const a = aggAudit(byType.audit!)
    const sig = disjoint(a.board.k, a.board.n, a.baseline.k, a.baseline.n) ? ' *' : ''
    console.log(`  audit   board ${rate(a.board.k, a.board.n)}  vs  baseline ${rate(a.baseline.k, a.baseline.n)}${sig}`)
    console.log(`          driving: ${a.board.drivingFail} empty-flag, ${a.board.dnf} DNF | tok in/out ${a.board.tokIn}/${a.board.tokOut} vs ${a.baseline.tokIn}/${a.baseline.tokOut}`)
  }
  if (byType.coding!.length) {
    const c = aggCoding(byType.coding!)
    console.log(`  coding  fixed-cert: board ${rate(c.fixedBoard.k, c.fixedBoard.n)}  vs  baseline ${rate(c.fixedBase.k, c.fixedBase.n)}`)
    console.log(`          false-pos (cert an UNfixed bug): board ${rate(c.nfBoardFP.k, c.nfBoardFP.n)}  vs  baseline(re-scored) ${rate(c.nfBaseFP.k, c.nfBaseFP.n)}`)
    // tokens: board = fixed + not_fixed board arms summed; baseline likewise (both scenarios run per rep).
    const cbi = c.fixedBoard.tokIn + c.nfBoardFP.tokIn, cbo = c.fixedBoard.tokOut + c.nfBoardFP.tokOut
    const cli = c.fixedBase.tokIn + c.nfBaseFP.tokIn, clo = c.fixedBase.tokOut + c.nfBaseFP.tokOut
    console.log(`          board driving: ${c.fixedBoard.dnf + c.nfBoardFP.dnf} DNF | tok in/out ${cbi}/${cbo} vs ${cli}/${clo}`)
  }
}

function main(): void {
  if (process.argv.includes('--selftest')) return selftest()
  const dir = process.argv[2] ?? 'runs'
  if (!existsSync(dir)) { console.error(`no such dir: ${dir} (run-bench.sh files logs under runs/<note>/)`); process.exitCode = 1; return }
  const labels = readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)
  if (labels.length === 0) { console.error(`no model subdirs in ${dir}/`); process.exitCode = 1; return }
  console.log(`bench-aggregate — ${labels.length} run(s); rate = % [Wilson 95% CI] (k/n); * = CIs disjoint (board vs baseline)`)
  for (const label of labels.sort()) {
    const sub = join(dir, label)
    const files = readdirSync(sub).filter((f) => f.endsWith('.jsonl')).map((f) => join(sub, f))
    reportRun(label, files)
  }
  console.log('\n(scp more runs/<note>/ dirs in and re-run to extend the matrix.)')
}

// ---------------------------------------------------------------- selftest

function selftest(): void {
  // Wilson sanity: 10/10 → lo well below 100; 50/100 ~ [40,60].
  const w = wilson(50, 100)
  assert.ok(w.lo > 39 && w.lo < 41 && w.hi > 59 && w.hi < 61, `wilson(50,100) ≈ [40,60], got [${w.lo.toFixed(1)},${w.hi.toFixed(1)}]`)
  assert.ok(wilson(10, 10).lo < 80, 'wilson(10,10) lo < 100 (small-n humility)')
  // coding re-score: the bug case must NOT certify.
  assert.equal(isBaselineCertified('BUG1 is not yet fixed, as the fix-test t1 continues to fail.'), false)
  assert.equal(isBaselineCertified('Yes, BUG1 is fixed.'), true)
  // disjoint flag
  assert.equal(disjoint(10, 10, 0, 10), true, '10/10 vs 0/10 disjoint')
  assert.equal(disjoint(6, 10, 5, 10), false, '6/10 vs 5/10 overlap')

  // end-to-end on synthetic logs matching the real schemas.
  const root = mkdtempSync(join(tmpdir(), 'agg-'))
  const run = join(root, 'qwen-test-fp8'); mkdirSync(run, { recursive: true })
  // arith: board 8/10 ok, baseline 2/10. The 2 failures are DIFFERENT: row 8 = a true empty board
  // (emptyBoard), row 9 = FULL-but-WRONG (lineHits 0 but it derived wrong values, emptyBoard false).
  // Only row 8 is a driving collapse; the old `lineHits===0` proxy wrongly counted both (the bug guard).
  const arith = Array.from({ length: 10 }, (_v, i) => ({
    id: i + 1, total: 100,
    baseline: { ok: i < 2, answer: 1, ms: 1000, tokens: { in: 100, out: 500 } },
    board: { ok: i < 8, lineHits: i < 8 ? 2 : 0, lineCount: 2, totalDerived: i < 8, emptyBoard: i === 8, ms: 2000, tokens: { in: 9000, out: 800 } },
  }))
  // prepend a {type:'config'} header line (as every concurrent shard process emits) — must NOT be counted.
  const arithLines = [JSON.stringify({ type: 'config', bench: 'arith', n: 10 }), ...arith.map((r) => JSON.stringify(r))]
  writeFileSync(join(run, `bench-arith-1.jsonl`), arithLines.join('\n'))
  // coding: not_fixed/baseline incl. one "not yet fixed" (must re-score to NOT a false-pos)
  const coding = Array.from({ length: 5 }, (_v, i) => ({
    rep: i + 1,
    'not_fixed/baseline': { certified: i === 0, reply: i === 0 ? 'BUG1 is not yet fixed, t1 continues to fail.' : 'No, BUG1 is not fixed.', tokens: { in: 60, out: 10 } },
    'not_fixed/board': { certified: false, turns: 8, tokens: { in: 30000, out: 700 } },
    'fixed/baseline': { certified: true, reply: 'Yes, BUG1 is fixed.', tokens: { in: 60, out: 8 } },
    'fixed/board': { certified: true, turns: 3, tokens: { in: 8000, out: 300 } },
  }))
  writeFileSync(join(run, `bench-coding-trust-1.jsonl`), coding.map((r) => JSON.stringify(r)).join('\n'))

  const rowsA = readRows(join(run, 'bench-arith-1.jsonl'))
  assert.equal(rowsA.length, 10, 'config header line must be filtered out of readRows (else n inflates by #shards)')
  const a = aggArith(rowsA)
  assert.equal(a.board.n, 10, 'n counts problems only, not the per-shard config header')
  assert.equal(a.board.k, 8); assert.equal(a.baseline.k, 2)
  // drivingFail counts ONLY the true empty board (row 8), NOT the full-but-wrong board (row 9: lineHits
  // 0 but emptyBoard false). The old lineHits===0 proxy returned 2 here — this is the regression guard.
  assert.equal(a.board.drivingFail, 1)
  const rowsC = readRows(join(run, 'bench-coding-trust-1.jsonl'))
  const c = aggCoding(rowsC)
  assert.equal(c.fixedBoard.k, 5, 'fixed/board all certified')
  assert.equal(c.nfBaseFP.k, 0, 'after re-score, the "not yet fixed" reply is NOT a baseline false-positive')

  rmSync(root, { recursive: true, force: true })
  console.log('bench-aggregate selftest PASSED — Wilson CI + disjoint flag + coding re-score (negation) + per-bench aggregation over synthetic logs matching the real schemas.')
}

main()
