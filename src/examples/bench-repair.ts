/**
 * bench-repair — A/B benchmark for the board's value in REPAIR, not just
 * in finding. The open question (#30): the logged evidence so far shows
 * the board makes a local model's *findings* trustworthy (audits, exact
 * arithmetic, diagnosis). It does NOT yet show the board makes its
 * *fixes* more correct. This fixture measures exactly that.
 *
 * Each problem is a small self-contained function carrying a SEEDED bug
 * (off-by-one, wrong operator, inverted boundary, ...). A hidden test
 * suite (input -> reference output) is ground truth. The SAME local model
 * repairs it two ways:
 *
 *   baseline  free-form: "here is the buggy code and one failing case,
 *             return the corrected function." The model patches directly.
 *   board     board-tracked diagnosis FIRST: assert observation(input,
 *             expected, got) facts for the failing cases, add a rule that
 *             DERIVES diagnosis(kind) from them, read it back, THEN return
 *             the fix. The repair must be preceded by a derived diagnosis.
 *
 * Scoring is from the hidden suite, not from prose: a repair counts only
 * if EVERY hidden test passes on the model's returned function. The board
 * arm additionally reports whether the derived diagnosis kind matched the
 * seeded bug kind — so we can separate "fixed it" from "understood it".
 *
 * The claim under test: does forcing a structured, board-derived
 * diagnosis before the edit raise fix-correctness vs. free-form patching?
 *
 * Usage:
 *   tsx src/examples/bench-repair.ts --selftest    # scripted model, no LLM
 *   tsx src/examples/bench-repair.ts               # all configured arms, real model
 *   RULITH_BENCH_ARM=both|all|<comma list>         # arm selection, see bench-arms.ts
 *   RULITH_BENCH_N / RULITH_BENCH_SEED             # problem count / seed
 *   RULITH_BENCH_TRANSCRIPT=1                       # keep raw replies
 *   RULITH_LLM_MODEL_B                             # model B: enables the cross-model
 *     arms (+ RULITH_LLM_BASE_URL_B / RULITH_LLM_API_KEY_B for a remote endpoint);
 *     default "both" then runs the three-way comparison A+board vs B bare vs A bare
 */
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { getLogicContext } from '../engine/logic-context.js'
import { runAgentTask, type ChatModel } from '../agent/task-loop.js'
import { ToolRegistry } from '../agent/tools.js'
import { LlmClient, modelBConfigFromEnv, parseToolCall, type ChatMessage } from '../agent/llm.js'
import { addUsage, emptyUsageTally, fmtUsage, resolveArms, usageRow, type BenchArm } from './bench-arms.js'

// ---------------------------------------------------------------- problems

type TestCase = { args: number[]; expected: number }
type Problem = {
  id: number
  name: string
  signature: string
  /** Buggy source: a function body the model must repair. */
  buggySource: string
  /** Ground-truth correct function (for generating expectations + selftest). */
  reference: (...args: number[]) => number
  /** Seeded bug category, ground truth for diagnosis scoring. */
  bugKind: string
  tests: TestCase[]
}

/** The seeded-bug library. Each entry is a small, dependency-free function
 *  with exactly one planted defect and a deterministic reference. */
type ProblemSpec = Omit<Problem, 'id' | 'tests'> & { inputs: number[][] }

const LIBRARY: ProblemSpec[] = [
  {
    name: 'clamp',
    signature: 'clamp(x, lo, hi)',
    // bug: uses < instead of <= on the upper bound is fine; real bug: returns lo on the hi branch
    buggySource: `function clamp(x, lo, hi){ if (x < lo) return lo; if (x > hi) return lo; return x; }`,
    reference: (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x),
    bugKind: 'wrong_return_value',
    inputs: [[5, 0, 10], [-3, 0, 10], [15, 0, 10], [10, 0, 10], [0, 0, 10]],
  },
  {
    name: 'isLeap',
    signature: 'isLeap(year) -> 1 or 0',
    // bug: missing the 400 exception (div by 100 not-leap, but 400 IS leap)
    buggySource: `function isLeap(y){ if (y % 4 !== 0) return 0; if (y % 100 === 0) return 0; return 1; }`,
    reference: (y) => (y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0) ? 1 : 0),
    bugKind: 'missing_case',
    inputs: [[2000], [1900], [2024], [2023], [2400]],
  },
  {
    name: 'sumTo',
    signature: 'sumTo(n) = 1+2+...+n',
    // bug: off-by-one, loop stops at n-1
    buggySource: `function sumTo(n){ let s=0; for(let i=1;i<n;i++){ s+=i; } return s; }`,
    reference: (n) => (n * (n + 1)) / 2,
    bugKind: 'off_by_one',
    inputs: [[1], [5], [10], [0], [3]],
  },
  {
    name: 'abs',
    signature: 'abs(x)',
    // bug: inverted condition
    buggySource: `function abs(x){ if (x > 0) return -x; return x; }`,
    reference: (x) => Math.abs(x),
    bugKind: 'inverted_condition',
    inputs: [[5], [-5], [0], [-1], [100]],
  },
  {
    name: 'max3',
    signature: 'max3(a, b, c)',
    // bug: wrong comparison, returns b when c is largest
    buggySource: `function max3(a,b,c){ let m=a; if(b>m)m=b; if(c>b)m=c; return m; }`,
    reference: (a, b, c) => Math.max(a, b, c),
    bugKind: 'wrong_operand',
    inputs: [[5, 1, 3], [1, 2, 3], [3, 2, 1], [9, 1, 2], [2, 1, 3]],
  },
  {
    name: 'countDigits',
    signature: 'countDigits(n) for n>=0',
    // bug: returns 0 for n=0 (should be 1)
    buggySource: `function countDigits(n){ let c=0; while(n>0){ c++; n=Math.floor(n/10); } return c; }`,
    reference: (n) => String(Math.abs(n)).length,
    bugKind: 'boundary_zero',
    inputs: [[0], [5], [99], [100], [12345]],
  },
  {
    name: 'gcd',
    signature: 'gcd(a, b)',
    // bug: standard Euclid but returns the wrong variable (b is 0 at exit)
    buggySource: `function gcd(a,b){ while(b!==0){ const t=b; b=a%b; a=t; } return b; }`,
    reference: (a, b) => {
      while (b !== 0) {
        const t = b
        b = a % b
        a = t
      }
      return a
    },
    bugKind: 'wrong_return_value',
    inputs: [[12, 8], [48, 36], [7, 13], [100, 10], [17, 5]],
  },
  {
    name: 'fib',
    signature: 'fib(n), fib(0)=0, fib(1)=1',
    // bug: wrong seed, starts a=1,b=1 so off by one position
    buggySource: `function fib(n){ let a=1,b=1; for(let i=0;i<n;i++){ const t=a+b; a=b; b=t; } return a; }`,
    reference: (n) => {
      let a = 0
      let b = 1
      for (let i = 0; i < n; i += 1) {
        const t = a + b
        a = b
        b = t
      }
      return a
    },
    bugKind: 'wrong_initialization',
    inputs: [[0], [1], [2], [7], [10]],
  },
]

function buildProblem(spec: ProblemSpec, id: number): Problem {
  const tests = spec.inputs.map((args) => ({ args, expected: spec.reference(...args) }))
  return { id, name: spec.name, signature: spec.signature, buggySource: spec.buggySource, reference: spec.reference, bugKind: spec.bugKind, tests }
}

// ---------------------------------------------------------------- scoring

/** Compile a returned function source and run it; undefined on any failure.
 *  Trusted callers only (our own seeded library) - no loop guard. */
function evalFunction(source: string, name: string): ((...a: number[]) => number) | undefined {
  try {
    // eslint-disable-next-line no-new-func
    const factory = new Function(`${source}; return typeof ${name} === 'function' ? ${name} : undefined;`)
    const fn = factory() as ((...a: number[]) => number) | undefined
    return typeof fn === 'function' ? fn : undefined
  } catch {
    return undefined
  }
}

/** Run MODEL-RETURNED source in a child process with a hard timeout, so an
 *  infinite loop in a bad repair fails the case instead of hanging the
 *  whole bench (a buggy fix is exactly the kind of code that loops). */
function repairPasses(source: string, p: Problem, timeoutMs = 3000): boolean {
  const harness =
    `${source}\n` +
    `const cases = ${JSON.stringify(p.tests)};\n` +
    `for (const c of cases) {\n` +
    `  let got;\n` +
    `  try { got = ${p.name}(...c.args); } catch { process.exit(2); }\n` +
    `  if (got !== c.expected) process.exit(3);\n` +
    `}\n` +
    `process.exit(0);\n`
  const res = spawnSync(process.execPath, ['-e', harness], { timeout: timeoutMs })
  return res.status === 0 && res.signal === null
}

/** Extract a function source from a model reply (fenced block or bare decl). */
export function parseRepair(reply: string, name: string): string | undefined {
  const fenced = [...reply.matchAll(/```(?:js|javascript|ts|typescript)?\s*([\s\S]*?)```/g)]
    .map((m) => m[1] ?? '')
    .filter((b) => b.includes(`function ${name}`))
  if (fenced.length > 0) return fenced[fenced.length - 1]!.trim()
  // bare: from "function <name>" to its matching brace depth
  const start = reply.indexOf(`function ${name}`)
  if (start < 0) return undefined
  let depth = 0
  let seen = false
  for (let i = start; i < reply.length; i += 1) {
    if (reply[i] === '{') {
      depth += 1
      seen = true
    } else if (reply[i] === '}') {
      depth -= 1
      if (seen && depth === 0) return reply.slice(start, i + 1)
    }
  }
  return undefined
}

/**
 * Find the repaired function in a model reply, WHEREVER the protocol put it.
 *
 * The task loop demands every reply be ONE JSON tool call, so an
 * instruction-obedient model delivers the fix inside the call (typically
 * done.args.summary) - where newlines are JSON escapes. Running parseRepair
 * on the RAW reply text there extracts \n-riddled source that can never
 * eval (the 2026-06-13 flash round: 8/8 diagnosed, 0/8 scored). So: decode
 * the tool call first and scan its string values (note + args, recursively);
 * fall back to raw-text extraction for protocol-violating replies.
 */
export function extractRepairSource(reply: string, name: string): string | undefined {
  const call = parseToolCall(reply)
  if (call) {
    const strings: string[] = []
    const collect = (v: unknown, depth: number): void => {
      if (depth > 6) return
      if (typeof v === 'string') strings.push(v)
      else if (Array.isArray(v)) for (const x of v) collect(x, depth + 1)
      else if (v && typeof v === 'object') for (const x of Object.values(v)) collect(x, depth + 1)
    }
    if (call.note) strings.push(call.note)
    collect(call.args, 0)
    for (const s of strings) {
      const r = parseRepair(s, name)
      if (r !== undefined) return r
    }
  }
  return parseRepair(reply, name)
}

// ---------------------------------------------------------------- baseline arm

function baselineMessages(p: Problem): ChatMessage[] {
  const failing = p.tests.find((t) => {
    const fn = evalFunction(p.buggySource, p.name)
    return fn ? fn(...t.args) !== t.expected : true
  })
  return [
    {
      role: 'system',
      content:
        'You are a precise software engineer. Fix the bug. Reply with the corrected, ' +
        'complete function in a single ```js code block and nothing else.',
    },
    {
      role: 'user',
      content:
        `This function ${p.signature} has a bug:\n\n\`\`\`js\n${p.buggySource}\n\`\`\`\n\n` +
        (failing
          ? `Failing case: ${p.name}(${failing.args.join(', ')}) should be ${failing.expected}.\n\n`
          : '') +
        `Return the corrected function.`,
    },
  ]
}

type BaselineOutcome = { fixed: boolean; dnf: boolean; row: Record<string, unknown> }

/** One free-form repair attempt - shared by the model A and model B bare arms. */
async function runBaselineArm(client: ChatModel, p: Problem): Promise<BaselineOutcome> {
  try {
    const reply = await client.chat(baselineMessages(p))
    const repair = parseRepair(reply, p.name)
    const fixed = repair ? repairPasses(repair, p) : false
    return {
      fixed,
      dnf: false,
      row: {
        fixed,
        ...(process.env.RULITH_BENCH_TRANSCRIPT === '1' ? { reply: reply.slice(0, 4000) } : {}),
      },
    }
  } catch (error) {
    return {
      fixed: false,
      dnf: true,
      row: { fixed: false, dnf: true, error: String(error).slice(0, 120) },
    }
  }
}

// ---------------------------------------------------------------- board arm

function boardGoal(p: Problem): string {
  const failing = p.tests
    .filter((t) => {
      const fn = evalFunction(p.buggySource, p.name)
      return fn ? fn(...t.args) !== t.expected : true
    })
    .slice(0, 3)
  return (
    `Diagnose then repair this buggy function ${p.signature}:\n${p.buggySource}\n\n` +
    `Step 1: for each failing case assert observation(input, expected, got) facts. Known failing cases: ` +
    failing.map((t) => `${p.name}(${t.args.join(',')}) expected ${t.expected}`).join('; ') +
    `. Step 2: add ONE rule that DERIVES a diagnosis(kind=<one of: off_by_one, wrong_operator, ` +
    `inverted_condition, missing_case, wrong_initialization, boundary_zero, wrong_operand, ` +
    `wrong_return_value>) fact from your observations - do not assert the diagnosis directly. ` +
    `Step 3: read the board; once diagnosis is DERIVED, finish with a done tool call whose ` +
    `args.summary contains the corrected COMPLETE function in a \`\`\`js block.`
  )
}

type RepairScore = {
  fixed: boolean
  diagnosisKind?: string
  diagnosisMatched?: boolean
  turns: number
  transcript?: string[]
}

function readDiagnosis(store: MemorySpaceStore, spaceId: string): string | undefined {
  const facts = [...getLogicContext(store, spaceId).facts, ...getLogicContext(store, spaceId).findings]
  const diag = facts.find((f) => f.atom.predicate === 'diagnosis' && f.derived)
  if (!diag) return undefined
  const vals = Object.values(diag.atom.args ?? {}).map(String)
  return vals[0]
}

// ---------------------------------------------------------------- selftest

/** Scripted model that diagnoses on the board and returns a correct fix. */
class ScriptedRepairModel implements ChatModel {
  private step = 0
  /**
   * deliverViaDone=true is the PROTOCOL-FAITHFUL path: the fix travels
   * inside the done tool call's summary (a JSON string), never as a bare
   * ```js reply. This is how an instruction-obedient model behaves - the
   * 2026-06-13 flash round scored 0/8 with perfect diagnoses because the
   * harness only read raw reply text (JSON-escaped \n made every extracted
   * function unparseable). Kept red-to-green by the selftest.
   */
  constructor(
    private readonly p: Problem,
    private readonly fixed: string,
    private readonly deliverViaDone = false,
  ) {}
  async chat(): Promise<string> {
    this.step += 1
    if (this.step === 1) {
      const failing = this.p.tests
        .filter((t) => {
          const fn = evalFunction(this.p.buggySource, this.p.name)
          return fn ? fn(...t.args) !== t.expected : true
        })
        .slice(0, 3)
      const ops: unknown[] = failing.map((t, i) => ({
        op: 'assert_fact',
        id: `o${i}`,
        predicate: 'observation',
        args: { input: t.args.join(','), expected: t.expected, got: 'wrong' },
      }))
      ops.push({
        op: 'add_axiom',
        id: 'ax_diag',
        label: 'diagnosis from observations',
        when: [{ predicate: 'observation', args: { input: '?i', expected: '?e', got: '?g' } }],
        then: [{ predicate: 'diagnosis', args: { kind: this.p.bugKind } }],
      })
      return JSON.stringify({ tool: 'update_working_memory', args: { operations: ops } })
    }
    if (this.step === 2) {
      if (this.deliverViaDone) {
        return JSON.stringify({
          tool: 'done',
          args: { summary: `Diagnosis derived. Fix:\n\`\`\`js\n${this.fixed}\n\`\`\`` },
        })
      }
      return `Here is the fix:\n\`\`\`js\n${this.fixed}\n\`\`\``
    }
    return JSON.stringify({ tool: 'done', args: { summary: 'repaired' } })
  }
}

async function selftest(): Promise<void> {
  const problems = LIBRARY.map((s, i) => buildProblem(s, i + 1))

  // The buggy sources must actually fail at least one hidden test.
  for (const p of problems) {
    const fn = evalFunction(p.buggySource, p.name)
    if (!fn) throw new Error(`selftest: buggy source for ${p.name} does not compile`)
    const anyFail = p.tests.some((t) => fn(...t.args) !== t.expected)
    if (!anyFail) throw new Error(`selftest: buggy ${p.name} passes all tests - bug not seeded`)
  }

  // A correct fix must pass repairPasses; the original buggy source must not.
  const clamp = problems.find((p) => p.name === 'clamp')!
  assert(!repairPasses(clamp.buggySource, clamp), 'buggy clamp must not pass')
  assert(
    repairPasses('function clamp(x,lo,hi){ if(x<lo)return lo; if(x>hi)return hi; return x; }', clamp),
    'correct clamp must pass',
  )

  // parseRepair pulls the function out of a fenced reply.
  const parsed = parseRepair('blah\n```js\nfunction clamp(x,lo,hi){return x;}\n```\ntail', 'clamp')
  assert(parsed === 'function clamp(x,lo,hi){return x;}', `parseRepair fenced: ${parsed}`)

  // Board arm with a scripted correct model: fixed + diagnosis derived + matched.
  const p = problems.find((x) => x.name === 'sumTo')!
  const fix = 'function sumTo(n){ let s=0; for(let i=1;i<=n;i++){ s+=i; } return s; }'
  const score = await runBoardArm(new ScriptedRepairModel(p, fix), p, 6)
  assert(score.fixed, 'scripted board arm must fix sumTo')
  assert(score.diagnosisKind === 'off_by_one', `diagnosis kind: ${score.diagnosisKind}`)
  assert(score.diagnosisMatched === true, 'diagnosis must match ground truth')

  // Protocol-faithful delivery: fix inside the done tool call's summary.
  // An instruction-obedient model NEVER replies with a bare code block -
  // the harness must capture the fix from decoded tool-call strings too
  // (flash round 2026-06-13: 0/8 scored, 8/8 diagnosed - harness gap).
  const faithful = await runBoardArm(new ScriptedRepairModel(p, fix, true), p, 6)
  assert(faithful.fixed, 'a fix delivered inside the done tool call must be captured and scored')

  // Bare-arm helper (shared by the model A and model B bare arms): a
  // scripted correct fix scores, a dead endpoint is a DNF row.
  const bare = await runBaselineArm({ chat: async () => `fix:\n\`\`\`js\n${fix}\n\`\`\`` }, p)
  assert(bare.fixed && !bare.dnf, 'bare arm must score a scripted correct fix')
  const deadBare = await runBaselineArm(
    {
      chat: async () => {
        throw new Error('endpoint down')
      },
    },
    p,
  )
  assert(!deadBare.fixed && deadBare.dnf, 'bare arm must record a DNF row on endpoint failure')

  console.log('bench-repair selftest PASSED (seeded bugs fail, fixes scored, board diagnosis derived, bare arm sane)')
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`selftest: ${msg}`)
}

async function runBoardArm(llm: ChatModel, p: Problem, maxTurns: number): Promise<RepairScore> {
  const store = new MemorySpaceStore()
  let spaceId = ''
  let turns = 0
  // Capture the reply that CONTAINS the fix, not merely the last reply -
  // the model emits the repair mid-conversation (or inside its final done
  // tool call); a later fix overwrites an earlier one, nothing else does.
  let repairSource: string | undefined
  const transcript: string[] = []
  const tap: ChatModel = {
    chat: async (messages) => {
      const reply = await llm.chat(messages)
      const src = extractRepairSource(reply, p.name)
      if (src !== undefined) repairSource = src
      if (process.env.RULITH_BENCH_TRANSCRIPT === '1') transcript.push(reply.slice(0, 4000))
      return reply
    },
  }
  await runAgentTask({
    store,
    llm: tap,
    reg: new ToolRegistry(),
    rootDir: process.cwd(),
    goal: boardGoal(p),
    maxTurns,
    onContext: (info) => {
      spaceId = info.spaceId
    },
    onTurn: () => {
      turns += 1
    },
  })
  const diagnosisKind = readDiagnosis(store, spaceId)
  const fixed = repairSource ? repairPasses(repairSource, p) : false
  const out: RepairScore = {
    fixed,
    diagnosisKind,
    diagnosisMatched: diagnosisKind === undefined ? undefined : diagnosisKind === p.bugKind,
    turns,
  }
  if (process.env.RULITH_BENCH_TRANSCRIPT === '1') out.transcript = transcript
  return out
}

// ---------------------------------------------------------------- main

async function main(): Promise<void> {
  if (process.argv.includes('--selftest')) {
    await selftest()
    return
  }

  const N = Math.min(LIBRARY.length, Number(process.env.RULITH_BENCH_N ?? LIBRARY.length))
  const bConfig = modelBConfigFromEnv()
  const arms = resolveArms(process.env.RULITH_BENCH_ARM, bConfig !== undefined)
  const maxTurns = Number(process.env.RULITH_BENCH_TURNS ?? 8)
  const problems = LIBRARY.map((s, i) => buildProblem(s, i + 1)).slice(0, N)

  const llm = new LlmClient()
  const llmB = bConfig ? new LlmClient(bConfig) : undefined
  mkdirSync('logs', { recursive: true })
  const logPath = join('logs', `bench-repair-${Date.now()}.jsonl`)
  const log = (e: unknown): void => appendFileSync(logPath, `${JSON.stringify(e)}\n`)
  // Attribution header: model identity from env, not from any UI banner
  // (lesson of validation #27 - environment labels are assertions too).
  log({
    type: 'config',
    bench: 'repair',
    startedAt: new Date().toISOString(),
    model: process.env.RULITH_LLM_MODEL ?? '(client default)',
    note: process.env.RULITH_BENCH_NOTE,
    arms: [...arms].join(','),
    modelB: bConfig?.model ?? null,
    baseUrlB: bConfig ? (bConfig.baseUrl ?? '(model A endpoint)') : null,
    n: N,
    maxTurns,
  })

  const ORDER: readonly BenchArm[] = ['baseline', 'baseline_b', 'board', 'board_b']
  const active = ORDER.filter((a) => arms.has(a))
  const clientFor = (a: BenchArm): LlmClient => (a.endsWith('_b') ? llmB! : llm)
  const tally = new Map(
    active.map((a) => [a, { fixed: 0, diagMatched: 0, dnf: 0, tok: emptyUsageTally() }]),
  )

  for (const p of problems) {
    const row: Record<string, unknown> = { id: p.id, name: p.name, bugKind: p.bugKind }
    for (const arm of active) {
      const t = tally.get(arm)!
      const client = clientFor(arm)
      client.consumeUsage() // open this arm's token window
      console.error(`[${p.name}] ${arm} arm started`)
      let armRow: Record<string, unknown>
      if (arm === 'baseline' || arm === 'baseline_b') {
        const outcome = await runBaselineArm(client, p)
        if (outcome.fixed) t.fixed += 1
        if (outcome.dnf) t.dnf += 1
        armRow = outcome.row
      } else {
        try {
          const score = await runBoardArm(client, p, maxTurns)
          if (score.fixed) t.fixed += 1
          if (score.diagnosisMatched) t.diagMatched += 1
          armRow = { ...score }
        } catch (error) {
          t.dnf += 1
          armRow = { fixed: false, dnf: true, error: String(error).slice(0, 120) }
        }
      }
      const u = client.consumeUsage()
      if (u.calls > 0) armRow.tokens = usageRow(u)
      addUsage(t.tok, u)
      row[arm] = armRow
    }
    log(row)
    console.log(JSON.stringify(row))
  }

  console.log('---')
  const label = (a: BenchArm): string =>
    a.endsWith('_b') ? `${a} [${bConfig?.model}]` : `${a} [${process.env.RULITH_LLM_MODEL ?? 'local-model'}]`
  for (const arm of active) {
    const t = tally.get(arm)!
    if (arm === 'baseline' || arm === 'baseline_b') {
      console.log(`${label(arm)} (free-form): ${t.fixed}/${problems.length} repaired (${t.dnf} DNF)`)
    } else {
      console.log(`${label(arm)} (diagnose-first): ${t.fixed}/${problems.length} repaired (${t.dnf} DNF)`)
      console.log(`${label(arm)} diagnosis matched ground truth: ${t.diagMatched}/${problems.length}`)
    }
    if (t.tok.calls > 0) console.log(`${label(arm)}: ${fmtUsage(t.tok)}`)
  }
  console.log(`log: ${logPath}`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})