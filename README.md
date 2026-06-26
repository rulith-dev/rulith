# RULITH

**An external reasoning board for LLM agents. Derived or it didn't happen.**

rulith — *rule* + *-lith* (Greek *líthos*, "stone") — is a stone tablet
for an agent's rules: a working memory with a rule engine. The agent proposes **facts**,
**rules**, and **actions**; the board computes deductive closures, does
exact arithmetic, tracks consumption/production, and keeps an evidence
chain for every conclusion. The agent cannot launder a guess into a
result: every claim on the board is **derived** (closure-backed), an
**effect** (action product), or **asserted** (bare claim) — and results
that rest on bare claims are rejected.

> First run through the published package, a 27B local model driving the
> board from Claude Code: the frontier model supervising the release
> (Claude Fable 5 Max) had mentally computed `9381274 × 6473` and
> confidently repeated the wrong answer three times. The board derived the right one. That incident is
> validation round #27 — the product demoing itself on its own author.

## Papers

- **The Driving Floor: When an External Symbolic Reasoning Board Helps an LLM** — the empirical study (board vs baseline across quantized local models). [PDF](https://github.com/rulith-dev/rulith/blob/main/docs/papers/preprint-draft.pdf) · [中文版](https://github.com/rulith-dev/rulith/blob/main/docs/papers/preprint-draft-zh.pdf)
- **The Rulith Decision Kernel: Proof-Carrying Decisions for Autonomous Agents** — the whitepaper (the trust invariants + the commitment ladder). [PDF](https://github.com/rulith-dev/rulith/blob/main/docs/papers/whitepaper-self-driving-kernel.pdf) · [中文版](https://github.com/rulith-dev/rulith/blob/main/docs/papers/whitepaper-self-driving-kernel-zh.pdf)

## Why

LLMs assert; they do not prove. For tasks where a wrong number or an
unverified claim is expensive — audits, invoices, inventory, multi-step
analysis — the fix is not a smarter model but a surface the model must
show its work on:

- **Exact-or-fail arithmetic** — integer math is exact within ±2^53;
  overflow, NaN, and silent precision loss fail loudly instead of
  rounding. The model never does arithmetic in its head.
- **Derivation gate** — `finding(...)` facts must be derived by the rule
  closure from primitive observations. Asserted findings block
  `record_result`. There is no way to claim without showing.
- **Actions with history** — consume/produce transformations archive what
  they consume and record an event (binding, consumed, produced). The
  board keeps the process, not just the end state.
- **Truth maintenance** — retract an input and everything resting on it
  falls; contradictions taint downstream conclusions as disputed.
- **Teaching errors** — every rejection explains how to fix the call.
  Validated to keep 27B-class local models productive.
- **No model, no GPU, no network** — rulith never calls an LLM. It is a
  pure local kernel (Node ≥ 20, two pure-JS dependencies) that the agent
  drives over MCP stdio.

## Install

As a Claude Code / Cowork plugin (MCP server + skill in one step):

```
/plugin marketplace add rulith-dev/rulith
/plugin install rulith@rulith
```

As a bare MCP server in Claude Code:

```bash
claude mcp add rulith -- npx -y rulith
```

Or in any MCP host, project-scoped `.mcp.json`:

```json
{
  "mcpServers": {
    "rulith": { "command": "npx", "args": ["-y", "rulith"] }
  }
}
```

Optional persistence across sessions: set env `RULITH_DB` to a `.jsonl`
file path. Without it, the board lives and dies with the session.

## Quickstart

rulith is driven by an agent over MCP: the agent proposes facts and rules,
the board derives the rest — and shows its work. A minimal session — line-item
costs the model can't fudge:

**1. Open a board.** `create_space` `{ "title": "invoice" }` returns a space id.

**2. Assert the line items + a costing rule** in one `update_working_memory` call:

```json
{
  "operations": [
    { "op": "assert_fact", "predicate": "line", "args": { "item": "widget", "unit": 1299, "qty": 7 } },
    { "op": "assert_fact", "predicate": "line", "args": { "item": "gasket", "unit": 4500, "qty": 12 } },
    { "op": "add_axiom", "id": "ax_cost", "label": "cost = unit * qty",
      "when": [
        { "predicate": "line", "args": { "item": "?i", "unit": "?u", "qty": "?q" } },
        { "predicate": "mul",  "args": { "left": "?u", "right": "?q", "result": "?t" } }
      ],
      "then": [{ "predicate": "cost", "args": { "item": "?i", "total": "?t" } }] }
  ]
}
```

`mul` is a built-in arithmetic predicate: the board computes `1299 × 7` and
`4500 × 12` exactly (BigInt-checked) and binds `?t`.

**3. Read the board.** `get_logic_context` returns `cost(widget, 9093)` and
`cost(gasket, 54000)`, each tagged `[derived]` with an evidence chain back to its
`line` fact. The model never did the arithmetic — the board did, and it cannot be
off by a digit.

From here: roll the costs into a total with `derive_aggregate` (sum), guard a
budget with the `gt` built-in, or consume/produce inventory with `define_action`.
Open goals come back with teaching hints (`needs via <rule>: ...`) naming the
missing fact. And `record_result` on a bare assertion — rather than a derived
fact — is rejected: show your work, or get nothing.

## Tools

`create_space`, `update_working_memory` (declare_goal / assert_fact /
add_axiom / define_action / declare_hypothesis / record_result /
retract_node / revise_fact), `simulate_action`, `apply_action`,
`get_logic_context`, `distill_space`, `list_spaces`.

Open goals come back with teaching hints: which rule is missing which
facts (`needs via ...`), and which defined action could produce the
missing atom (`producible via action ...`).

## Validated, not vibe-coded

This kernel was built against a discipline of red-tests-first and
real-model validation: 100+ logged rounds of
local models (gemma/qwen, 27B–35B class) driving the board through real
tasks — judgment, diagnosis-and-repair, open-ended audit, stoichiometric
reactions — each round documented with board evidence, each kernel gap
found by a real run, exposed by a failing test, then fixed. The entire
series ran on an AMD Strix Halo iGPU (Radeon 8060S); no discrete GPU was
involved at any point.

**Hard-arithmetic A/B** (validation round #28, seeded and reproducible —
8-digit × 5-digit line items, 5–8 lines, exact totals, same 27B local
model both arms): plain chat scored **0/10** (three confidently wrong
totals, seven non-terminating DNFs at a 10-minute cap); the board arm
scored **8/10**, every solved value closure-derived, median 3 turns.
Across all ten problems the board never displayed a single wrong number
— it either derived the exact value or claimed nothing. The two board
losses were generation-level runaways, replayed clean and re-verified
with BigInt. 1,200+ unit tests; CI on Linux and Windows.

Eight A/B benchmark fixtures ship in `src/examples/`, each pitting board against
baseline on a dimension a strong model still gets wrong:
`bench-arith` (exact arithmetic), `bench-audit` (error-finding audits),
`bench-coding-trust` (a fabricated "fix" is blocked — the board certifies a repair
only from a real edit plus a passing test), `bench-repair` (diagnosis-first bug
repair), `bench-revision` (content-addressed consistency under concurrent edits),
and `bench-aggregate` (exact summation) — plus the `bench-arms` / `bench-pool`
cross-model harness (run any fixture board-vs-baseline, two models, token-metered).

## License

[Apache License 2.0](LICENSE).
