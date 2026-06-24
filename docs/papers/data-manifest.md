# Data manifest — run provenance for *The Driving Floor*

Ties each Table 1 / A1 row (and the figures derived from it) to the on-disk run that produced it.
**Auto-derived** from the `{"type":"config", ...}` header line of each `runs/<dir>/bench-*.jsonl`
log (canonical `runs/` tree; the `runs - 副本/` backup and the three excluded runs of §5.1 are
not listed).

**Reading the table.** `served model` is the id actually sent to the endpoint; where it is the
LM Studio generic **`board-llm`**, the real model is given by `run-dir` + `note` (this is the gap
the manifest closes — the log alone does not name the weight). `quant` / `mode` are encoded in the
run-dir name. **seed = 1000 for every run.** **`commit` is not recorded** in these logs (they
predate per-run commit stamping) — use the release tag as the code reference. `endpoint` is
**inferred, not logged**: `deepseek-*` = DeepSeek cloud API; all others = local LM Studio
(`http://127.0.0.1:1234/v1`).

| run-dir | served model | note (label) | quant · mode | n | started (UTC) | endpoint |
|---|---|---|---|---:|---|---|
| **qwen3.6-35b-a3b** | qwen/qwen3.6-35b-a3b-mtp | a3b-harness-Q4_K_M | 4-bit (Q4_K_M) · nothink — **anchor, 3B-active** | 50 | 2026-06-23 01:43 | local |
| qwen3.6-27b | qwen/qwen3.6-27b-mtp | qwen3.6-27b-mtp | 4-bit (mtp) · nothink | 50 | 2026-06-19 06:54 | local |
| qwen3.6-27b-fp16-nothink | board-llm | qwen27b-fp16-nothink | fp16 · nothink | 50 | 2026-06-17 00:18 | local |
| qwen3.6-27b-fp16-think | board-llm | qwen3.6-27b-fp16-think | fp16 · think | 36 † | 2026-06-22 19:13 | local |
| qwen3.6-27b-fp8-nothink | board-llm | qwen3.6-27b-fp8-nothink-rerun | fp8 · nothink | 50 | 2026-06-21 09:37 | local |
| qwen3.6-27b-fp8-think | board-llm | qwen3.6-27b-fp8-think-rerun | fp8 · think | 50 | 2026-06-21 09:50 | local |
| gemma-4-31b | google/gemma-4-31b-qat | gemma-4-31b-qat | 4-bit (qat) · nothink | 50 | 2026-06-19 18:02 | local |
| gemma-4-31b-fp16-nothink | board-llm | gemma-4-31b-fp16-nothink-rerun | fp16 · nothink | 50 | 2026-06-21 13:45 | local |
| gemma-4-31b-fp16-think | board-llm | gemma-4-31b-fp16-think-rerun | fp16 · think | 50 | 2026-06-21 14:09 | local |
| gpt-oss-120b-lowthink | openai/gpt-oss-120b | gpt-oss-120b | 4-bit (official) · lowthink | 50 | 2026-06-20 13:36 | local |
| deepseek-v4-flash-nothink | deepseek-chat | deepseek-flash-v4-chat2 | full · nothink | 50 | 2026-06-20 22:18 | cloud |
| deepseek-v4-flash-think | deepseek-reasoner | deepseek-flash-v4-reasoner2 | full · think | 50 | 2026-06-20 22:21 | cloud |
| gemma-4-26b-a4b | google/gemma-4-26b-a4b | gemma-4-26b-a4b-harness | 4-bit · nothink — 4B-active MoE, **partial driver** | 50 | 2026-06-22 14:34 | local |
| gemma-4-12b | google/gemma-4-12b-qat | gemma-4-12b-qat | 4-bit (qat) · nothink — **boundary probe** | 50 | 2026-06-22 19:31 | local |
| qwen3.5-9b | qwen/qwen3.5-9b | qwen3.5-9b | 4-bit · nothink — **boundary probe** | 50 ‡ | 2026-06-22 22:41 | local |

† Config requested **n=50**, but the run **did not finish (too slow)** — **36 of 50 completed**,
which is the n Table A1 cites. Not a discrepancy: the paper's n=36 is the correct scored count.

‡ `qwen3.5-9b` has **4 `bench-arith` re-runs** in `runs/` (boundary probe; timestamps
`…086277 / …351694 / …300194 / …916160`). Table 1 reports it small-n on the strict harness (§5.5).

---

Each run-dir holds `bench-arith`, `bench-audit`, `bench-coding-trust` `*.jsonl`. The aggregate
(Table 1 / A1, Wilson CIs, token tallies) is produced by
`npx tsx src/examples/bench-aggregate.ts` over these logs. The `runs/` tree is local (gitignored,
not shipped); the §Reproducibility appendix gives the env + commands to regenerate it.
