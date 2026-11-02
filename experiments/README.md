# Evaluation harness

**Status: infrastructure only.** This directory contains the benchmark dataset, runner, and
metrics for evaluating the LangGraph sales agent. No baseline has actually been run yet — the
numbers this harness produces are not claimed anywhere until a real run is executed and the
resulting JSON file is committed.

## Purpose

We want to compare LLM/agent configurations (different models, prompts, tool sets, retrieval
strategies, etc.) against the same fixed set of shopping queries, rather than judging a model
change by a handful of manual chat messages. This harness is the reusable mechanism for that:
same dataset, same scoring code, any model.

## Directory structure

```
experiments/
├── benchmarks/
│   ├── saree_shopping_v1.json      # the benchmark dataset (42 cases)
│   └── seed_eval_catalog.sql       # dedicated ACTIVE fixture products the benchmark queries against
├── configs/
│   └── baseline_qwen3-8b.json      # named experiment config (provider/model/benchmark file)
├── results/                        # run outputs land here as JSON, one file per run (gitignored)
└── README.md
```

The harness code itself lives in `src/evaluation/` (source, not experiment output):
`dataset.py` (loads the JSON), `runner.py` (executes cases through the agent graph),
`metrics.py` (aggregates scores), `cli.py` (ties it together and writes a results file).

## Why a dedicated fixture catalog

The real sample products (`SAM-2026-0001..0003`) are intentionally left in `DRAFT` status
(see project history) and every agent tool only returns `ACTIVE` products. A benchmark also
shouldn't depend on mutable production/business data anyway — if someone marks a real product
`SOLD` or edits its attributes, results would stop being comparable over time. So
`seed_eval_catalog.sql` inserts 8 clearly-named `EVAL-0001..EVAL-0008` products with their own
prices and inventory (including one `SOLD` and one `RESERVED` item, so availability-related
queries have something real to distinguish). The 3 real `SAM-2026` products are never touched.

Run it once before evaluating (idempotent, safe to re-run):

```bash
docker exec -i sambalpuri-ai-postgres psql -U app_user -d sambalpuri_ai < experiments/benchmarks/seed_eval_catalog.sql
```

### Fixtures are excluded from the real catalog and agent by default

`EVAL-*` products are inserted as `ACTIVE` (so the evaluation harness can actually search/recommend
them), which means without an explicit exclusion they would also appear in the real public catalog
and to real customers talking to the agent. To prevent that:

- `catalog.public_products` (used by `GET /products` and `GET /products/{sku}`) filters out any
  `sku LIKE 'EVAL-%'` at the view level.
- `search_products()`, `recommend_products()`, and `get_product_details()` in `src/search/` take an
  `include_eval_fixtures` argument, defaulting to `False` everywhere — the public API
  (`src/api/routers/products.py`) never sets it, so `/products/search` and `/products/recommendations`
  never see fixtures either.
- The agent's tools (`src/agent/tools.py`) check the `AGENT_INCLUDE_EVAL_FIXTURES` env var and pass
  it through. It's off by default, so a real customer talking to the agent never sees fixtures.
  `src/evaluation/cli.py` sets it to `"1"` for itself before running, which is the only place that does.

## Dataset structure

Each case in `saree_shopping_v1.json` has:

| field | meaning |
|---|---|
| `id` | unique case id |
| `category` | one of: product_discovery, occasion, color, fabric, bandha_motif, thread_count, budget, availability, product_detail, recommendation, ambiguous, multi_constraint, conversational_followup |
| `query` (or `turns`) | the customer message; `turns` is a list for multi-turn conversational cases, sent through the graph one after another with state carried forward |
| `expects_tool_call` | whether a well-behaved agent should call a tool at all for the final turn |
| `expected_tool` | tool name, a list of acceptable tool names, or `null` if not gradable |
| `expected_constraints` | dict of argument name → expected value, checked against the *first* tool call's arguments; `null` when the correct arguments can't be determined in advance (e.g. superlatives like "cheapest", or attributes no tool currently supports) |
| `relevant_skus` | reserved for future ranking-quality metrics (Precision@K / Recall@K / NDCG@K); always `null` today, not fabricated |
| `capability_gap` | `true` if the case tests something no current tool can actually satisfy (see below); defaults to `false` |
| `notes` | why the case is graded the way it is, including known tool-surface gaps (e.g. no `min_price`, no motif filter) |

Every expectation is grounded in the actual `EVAL-*` fixture data and the actual parameters the
three tools (`search_products`, `recommend_products`, `get_product_details`) currently accept —
not assumed or aspirational behavior. Several cases deliberately test real current limitations
(no motif filter, no `min_price`, in-stock-only defaults hiding sold/reserved items) because
those are useful, honest signals for a first baseline.

Five cases (`bandha_003`, `thread_003`, `budget_003`, `budget_004`, `detail_003`) are marked
`capability_gap: true` because there is no tool argument that could ever satisfy them correctly
(no motif filter, no "highest"/"cheapest" reasoning, no `min_price`, no name-based lookup). These
are excluded from the accuracy metrics — see below — so a model is never penalized as if it made
a wrong decision when the tool layer itself has no right answer to give it.

## Evaluation methodology

For each case, the runner sends its turn(s) through `build_graph()` (the same graph the real
agent uses), collects every tool call and tool result, and records the final assistant message.
Scoring is structural, not natural-language matching:

- **Tool selection correct**: a tool was called iff `expects_tool_call` says one should be, and
  (when specified) the first tool called is one of `expected_tool`.
- **Argument match ratio**: fraction of `expected_constraints` keys whose value matches the
  actual tool call argument (case-insensitive for strings). `null` when no constraints are
  defined for the case.
- **Tool call valid**: the tool executed without returning an `{"error": ...}` payload. `null`
  when no tool was called.
- **Task success**: tool selection was correct, no tool call was invalid, and a final response
  was produced.
- **Latency**: wall-clock seconds for the whole case (all turns).
- **Execution failure**: the case raised an exception (e.g. the model/agent crashed outright),
  captured separately from "graceful" scoring failures.

## Metrics produced

`compute_metrics()` reports, unconditionally over every case:

- `execution_failure_rate`, `latency_seconds` (mean / median / p95 / min / max)
- `graded_cases` / `capability_gap_cases` counts

And, computed only over the non-`capability_gap` ("graded") cases, overall and broken down by
category:

- `tool_selection_accuracy`
- `mean_argument_match_ratio`
- `tool_call_validity_rate`
- `task_success_rate`

The 5 `capability_gap` cases get their own `capability_gap_summary` (case ids + task success
rate) instead of being folded into the numbers above.

**Not included yet, by design:** Precision@K, Recall@K, NDCG@K for search/recommendation
ranking quality. The `relevant_skus` field exists so a future version can compute these against
a human-labeled relevance set, without changing the dataset schema or the runner's output shape.
No such scores are fabricated in the meantime.

## How to run the baseline

```bash
# 1. make sure the eval fixture catalog exists (idempotent)
docker exec -i sambalpuri-ai-postgres psql -U app_user -d sambalpuri_ai < experiments/benchmarks/seed_eval_catalog.sql

# 2. pull the baseline model once
ollama pull qwen3:8b

# 3. run the benchmark (LLM_PROVIDER/LLM_MODEL already default to ollama/qwen3:8b in .env.example)
python -m src.evaluation.cli
```

This writes `experiments/results/<provider>_<model>_<timestamp>.json` containing the full
per-case results plus the aggregate metrics, and prints the metrics summary to stdout.

## Adding a new experiment

- **New model/provider**: set `LLM_PROVIDER`/`LLM_MODEL` (or any other provider env vars) and
  re-run the same CLI command — nothing in the harness is Ollama- or Qwen-specific.
- **New agent configuration** (different prompt, extra tools, memory, multimodal retrieval,
  etc.): as long as `build_graph()` still returns a graph with the same `messages`-in/out
  shape, the runner needs no changes.
- **New benchmark version**: add another JSON file under `benchmarks/` and pass
  `--benchmark path/to/file.json` to the CLI; old result files stay comparable to old benchmark
  versions since each result records which benchmark file produced it.
- **New metric**: add it to `metrics.py`'s `compute_metrics()` — the per-case `CaseResult`
  objects already carry tool calls, tool results, and timing, so most new metrics are a matter
  of aggregating fields that already exist.
