# Feature Overview — 2026-10-05

MLX serving, a cross-request state cache, a benchmark suite (synthetic + real corpora),
and a JevBench-validated comparison against the frozen Qwen3.5-4B torso. All work was
done on an Apple M5 Max (128 GB) against `StrandsAgents/strands-decider-2B-hobson-v21`.

## 1. MLX serving setup

The package installs from a clone with the `mlx` extra and serves on Apple silicon:

```bash
uv venv && uv pip install -e ".[mlx]"
.venv/bin/strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 \
    --device mlx --state-cache 8 --port 8000
```

All three question types (`noul`, `choice`, `score`) verified over HTTP and through the
CLI. Note: the README's curl example omits the required `criteria` field; the server
schema (`src/strands_decider/schema.py`) is authoritative — `choice` takes a
`{option: description}` dict, `score` an ascending rubric list.

## 2. Cross-request state cache (the feature)

### Problem

The engine already had an *intra-request* prefix cache (`use_prefix_cache`): one state,
many questions — the state forwards once and every question reuses it. But across HTTP
requests the state was re-encoded every time. Measured: the same ~1400-token state sent
40 times cost ~98 ms per request, every time. Interactive agent workloads (a Strands
intervention asking follow-up questions about the same conversation) pay that repeatedly.

### Design

`MLXEngine` (`src/strands_decider/mlx_engine.py`) now keeps an LRU cache mapping
**state token ids → a batch-1 snapshot of the prompt-cache layers** produced after
encoding that state.

- **Populate:** only `_slot_probs_shared_prefix` fills it — the one path that already
  builds a state-only prefix (requests with ≥2 questions, including every chunk of a
  chunked request). No request ever pays an extra forward to fill an entry; the
  single-question miss path is byte-for-byte what it was.
- **Read:** `_slot_probs_batched` (single-question requests — the common server case)
  consults the cache first. On a hit it fans the snapshot out with
  `merge([layer] * len(questions))` and forwards only the question suffix.
- **Key:** the post-`_fit` token ids, so truncation is accounted for; two requests with
  the same text and different question sets share an entry.
- **Isolation:** snapshots are copies; the fan-out `merge` never mutates them (pinned by
  test: repeated requests give identical answers). All access is under the engine's
  existing evaluation lock.
- **Eviction:** LRU, capacity `--state-cache N` (default 8, `0` disables). `/health`
  reports the setting; `load_mlx_engine(..., state_cache_entries=N)` for library use.

### mlx-lm workaround

`KVCache.merge` returns a `BatchKVCache` whose `offset` is a per-row mx **array**;
merging such a cache a second time slices with the array and raises
`ValueError: Slice indices must be integers`. mlx-lm only ever merges a fresh cache, so
this was unexercised upstream. `_snapshot()` converts the batch-1 copy back to a plain
`KVCache` (integer offset) so snapshots merge as often as needed. `ArraysCache` (the
Gated DeltaNet states) already returns re-mergeable caches and passes through.

### Measured (warm, M5 Max, MLX)

| scenario | before | after | speedup |
| --- | --- | --- | --- |
| repeated ~1400-token state, single question | 98.2 ms | 18.8 ms | **5.2×** |
| short state, single question | 54.8 ms | 19.0 ms | 2.9× |
| real corpora, grouped 2-question states | 52.3 ms | 22.4 ms | **2.3×** |
| varied questions against a cached state (noul / choice 2–10 options / score) | — | 17.7–18.6 ms | type-blind |
| many questions per request (10–30) | — | 2.5× → ~1× | amortises regardless |

### Operational caveat

**Cache capacity must exceed the client's distinct-state working set.** With 60 states
cycling through an 8-entry cache, pass-2 gains were 0% — everything was evicted before
reuse. At `--state-cache 128` the same workload gained 2.3×. Memory per entry scales
with state length (KV + DeltaNet state for up to 4096 tokens); size the flag to the
workload.

### Correctness

- 4 new tests in `tests/test_mlx_engine.py`: a repeated state is encoded once
  (`state_encodes` counter); a single question reuses a cached state and matches the
  uncached answers within 1e-4; LRU eviction order; `state_cache_entries=0` disables.
- Full JevBench v1 public set (231 tasks) through the cached server: **0.766 (177/231),
  Brier 0.3231** against the published 0.762 / 0.323 — one task of difference, inside
  the repo's own "under ~10 tasks is unresolved" rule. The cache changes no answers.
- 3 pre-existing `test_mlx_answers_as_torch_does[*-gpu]` parity drifts (max 0.0003 vs
  the 1.5e-4 tolerance) were confirmed on unmodified HEAD — Metal kernel numerics on
  this machine, not this change.

## 3. Benchmark suite (`bench/`)

| script | what it measures |
| --- | --- |
| `bench_types.py` | 1000 tests per question type; per-request and many-questions-per-request modes |
| `bench_all.py` | synthetic sweep in 7 sections: state length × type, questions/request, choice width (2–50 options), mixed vs separate requests, state-cache hit/miss, concurrency (1/4/16 clients), full-window steady state |
| `bench_corpus.py` | **real data**: `data/synthetic/*_eval.jsonl` — replay (per-type latency + accuracy + confidence-when-right/wrong), grouped-by-state (cache miss vs hit passes), instruction-paraphrase consistency |
| `jevbench_4b_letters.py` | frozen Qwen3.5-4B on the 231 public tasks, SemIf-style option-letter logits, for model comparison |
| `run.sh` | runs all of the above against one live server |

Pure mapping functions in `bench_corpus.py` (`record_questions`, `expected_answer`,
`check`, `score_question`, `load_records`) are pinned by `tests/test_bench_corpus.py`
(6 tests) so a schema drift in the corpora or the API fails in CI rather than silently
producing a benchmark that measures the wrong thing.

### Baseline numbers (M5 Max, MLX, warm)

- Single question, short state: p50 ~34–36 ms; type makes no difference.
- Latency scales ~linearly with state tokens until the 4096 window clamps it
  (~290–320 ms at full window).
- Questions amortise: 51 ms/q single → 6.7 ms/q at 500 questions in one request.
- Choice width 2 → 50 options: 31.7 → 52.7 ms (the pointer head scales with option
  tokens, cheaply).
- Mixed types in one request: 2.1× cheaper than three separate requests.
- Concurrency: throughput flat ~50 req/s at 1/4/16 clients — Metal serialises GPU
  work; parallel clients only inflate per-request latency. Single client is optimal.
- Real corpora: choice p50 121.6 ms / acc 0.73, noul 32.1 ms / acc 0.72, paraphrase
  consistency 1.00, confidence 0.79 when correct vs 0.44 when wrong.

## 4. Frozen 4B comparison

`bench/jevbench_4b_letters.py` reads Qwen3.5-4B through its chat template and scores one
forward pass per task over the option-letter logits — the repo's reference method for
frozen models. Results on the same 231 tasks, same machine:

| system | accuracy | median | p95 |
| --- | --- | --- | --- |
| decider 2B (MLX + state cache) | **0.766** | 20 ms | 185 ms |
| frozen 4B, letter logits (all tasks) | 0.710 | 48 ms | 684 ms |
| frozen 4B, excluding score tasks | 0.770 (164/213) | — | — |
| published SemIf run on frozen 4B | 0.805 (their engineered prompt) | — | — |

Per family, the 4B wins `probability` (0.70 vs 0.30), `tradeoff` (1.00 vs 0.50) and
`multi_hop` (0.56 vs 0.44); the decider wins `ordinal` outright (**1.00 vs 0.00** —
letter-logit reading cannot do ordinal expected value; SemIf excludes score for this
reason) and ties or wins the classification families. `temporal_numeric` is 0.267 for
both — capability-bound, as the research notes predict.

Implication recorded for future training: probability / tradeoff / multi_hop are
learnable — the knowledge exists in the 4B teacher, and v21's distillation method
(where the teacher agrees with gold) extends to generated data for exactly those
families.

**Gotcha documented in the script:** Qwen3.5 is a thinking model. Logits taken at the
generation prompt sit inside the `<think>` block and score 0.355; appending the
empty-think prefix (`<think>\n\n</think>\n\n`) puts the read at the answer position and
doubled accuracy.

## 5. File inventory

| file | change |
| --- | --- |
| `src/strands_decider/mlx_engine.py` | state cache, `_snapshot` (BatchKVCache workaround), `state_cache_entries` |
| `src/strands_decider/server.py` | `state_cache` plumbing, `/health` field, direct MLX engine load |
| `src/strands_decider/cli.py` | `serve --state-cache N` |
| `tests/test_mlx_engine.py` | 4 state-cache tests |
| `tests/test_bench_corpus.py` | 6 mapping tests (new) |
| `bench/bench_types.py` | new |
| `bench/bench_all.py` | new |
| `bench/bench_corpus.py` | new |
| `bench/jevbench_4b_letters.py` | new |
| `bench/run.sh` | new |

Not committed as of this writing. Raw artifacts under `/tmp`: JevBench results
(`/tmp/jev_results.jsonl`), 4B letter-logit results (`/tmp/jev_4b_letters.jsonl`),
task file (`/tmp/jevbench_all.jsonl`, pinned commit `1bcc55e`).
