# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); entries are generated
from the Conventional Commits history (`git log --no-merges`).

No version has been released yet — `0.1.0` is the first planned tag
(`[tool.commitizen]` in `pyproject.toml`; publishing runs by manual dispatch
via `.github/workflows/release-pypi.yml`). When a tag lands, move the
Unreleased block under it and cut a new empty one.

## [Unreleased]

### Added

- Serving benchmark suite (`bench/`, ADR-005): corpus, concurrency and type
  sweeps against the live server; results captured under `bench/results/`.
- Cross-request state cache for the MLX engine (`--state-cache`, ADR-004):
  18.8 ms against 98.2 ms on a repeated ~1400-token state — a measured 5.2×.
- Image input for Strands Decider (v19): `serve --vision`, `ask --image`,
  vision tower kept, no retraining.
- Serving on Apple silicon through MLX: `--device mlx` (ADR-003).
- Strict context window, `max_batch`, and a wider request schema on the
  serving endpoint.
- RLVR with GRPO — PRD and SPEC (proposed, `tasks/prd-rlvr-grpo.md`,
  `tasks/spec-rlvr-grpo.md`). RLHF/DPO/RLCD rejected: no preference data.
- Architecture Decision Records: MLX backend (ADR-003), cross-request state
  cache (ADR-004), benchmark suite (ADR-005), joining the pointer-head (ADR-001)
  and question-first-window (ADR-002) records.

### Changed

- Evaluations are now reentrant: per-request option offsets are scoped to the
  request (race #9), so the torch engine evaluates concurrently; the MLX engine
  serialises under its lock (shared forward path, and the lock guards the state
  cache).
- Type gates: mypy baseline in `src/` cleared to zero.

### Fixed

- Race #9: two interleaved evaluations in the server's thread pool scored one
  request's options at the other's token positions. Both torch paths and the
  vision engine now interleave correctly, pinned by deterministic barrier tests.
- MLX equivalence tests hold their tolerance to Metal's fp32 matmul fast path.
- Base model pinned to the revision the adapter was trained on (#19).
- MPS keeps its chunk rule when flash-linear-attention cannot run (#15).
- Restored the `load_mlx` seam in the server; ruff-clean JevBench letters
  runner.
- CLI warning noise on CPU and MPS suppressed (#3).

### Tests

- Recipe converters unit-tested (`tests/test_recipes.py`, 27 tests); data-build
  CLI entry points excluded from coverage as glue over tested parts.

[Unreleased]: https://github.com/KenSemerkant/strands-decider/compare/main@{latest}
