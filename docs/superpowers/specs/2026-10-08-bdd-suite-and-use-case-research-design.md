# BDD Suite + Production Use-Case Research — Design

Date: 2026-10-08
Status: approved (design); implementation pending

## Goal

Give strands-decider a behavioural (BDD) test layer over its user-facing
flows, and ground it in research: a `docs/use-cases.md` that identifies
practical, production-level uses for a small typed-classifier decision model,
with each surviving use case justifying the Gherkin scenarios that pin it.

## Decisions (settled in brainstorming)

| Decision | Choice |
| --- | --- |
| Scope | User-facing flows: HTTP API, CLI `ask`, agent `before_tool_call` example |
| Tool | `pytest-bdd` (new dev dependency) — runs inside the existing pytest suite |
| Engine under test | Stub engine, the `tests/test_server.py` `_StubEngine` pattern — offline, CPU, CI-safe (`HF_HUB_OFFLINE=1`) |
| Research deliverable | `docs/use-cases.md` + scenarios derived from its top use cases |
| Real-model flows | Out of scope — the existing gpu/mlx/distributed suites own them |

## Components

### 1. `docs/use-cases.md` (research output)

Web research into how small classifier/decision models are used in production
agent stacks: tool-call gating, LLM routing (RouteLLM-style), guardrail and
grounding checks, request triage, eval scoring. Each candidate use case is
mapped to strands-decider's real capabilities — noul/choice/score answers,
pointer readout over a 2B torso, local serving (~150 ms/question warm,
5.2× on repeated states) — with an honest fit verdict and sources. The top
4–6 use cases name which Gherkin scenarios they justify.

### 2. `tests/bdd/` (suite)

```
tests/bdd/
  conftest.py          # stub engine + TestClient + CliRunner fixtures
  features/
    serving.feature    # health, per-type ask, multi-question request, 422s
    limits.feature     # slot ceiling, window overflow, empty questions
    agent_gate.feature # examples/strands before_tool_call gating flow
    cli_ask.feature    # ask flags render to the same request shape as the API
  steps/               # one module per feature
```

- Collected by the existing `testpaths = ["tests"]`; no CI workflow change.
- Steps reuse the request/response types from `strands_decider.schema`; no
  re-implementation of the wire contract.
- Scenario names trace to `docs/use-cases.md` entries (the feature preamble
  names the use case each file serves).

## Error handling

Scenarios assert the existing user-visible error contract only: 422 status,
response body shape, message text, surviving fields. No new error paths are
invented by the BDD layer.

## Non-goals

- No real-model scenarios (existing gpu/mlx suites own those).
- No training or data-pipeline BDD (the unit suite covers the converters;
  entry points are pragma-excluded glue).
- No concurrency Gherkin — race #9 is already pinned by deterministic
  interleaving tests; a scenario layer would restate them, slower.
- No new runner: everything executes under plain `pytest -q`.

## Testing the tests

- Every scenario fails if the behaviour it names breaks.
- `pytest tests/bdd` completes in under ~5 s on CPU.
- Full suite (unit + BDD) stays green; CI line unchanged.

## Risks

- `pytest-bdd` adds a dev dependency — pinned minimum, installed only in the
  `dev` extra, matching how `pytest` itself is declared.
- Gherkin duplication of `tests/test_server.py` assertions — mitigated by
  scoping features to *flows a stakeholder reads* (gate a tool call, ask a
  mixed request), not field-by-field contract pins, which stay in test_server.
