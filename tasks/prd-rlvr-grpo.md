# PRD — RLVR with GRPO for strands-decider

Status: proposed (regenerated). Supersedes the deleted `tasks/` PRD for the same scope.
Grounded in code at commit `abd8b4a`; every `file.py:line` anchor below was verified
against the worktree at writing time.

## Background

strands-decider trains with supervised fine-tuning only: a label cross-entropy on the
option distribution (via `F.nll_loss` over `log_probs`, `src/strands_decider/modeling.py:479`)
plus two optional auxiliary terms — a distillation KL from a frozen teacher
(`src/strands_decider/train.py:526-533`, teacher distributions from
`src/strands_decider/data/teacher.py`) and a frozen-reference KL that pulls the trained
head toward the untouched torso's own option-number readout
(`src/strands_decider/train.py:491-524`, reference built by
`StrandsDeciderModel.frozen_slot_log_probs`, `src/strands_decider/modeling.py:412`,
or precomputed by `_frozen_reference`, `src/strands_decider/train.py:224`).

The architectural fact that makes RL cheap here: one forward pass over a question
yields **one categorical distribution over the K options** — `log_probs [B, K]`
(`src/strands_decider/modeling.py:468-469`; pointer-head branch
`src/strands_decider/modeling.py:454-460`, head at `src/strands_decider/modeling.py:124`).
There is no token generation. A GRPO "rollout" is therefore a categorical sample from
that distribution: G rollouts cost G draws of a single forward pass, not G decoding
loops. GRPO (Group Relative Policy Optimization) needs exactly this — a group of G
sampled answers per prompt, a verifiable reward (did the sample match the gold
label?), and an advantage computed relative to the group mean.

Everything RLVR needs already exists in the repo:

- **Gold labels**: `Example.label` is an index into the canonical option order
  (`src/strands_decider/data/format.py:40`, class at `:23`), validated at construction.
- **Accuracy as the reward oracle's definition**: `Prediction.correct`
  (`src/strands_decider/evaluate.py:97`) is `pred == label`, and `summarise` reports
  accuracy at `src/strands_decider/evaluate.py:289`.
- **A frozen reference for the KL anchor**: `frozen_slot_log_probs`
  (`src/strands_decider/modeling.py:412`) returns the adapter-disabled torso's option
  distribution at one extra forward and no extra weights.
- **A CLI pattern to match**: `strands-decider train`
  (`src/strands_decider/cli.py:154`) reads a YAML config into `TrainConfig`
  (`src/strands_decider/train.py:37`) and calls `train(cfg)`
  (`src/strands_decider/train.py:270`).

No preference data exists anywhere in the repo, which rules out RLHF by construction
and is precisely the situation RLVR is for.

## Goal

Close the SFT-only gap: add a GRPO-style RLVR training path that improves the
model's option-picking accuracy beyond what label CE reaches, measurable on JevBench
(not only on the repository's internal held-out sets), under the repository's
experiment rules (`CONTRIBUTING.md:104`, `:106`).

## Scope constraints (carried over from the prior, user-approved PRD)

- RLVR-only, using GRPO. Reward comes solely from the verifiable gold label.
- Entry point: a `strands-decider rl` CLI subcommand with the same UX as `train`
  (YAML config + option overrides). Default group size **G = 8**.
- Reuse existing machinery: `Example.label` for verification, the
  `evaluate.py` accuracy notion for the reward, and the frozen-reference KL builder
  (`frozen_slot_log_probs`) as the policy anchor.
- Hardware budget: 24 GB VRAM (single NVIDIA GPU; the reference SFT recipe runs at
  `micro_batch_size 8` / `max_length 3072`, `src/strands_decider/train.py:49,88`,
  on a Qwen3-1.7B-Base torso, `src/strands_decider/train.py:45`).
- Benchmark story: any claimed RLVR gain must be ranked on JevBench
  (`evaluation/jevbench/jevbench.sh`), per `CONTRIBUTING.md:106` and
  `evaluation/README.md:4` ("JevBench's 231 public tasks rank it"; the internal
  held-out sets "do not rank models", `evaluation/README.md:56-57`).

## Hard non-goals

- **RLHF** — no preference data exists; no reward model will be trained or used.
- **RLCD** — no contrastive-preference objectives.
- **MLX training loops** — training stays on the PyTorch path (Linux/WSL2 NVIDIA,
  per `CONTRIBUTING.md:106`); MLX remains inference-only (`mlx_engine.py`).
- **Token-level rollouts** — rollouts are categorical samples over K options from
  `log_probs`, never free-form generation or decoding loops.
- No changes to the serving API, schema, or data build.

## User stories

Each story is a vertically sliced, independently demoable unit. Story N does not
require story N+1 to demonstrate its own acceptance.

### S1 — Reward verifier

**As a** trainer, **I want** a verifier that scores a sampled option index against
the gold label, **so that** GRPO has a verifiable, binary reward with zero human
labeling.

- Input: an `Example` (`src/strands_decider/data/format.py:23`) and a sampled index.
- Reward definition mirrors `Prediction.correct` (`src/strands_decider/evaluate.py:97`):
  1.0 when `sampled == Example.label`, else 0.0.
- Demoable alone: unit test showing reward 1/0 on hand-built examples, including
  edge cases (label at index 0, label at the last option).

### S2 — Group sampler

**As a** trainer, **I want** to draw a group of G option samples per question from
the model's own `log_probs`, **so that** GRPO has on-policy rollouts.

- Samples come from the categorical defined by `out["log_probs"]`
  (`src/strands_decider/modeling.py:469`) — one forward, G draws; masked slots
  (past `n_slots`, `-inf` from `masked_log_softmax`,
  `src/strands_decider/modeling.py:210`) are never sampled.
- Default G = 8; G is configurable.
- Demoable alone: sampling from a fixed-seeded model on a handful of examples yields
  exactly G indices per question, all `< n_options`
  (`Example.n_options`, `src/strands_decider/data/format.py:56`), reproducible under
  a fixed torch seed.

### S3 — GRPO optimization step

**As a** trainer, **I want** a single GRPO update: group-normalized advantages,
a clipped policy-ratio objective, and a KL anchor to the frozen reference, **so
that** the policy improves on verifiable accuracy without drifting off the torso's
readout.

- Advantage per sample: `(r - mean(r_group)) / (std(r_group) + eps)` with std=0
  groups (all-correct or all-wrong) contributing zero advantage — those prompts are
  skipped, the standard GRPO degenerate-group rule.
- The KL anchor reuses `frozen_slot_log_probs` (`src/strands_decider/modeling.py:412`)
  exactly as the SFT loss does (`src/strands_decider/train.py:491-524`), restricted
  to eligible rows.
- Demoable alone: a synthetic distribution whose correct option is under-weighted
  moves toward it after one step, and a distribution whose group is all-correct does
  not move (zero advantage), verified in a unit test.

### S4 — `strands-decider rl` entrypoint and config

**As an** operator, **I want** an `rl` subcommand matching the `train` UX
(YAML config, key option overrides), **so that** RLVR runs are configured and
launched exactly like the SFT runs operators already know.

- Mirrors `train_cmd` (`src/strands_decider/cli.py:154-181`): `--config/-c`,
  `--train-file`, `--output-dir`, `--max-steps` overrides; error on missing
  training files, as `train` does.
- `RLConfig` covers: group size (default 8), RL learning rate, clipping epsilon,
  KL coefficient, rollouts retained per step, prompt batch size, and the fields
  needed to construct `TrainConfig`-compatible model loading (base model, head type,
  LoRA settings, `max_length`).
- Saves checkpoints with `save_pretrained` (`src/strands_decider/modeling.py:491`)
  so `eval`/`serve` load RL checkpoints unchanged.
- Demoable alone: `strands-decider rl --help` renders; `--max-steps 2` on a tiny
  JSONL produces a loadable checkpoint directory.

### S5 — Evaluation hook

**As an** evaluator, **I want** RL checkpoints to flow through the existing
evaluation path unchanged, **so that** RLVR progress is measured with the same
accuracy/ECE/NLL numbers as SFT.

- `strands-decider eval` (`src/strands_decider/cli.py:221`) /
  `evaluate_checkpoint` (`src/strands_decider/evaluate.py:390`) work on RL output
  directories with no code change beyond checkpoint compatibility.
- Demoable alone: running `eval` on the S4 smoke checkpoint returns the standard
  summary dict (accuracy present, `src/strands_decider/evaluate.py:289`).

### S6 — Bench integration

**As a** benchmark operator, **I want** the RL training path covered by the
benchmark suite's measured flows, **so that** RLVR does not silently regress
serving performance or misreport accuracy.

- RL checkpoints serve through the existing stack that `bench/bench_all.py`
  exercises (warm server, option-width scaling, concurrency) without new endpoints;
  the suite runs against an RL checkpoint as-is.
- `bench/jevbench_4b_letters.py` needs no change (it scores frozen Qwen, not the
  decider); it stays the frozen-baseline reference for comparison.
- Demoable alone: `bench/bench_all.py --url ...` against a server loaded with an RL
  checkpoint completes every section with numbers in the same ranges as the SFT
  checkpoint.

### S7 — End-to-end RLVR test

**As a** maintainer, **I want** an E2E test that runs a tiny GRPO run
(train-file in, checkpoint out, eval on), **so that** the whole path is
regression-protected in CI.

- Uses `--max-steps 2` (the same smoke valve `train` has,
  `src/strands_decider/train.py:115` and the `--max-steps` option at
  `src/strands_decider/cli.py:162`), a synthetic JSONL of a few dozen examples,
  CPU-or-single-GPU, minutes not hours.
- Asserts: run exits 0, checkpoint loads via `StrandsDeciderModel.load`, and
  `evaluate_checkpoint` returns a finite accuracy.
- Demoable alone: the test passes on a clean checkout with the `train` extra
  installed.

## Functional requirements

1. **FR1 — Reward**: `reward(example, sampled_index) ∈ {0.0, 1.0}` derived solely
   from `Example.label`; no model-in-the-loop, no human labels.
2. **FR2 — Sampling**: G categorical draws per prompt from `log_probs` honoring the
   `n_slots` mask; masked slots are unreachable; G defaults to 8 and is configurable.
3. **FR3 — Objective**: clipped policy-ratio GRPO loss with group-normalized
   advantages; degenerate groups (zero reward variance) are skipped, not
   zero-weighted through the loss.
4. **FR4 — KL anchor**: a coefficient-configurable KL to the frozen reference via
   `frozen_slot_log_probs` (`src/strands_decider/modeling.py:412`); setting the
   coefficient to 0 disables the term, mirroring `kl_frozen_weight`
   (`src/strands_decider/train.py:66`).
5. **FR5 — Entry point**: `strands-decider rl` with `--config/-c` YAML plus the
   same override options `train` offers; missing training files is a
   `BadParameter` error exactly as in `src/strands_decider/cli.py:180`.
6. **FR6 — Checkpoints**: RL output directories are loadable by
   `StrandsDeciderModel.load`, `strands-decider eval`, and `strands-decider serve`
   with no format changes.
7. **FR7 — Resource ceiling**: a default configuration runs within 24 GB VRAM on
   one NVIDIA GPU (batch geometry no larger than the SFT recipe's
   `micro_batch_size 8` × `max_length 3072`, `src/strands_decider/train.py:49,88`;
   G draws add negligible memory because they re-use one forward's `log_probs`).
8. **FR8 — Benchmark accounting**: any RLVR improvement claim ships with a JevBench
   ranking run (`evaluation/jevbench/jevbench.sh`) and, for A/B claims against the
   SFT reference, the paired McNemar test (`evaluation/jevbench/paired.py`), per the
   preregistration rule (`CONTRIBUTING.md:34`, `:106`; configs live under
   `configs/experiments/`).

## Success criteria

- The 7 stories each demoable as specified.
- An RLVR run that beats the SFT reference on JevBench by the preregistered margin;
  a run that misses its bar does not replace the reference model
  (`CONTRIBUTING.md:104`).

## Out of scope / risks

- Reward hacking is structurally impossible here (the reward is exact-match on a
  stored index), but training instability from early degenerate groups is a real
  risk; FR3's skip rule is the mitigation and S3's test is its guard.
- Score-kind questions ("score" ordinal levels) are treated as plain categorical
  options for reward purposes; ordinal smoothing and calibrated confidence remain
  inference-side (`src/strands_decider/evaluate.py:101-108`) and untouched.
