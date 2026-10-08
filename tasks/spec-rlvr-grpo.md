# SPEC — RLVR with GRPO for strands-decider

Companion to `tasks/prd-rlvr-grpo.md`. Implementation phases map 1:1 to the PRD's
user stories S1-S7. All `file.py:line` anchors verified at commit `abd8b4a`.

## Design decisions

| # | Decision | Rationale | Anchors |
|---|----------|-----------|---------|
| D1 | A rollout is a categorical sample over K options from one forward's `log_probs`, never token generation | The head emits `log_probs [B, K]` per question — one categorical; G rollouts cost G draws of one forward pass, so GRPO needs no decoding loop | `src/strands_decider/modeling.py:468-469` (out dict), `:454-460` (pointer branch), `:124` (`PointerHead`), `:154-159` (head forward `-> logits [B, K]`) |
| D2 | Reward is exact-match on `Example.label`; binary {0, 1} | Gold label is a canonical-order index validated at construction; verifiable by definition, no reward model, no preference data exists | `src/strands_decider/data/format.py:40` (`label: int`), `:23` (class), `src/strands_decider/evaluate.py:97` (`correct`) |
| D3 | GRPO with group-normalized advantage, clipped ratio, degenerate-group skip | Group-relative baseline needs no value head; all-correct/all-wrong groups carry zero gradient signal and are skipped | standard GRPO; group source = D1 sampling |
| D4 | KL anchor reuses the frozen-reference builder, not a second policy copy | `frozen_slot_log_probs` disables adapters and reads the torso's own option-number distribution — one extra forward, no extra weights; the SFT loss already consumes it the same way | `src/strands_decider/modeling.py:412-438`, `src/strands_decider/train.py:491-524` |
| D5 | New module `src/strands_decider/rl.py` + `rl` CLI subcommand; no changes inside `train.py` | Mirrors the repo's separation (train/evaluate/infer are siblings); keeps the SFT path untouched; the CLI pattern to copy is `train_cmd` | `src/strands_decider/cli.py:154-181`, `src/strands_decider/train.py:270` |
| D6 | Checkpoints written with `save_pretrained` | RL checkpoints must be loadable by `StrandsDeciderModel.load`, `eval`, and `serve` with zero format change | `src/strands_decider/modeling.py:491` |
| D7 | Batch geometry bounded by the SFT recipe (micro-batch 8 × `max_length` 3072) | 24 GB VRAM ceiling; G draws add negligible memory since they re-use the same `log_probs` tensor | `src/strands_decider/train.py:49,88`, torso default `:45` |
| D8 | Collation reuses `SystemOneCollator` | Rollouts must see prompts identical to SFT/serving (option shuffling, instruction variants, `opt_idx` for pointer heads) — the collator is that contract | `src/strands_decider/data/collate.py:42` (`SystemOneCollator`), `:162` (`__call__`) |
| D9 | Gains are ranked on JevBench with McNemar for A/B, preregistered first | Repository experiment rule; internal held-out sets explicitly do not rank models | `CONTRIBUTING.md:34,104,106`, `evaluation/README.md:4,56-57`, `evaluation/jevbench/jevbench.sh`, `evaluation/jevbench/paired.py`, `configs/experiments/` |

Rejected alternatives:

- **PPO**: needs a value network (extra weights, extra VRAM, extra tuning) for a
  problem where the group baseline is free.
- **DPO/RLCD/RLHF**: no preference data exists in the repo; synthesizing it is a
  different project.
- **MLX training loop**: training is the PyTorch/NVIDIA path; MLX is inference-only.
- **Token-level rollouts with the LM head**: the LM head is removed by design
  (`src/strands_decider/modeling.py:1-4`); the pointer/slot head is the policy.

## Data flow

```
data/*.jsonl
   │  load_examples (same loader as train)
   ▼
list[Example]                          label: int  (format.py:40)
   │  SystemOneCollator (collate.py:42,162)   -- shuffle, variants, opt_idx, n_slots
   ▼
batch {input_ids, attention_mask, n_slots, opt_idx, labels, ...}
   │  model.forward (modeling.py:440)
   ▼
log_probs [B, K]                       (modeling.py:469)
   │  sample G indices/question        -- S2 sampler, honors n_slots mask
   ▼
samples [B, G]
   │  reward vs Example.label          -- S1 verifier, {0.0, 1.0}
   ▼
rewards [B, G]
   │  group-normalize (skip std==0)    -- S3
   ▼
advantages [B, G]
   │  clipped ratio loss + β·KL(π ‖ frozen reference)
   │     ref = frozen_slot_log_probs (modeling.py:412)
   ▼
loss.backward() → optimizer.step()
   ▼
save_pretrained(out_dir)               -- S4 (modeling.py:491)
   ▼
strands-decider eval (cli.py:221) / evaluate_checkpoint (evaluate.py:390)   -- S5
   ▼
evaluation/jevbench/jevbench.sh + paired.py (McNemar)                       -- S6/S7 evidence
```

## Interfaces (signatures sketch)

New module `src/strands_decider/rl.py`. Signatures are sketches, not contracts;
names may flex during implementation but arity/roles are the design.

```python
@dataclass
class RLConfig:
    # data / model loading (TrainConfig-compatible subset)
    train_files: list[str]
    base_model: str = "Qwen/Qwen3-1.7B-Base"       # train.py:45 default
    init_checkpoint: str | None = None              # SFT checkpoint to start from
    head_type: str = "pointer"
    max_length: int = 3072                          # train.py:49 default
    use_lora: bool = True
    # GRPO
    group_size: int = 8                             # G (hard default from the PRD)
    prompts_per_step: int = 8                       # questions per optimizer step
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.05                          # 0 disables the anchor (FR4)
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    max_steps: int = 0                              # 0 = full pass; smoke valve (train.py:115)
    output_dir: str = "checkpoints/rl"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "RLConfig": ...


def reward(example: Example, sampled: int) -> float:
    """1.0 if sampled == example.label else 0.0 (S1). Mirrors evaluate.py:97."""


def sample_group(log_probs: torch.Tensor, n_slots: torch.Tensor,
                 group_size: int, generator: torch.Generator) -> torch.Tensor:
    """G categorical draws per row from log_probs [B, K] -> [B, G] (S2).

    Masked slots are -inf from masked_log_softmax (modeling.py:210-221) and are
    unreachable by construction; asserts output < n_slots per row.
    """


def grpo_loss(log_probs: torch.Tensor,          # current policy, [B, K], differentiable
              samples: torch.Tensor,            # [B, G], detached
              advantages: torch.Tensor,         # [B, G], group-normalized, std==0 rows zeroed
              old_log_probs: torch.Tensor,      # [B, G], behavior policy, detached
              ref_log_probs: torch.Tensor | None,  # frozen reference, [B, K], or None
              clip_epsilon: float,
              kl_coeff: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped-ratio GRPO objective + KL anchor (S3). Returns (loss, metrics)."""


def rl(cfg: RLConfig) -> str:
    """The training loop (S3/S4): collate -> forward -> sample -> reward ->
    advantage -> grpo_loss -> step; saves with save_pretrained (modeling.py:491)
    and returns the output directory, matching train()'s contract (train.py:270)."""
```

CLI (new block in `src/strands_decider/cli.py`, after `train_cmd` at `:154`):

```python
@app.command("rl", hidden=True)
def rl_cmd(
    config: str | None = typer.Option(None, "--config", "-c", ...),
    train_file: list[str] | None = typer.Option(None, "--train-file", ...),
    init_checkpoint: str | None = typer.Option(None, ...),
    output_dir: str | None = typer.Option(None, ...),
    max_steps: int | None = typer.Option(None, ...),
) -> None: ...
```

E2E test (S7), `tests/test_rl_e2e.py`:

```python
def test_rl_smoke(tmp_path):
    # tiny synthetic JSONL -> rl(cfg with max_steps=2) -> StrandsDeciderModel.load
    # -> evaluate_checkpoint returns finite accuracy (evaluate.py:289)
```

## Error handling

| Condition | Behavior | Precedent |
|----------|----------|-----------|
| No training files passed | `typer.BadParameter`, same message shape as train | `src/strands_decider/cli.py:180-181` |
| `Example.label` out of range | Impossible by construction (`__post_init__` raises); verifier never sees it | `src/strands_decider/data/format.py:49-52` |
| Degenerate group (all rewards equal) | Row masked out of the loss before backward; counted in a logged `skipped_groups` metric, never silent | FR3; fail-loud rule |
| Sampled index >= a row's `n_slots` | Cannot happen (`-inf` masking); `sample_group` asserts it anyway as a tripwire | `src/strands_decider/modeling.py:210-221` |
| `kl_coeff > 0` but reference row ineligible (option count beyond single-token digits) | Row excluded from the KL term exactly as SFT does; counted and logged | `src/strands_decider/modeling.py:412-438` (eligibility mask), `src/strands_decider/train.py:498-505` |
| `init_checkpoint` missing / unloadable | Fail before training starts with the loader's error; no partial output dir | `StrandsDeciderModel.load` contract |
| Non-finite loss / NaN gradients | Step skipped, warning logged with the offending batch's prompt ids; run continues (grad-clip already guards the SFT path) | `src/strands_decider/train.py:544-546` clip pattern |
| OOM | Not caught; the config geometry (FR7) is the guard, documented in `RLConfig` field comments | D7 |

## Implementation phases (1:1 with PRD stories)

| Phase | Story | Deliverable | Demo / verification |
|-------|-------|-------------|---------------------|
| P1 | S1 verifier | `reward()` + unit tests | Hand-built `Example`s: reward 1 at `label`, 0 elsewhere, index-0 and last-index labels |
| P2 | S2 sampler | `sample_group()` + unit tests | Fixed seed → exactly G indices/question, all `< n_options` (format.py:56), reproducible |
| P3 | S3 GRPO step | `grpo_loss()` + `rl()` loop skeleton | Under-weighted correct option moves toward correct; all-correct group does not move; KL anchor reduces drift vs no-anchor on a synthetic run |
| P4 | S4 entrypoint/config | `RLConfig`, `rl_cmd` in cli.py | `strands-decider rl --help`; `--max-steps 2` on tiny JSONL → loadable checkpoint dir |
| P5 | S5 eval hook | none beyond checkpoint compat | `strands-decider eval` on the P4 checkpoint returns the standard summary incl. accuracy |
| P6 | S6 bench | none beyond checkpoint compat (suite runs as-is) | `bench/bench_all.py` against a server with an RL checkpoint; all sections complete, ranges comparable to SFT |
| P7 | S7 E2E test | `tests/test_rl_e2e.py` | Test green in CI: rl(2 steps) → `Model.load` → `evaluate_checkpoint` finite accuracy |

Dependencies: P1, P2 independent of each other; P3 needs P1+P2; P4 needs P3;
P5, P6, P7 need P4. P5 and P6 are verification-only phases by design — D6 keeps
checkpoints format-compatible, so no eval/bench code should change.

## Open questions (to resolve in the preregistration, not in code)

1. Prompt curriculum: RLVR on the full train mix vs only the kinds where SFT
   accuracy lags (the per-task breakdown `summarise` provides,
   `src/strands_decider/evaluate.py:283`, is the diagnostic).
2. Whether to init from the current reference model (v19, `configs/train.yaml`)
   or its parent — decided by the preregistration's predictions, per
   `CONTRIBUTING.md:104`.
3. G > 8 for score-kind questions specifically (more options → more group
   variance); only if the default-G run clears its bar.
