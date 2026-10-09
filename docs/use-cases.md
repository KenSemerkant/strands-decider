# Production use cases

How a small typed-classifier decision model is used in real agent stacks,
and where strands-decider fits. Each surviving use case names the BDD
feature file that pins it (tests/bdd/features/).

## 1. Tool-call gating in agent loops
**Decision shape:** `noul` — one or two yes/no judgements about a pending
tool call ("are the arguments grounded in the conversation?", "is this call
premature?"), composed by the caller into run / guide / block.
**Production precedent:** Strands ships hooks that fire before each tool
call and can replace or refuse the invocation, with a documented example of
counting tool invocations per request and returning an error instead of
executing ([Strands Agents SDK hooks docs](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)).
Claude Code's `PreToolUse` hooks are the same pattern: they run synchronously,
block the tool until they finish, and can veto the call — handlers prevent
side effects rather than detect them ([Nader Dabit, deterministic control for agent workflows](https://nader.substack.com/p/agent-hooks-deterministic-control);
[Hooks: the enforcement layer that turns agent policy into code](https://ranjankumar.in/hooks-policy-as-code-agent-enforcement)).
NeMo Guardrails names this rail type explicitly: "execution rails" govern tool
and action calls, alongside input, dialog, retrieval and output rails
([NeMo Guardrails overview](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview)).
**Why strands-decider fits / does not:** Fits. The hook is synchronous, so
the gate's latency lands directly on every tool call; a 2B local model at
~150 ms/question warm (18.8 ms on a repeated state via the state cache)
stays inside a budget that a frontier-API call cannot. The typed yes/no
answer with a calibrated probability is exactly the run/guide/block signal,
and the repo's worked example (a `before_tool_call` gate on a weather tool
using two `noul` judgements) is this use case. It does not replace
policy-as-code checks (allowlists, rate limits) that should be deterministic.
**Pinned by:** `tests/bdd/features/agent_gate.feature`;
the latency and concurrency behaviour the gate depends on is pinned by
`tests/bdd/features/limits.feature`.

## 2. Model routing and cascades
**Decision shape:** `choice` — pick the handler for a request from a
caller-defined option list (weak model / strong model / deflect), optionally
with a `noul` "can the cheap path answer this?" first.
**Production precedent:** RouteLLM trains routers on preference data to send
easy queries to weak models, reporting up to 85% inference cost savings while
retaining ~95% of GPT-4 response quality ([RouteLLM paper, arXiv 2406.18665](https://arxiv.org/html/2406.18665v4)).
Production writeups of RouteLLM-style deployments repeat those numbers but
flag router calibration and the router's own latency overhead as the failure
modes ([RouteLLM in production: dynamic cascades](https://effloow.com/articles/routellm-hybrid-model-routing-cost-optimization-poc-2026);
[The router layer: semantic and cost-aware LLM routing](https://niteagent.com/blog/semantic-llm-routing-production)).
**Why strands-decider fits / does not:** Partial fit. RouteLLM's routers are
learned from preference data for a fixed model pair; strands-decider is
zero-shot — the option list is defined per request in the prompt, so
changing the model roster or adding a category needs no retraining, and the
calibrated confidence supports a "defer to the strong model when unsure"
cascade. It will not match a router trained on your traffic, and it is a
2B judge of difficulty, not a free regex: it earns its keep only where a
trained router does not exist or the roster changes often. Running locally
removes the router-latency-overhead failure mode the production writeups
describe.
**Pinned by:** `tests/bdd/features/serving.feature` (the one-POST contract —
mixed question types, one forward pass, usage tokens and measured latency in
every response — that a router sidecar consumes).

## 3. Input guardrails and content-policy classification
**Decision shape:** `noul` (is this input in policy? calibrated probability)
plus `choice` when the taxonomy has more than two outcomes.
**Production precedent:** NVIDIA ships a dedicated 4B classifier
(Nemotron-Content-Safety-Reasoning-4B) as a guardrail over prompts and
responses against a 22-category taxonomy
([model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B),
[deployment tutorial](https://docs.nvidia.com/nemo/guardrails/get-started/tutorials/nemotron-content-safety-reasoning-deployment)),
and NeMo Guardrails deployments pair the rail runtime with Llama Guard-class
binary safe/unsafe classifiers
([production deployment writeup](https://www.spheron.network/blog/nemo-guardrails-production-deployment-llm-gpu-cloud/)).
**Why strands-decider fits / does not:** Fits as a policy gate, not as a
safety model. Zero-shot label sets defined per request mean a new policy
category is a prompt edit, not a fine-tune — the opposite tradeoff to
Nemotron's baked taxonomy. But safety taxonomies are adversarially hardened
and benchmarked; strands-decider is not a safety-rated model and should not
be the only thing between users and harm. For internal content policy
(non-adversarial rules the operator defines), the calibrated probability and
~150 ms local latency fit well.
**Pinned by:** — (unit tests cover it).

## 4. LLM-as-judge scoring at volume
**Decision shape:** `score` — rate an output against a caller-supplied
rubric; expected value, legend and full distribution back.
**Production precedent:** Flow Judge is an open ~7B model built specifically
as a judge for LLM system evaluation ([Flow AI](https://flow-ai.com/blog/flow-judge)).
Snowflake's evaluation guidance recommends smaller judge models for
high-volume routine scoring to cut cost
([Snowflake LLM evaluation](https://www.snowflake.com/en/artificial-intelligence/natural-language-processing/large-language-models/llm-evaluation/)).
Two caveats recur: judge biases (position bias etc.) and that small judges
generalize worse to new scoring tasks than large ones
([Openlayer guide](https://www.openlayer.com/blog/llm-as-judge-evaluation-guide),
[Comet ML on LLM juries](https://www.comet.com/site/blog/llm-juries-for-evaluation/)).
Research is converging on reading evaluation verdicts off a small model's
internal representations instead of its generated text — INSPECTOR probes
small models to predict aspect-level scores ([arXiv 2601.22588](https://arxiv.org/abs/2601.22588)).
**Why strands-decider fits / does not:** Fits for rubric scoring, not for
open-ended judging. `score` rates against a rubric the caller supplies and
returns an expected value plus distribution with no token generation — the
same representation-reading idea INSPECTOR validates, already productized —
and the calibration makes the confidence a usable acceptance threshold. It
cannot write the critique a generative judge produces, and the
small-judges-generalize-worse caveat applies: validate the rubric against
labelled examples before trusting it blind.
**Pinned by:** — (unit tests cover it).

## 5. Triage classification from scripts and CI
**Decision shape:** mixed — a `choice` (which queue? which owner?) and a
`noul` (does this need a human now?) in one request.
**Production precedent:** Fine-tuned SLMs routinely beat frontier models on
narrow production classification at a fraction of the cost and latency
([Fastino, a guide to SLMs](https://fastino.ai/blog/a-guide-to-small-language-models));
production healthcare triage reports encoder-class models at 20–300 ms while
generative SLM calls run 1.5–3 s
([Innovaccer](https://innovaccer.com/gravity/blogs/specialized-small-language-models-for-healthcare-ai)).
The common shape is a pipeline step that classifies a document, log, ticket
or PR and routes it.
**Why strands-decider fits / does not:** Fits. The `strands-decider ask` CLI
makes the model usable from shell scripts, CI jobs and notebooks with no
server to run, one forward pass answers both questions, and zero-shot label
sets let a triage scheme change per run. It is a 2B judge of text it is
given — if triage needs retrieval or long context assembly, build that in
the script; the model only sees the state passed to it.
**Pinned by:** `tests/bdd/features/cli_ask.feature`.
