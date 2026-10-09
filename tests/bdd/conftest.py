"""Shared BDD fixtures: a stub engine behind the real routes, and a fake
Decider for the agent tool-gate flow. Everything runs offline on CPU."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strands_decider import server
from strands_decider.infer import _to_answer
from strands_decider.prompting import render_question
from strands_decider.schema import SystemOneRequest, SystemOneResponse, Usage


class _StubCfg:
    model_name = "strands-decider-test"
    device = "cpu"
    use_prefix_cache = True


class _StubModelCfg:
    base_model = "stub"
    num_slots = 24
    temperature = 1.0
    max_length = 512


class _StubModel:
    config = _StubModelCfg()


class StubEngine:
    """Fixed, concentrated distribution: deterministic answers, no weights."""

    cfg = _StubCfg()
    model = _StubModel()

    def evaluate(self, request):
        answers = {}
        for name, q in request.questions.items():
            rq = render_question(q)
            n = rq.n_slots
            probs = [0.7] + [0.3 / (n - 1)] * (n - 1)
            answers[name] = _to_answer(rq, probs)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=42, output_tokens=len(request.questions)),
        )

    def ask(self, state, questions):
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_engine", StubEngine())
    app = FastAPI(title="bdd")

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest):
        return server.get_engine().evaluate(request).model_dump()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": server.get_engine().cfg.model_name}

    return TestClient(app)


class _GuardModelCfg:
    base_model = "stub"
    num_slots = 24
    max_length = 512
    head_type = "fixed"


class _NoParamTorso:
    """A torso with no parameters: skips the CPU fp32 upcast in __init__."""

    def parameters(self):  # noqa: ANN201
        return iter([])


class _GuardModel:
    """Just enough of StrandsDeciderModel for SystemOneEngine.__init__ and the
    slot-ceiling guard in evaluate(); the guard fires before any torch call."""

    config = _GuardModelCfg()
    tokenizer = None
    torso = _NoParamTorso()

    def to(self, device):  # noqa: ANN001, ANN201
        return self

    def eval(self):  # noqa: ANN201
        return self


@pytest.fixture
def real_client(monkeypatch):
    """create_app mount with the REAL engine and route.

    The plain `client` mounts a bare route over StubEngine, which has no
    slot-ceiling check -- that guard lives in SystemOneEngine.evaluate
    (infer.py), and the route maps its ValueError to 422. Limits scenarios
    must exercise that real path, so we stub only the weights: the guard
    runs before any forward, and rejects oversized questions offline.
    """
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _GuardModel())
    )
    app = server.create_app("checkpoints/does-not-exist", device="cpu")
    return TestClient(app)


@pytest.fixture
def response():
    """Per-scenario scratch space shared between when and then steps."""
    return {}


class FakeDecider:
    """Stands in for examples/strands/_client.py Decider: ask() returns canned
    noul probabilities, chosen per scenario through its answer attribute, or
    per question through per_question (which wins for the names it lists)."""

    def __init__(self):
        self.answer: float = 1.0
        self.per_question: dict[str, float] = {}
        self.states: list[str] = []

    def ask(self, state, questions):
        self.states.append(state)
        return {name: {"noul": self.per_question.get(name, self.answer)} for name in questions}
