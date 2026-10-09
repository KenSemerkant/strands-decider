"""Steps for agent_gate.feature: the examples/strands before_tool_call flow."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "strands"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

# tool_call_intervention imports the strands SDK, which is not a dev dependency.
# The gate itself needs only the Guide/Proceed marker classes from it, so stand
# those in when the SDK is absent; the real classes win whenever it is installed.
try:
    import strands  # noqa: F401
except ModuleNotFoundError:
    strands = types.ModuleType("strands")
    interventions = types.ModuleType("strands.interventions")


    class InterventionHandler:  # noqa: D101
        pass


    class Guide:
        def __init__(self, feedback: str | None = None):
            self.feedback = feedback


    class Proceed:
        pass


    interventions.InterventionHandler = InterventionHandler
    interventions.Guide = Guide
    interventions.Proceed = Proceed
    strands.Agent = None
    strands.tool = lambda f: f
    strands.interventions = interventions
    sys.modules["strands"] = strands
    sys.modules["strands.interventions"] = interventions

from tool_call_intervention import Guide, Proceed, ToolCallReviewer  # noqa: E402

scenarios("features/agent_gate.feature")


@pytest.fixture
def gate():
    from tests.bdd.conftest import FakeDecider

    return ToolCallReviewer(FakeDecider())


class _Event:
    """Just the fields the gate reads: tool_use and the agent's messages."""

    def __init__(self):
        self.tool_use = {"name": "weather", "input": {"city": "Reykjavik"}}
        self.agent = SimpleNamespace(
            messages=[{"role": "user", "content": [{"text": "what is the weather in Reykjavik?"}]}]
        )


@given(parsers.parse("a gate with a decider answering {value:f}"))
def gate_answering(gate, value):  # noqa: ANN001
    gate._decider.answer = value
    return gate


@given(
    parsers.parse(
        "a gate with a decider answering grounded {grounded:f} and premature {premature:f}"
    )
)
def gate_answering_per_question(gate, grounded, premature):  # noqa: ANN001
    gate._decider.per_question = {"args_grounded": grounded, "premature": premature}
    return gate


@when("the agent proposes a tool call")
def propose(gate, response):  # noqa: ANN001
    response["out"] = gate.before_tool_call(_Event())


@then("the tool call is returned unchanged")
def unchanged(response):  # noqa: ANN001
    # The proceed branch returns Proceed(): the gate lets the call run as proposed.
    assert isinstance(response["out"], Proceed)
    assert not isinstance(response["out"], Guide)


@then("the action is a Guide")
def is_guide(response):  # noqa: ANN001
    assert isinstance(response["out"], Guide)


@then("the feedback tells the agent to confirm values with the user")
def guide_feedback(response):  # noqa: ANN001
    assert "confirm the values" in response["out"].feedback


@then("the feedback tells the agent to clarify with the user first")
def premature_feedback(response):  # noqa: ANN001
    assert "Clarify with the user first" in response["out"].feedback


@then("the rendered state contains the tool name and its arguments")
def state_contains(gate, response):  # noqa: ANN001
    state = gate._decider.states[0]
    assert "weather" in state and "Reykjavik" in state
