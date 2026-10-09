"""Steps for serving.feature: the one-shot ask flow over HTTP."""

from __future__ import annotations

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

scenarios("features/serving.feature")


@given("the server is mounted with a stub engine")
def the_server_is_mounted(client):  # noqa: ANN001 - fixture
    return client


@when("the client checks health")
def check_health(client, response):  # noqa: ANN001
    response["out"] = client.get("/health").json()


@when("the client asks \"is the sky blue?\" as a noul question")
def ask_noul(client, response):  # noqa: ANN001
    body = SystemOneRequest(
        state="the sky over Reykjavik",
        questions={"q": NoulQuestion(instructions="is the sky blue?")},
    )
    response["out"] = client.post(
        "/v1/systemone", json=body.model_dump()
    ).json()["answers"]["q"]


@when("the client asks a choice question with options \"red, green, blue\"")
def ask_choice(client, response):  # noqa: ANN001
    body = SystemOneRequest(
        state="a traffic light",
        questions={"q": ChoiceQuestion(
            instructions="what color is showing?",
            criteria={"red": "", "green": "", "blue": ""},
        )},
    )
    response["out"] = client.post(
        "/v1/systemone", json=body.model_dump()
    ).json()["answers"]["q"]


@when("the client asks a score question with levels \"low, mid, high\"")
def ask_score(client, response):  # noqa: ANN001
    body = SystemOneRequest(
        state="a calm support ticket",
        questions={"q": ScoreQuestion(
            instructions="how urgent is this?",
            criteria=["low", "mid", "high"],
        )},
    )
    response["out"] = client.post(
        "/v1/systemone", json=body.model_dump()
    ).json()["answers"]["q"]


@when("the client asks a mixed request of noul, choice and score")
def ask_mixed(client, response):  # noqa: ANN001
    body = SystemOneRequest(
        state="one state, three questions",
        questions={
            "n": NoulQuestion(instructions="grounded?"),
            "c": ChoiceQuestion(instructions="route?", criteria={"a": "", "b": ""}),
            "s": ScoreQuestion(instructions="severity?", criteria=["lo", "hi"]),
        },
    )
    response["out"] = client.post("/v1/systemone", json=body.model_dump()).json()


@when("the client posts a question of unknown type \"wat\"")
def post_unknown(client, response):  # noqa: ANN001
    response["out"] = client.post(
        "/v1/systemone",
        json={"state": "s", "questions": {"q": {"type": "wat", "instructions": "x"}}},
    )


@when("the client posts a request with no questions")
def post_empty(client, response):  # noqa: ANN001
    response["out"] = client.post(
        "/v1/systemone", json={"state": "s", "questions": {}}
    )


@then("the status is ok and the model is strands-decider-test")
def health_ok(response):  # noqa: ANN001
    assert response["out"]["status"] == "ok"
    assert response["out"]["model"] == "strands-decider-test"


@then("the answer type is \"noul\"")
def type_noul(response):  # noqa: ANN001
    assert response["out"]["type"] == "noul"


@then("the answer value is 0.3")
def value_true_slot(response):  # noqa: ANN001
    # Noul slots are canonically [false, true]; the stub concentrates 0.7 on
    # slot 0, so P(true) = the leftover mass = 0.3. The answer is P(true),
    # never P(argmax) -- that is the invariant under test.
    assert response["out"]["noul"] == pytest.approx(0.3)
    assert 0.0 <= response["out"]["noul"] <= 1.0


@then("the answer type is \"choice\"")
def type_choice(response):  # noqa: ANN001
    assert response["out"]["type"] == "choice"


@then("the picked option is the first criterion")
def choice_first(response):  # noqa: ANN001
    assert response["out"]["choice"] == "red"


@then("the answer type is \"score\"")
def type_score(response):  # noqa: ANN001
    assert response["out"]["type"] == "score"


@then("the answer carries a probability for each level")
def score_levels(response):  # noqa: ANN001
    # Levels are indexed, not named: probabilities are keyed "0","1","2" and
    # the legend explains what each index meant in the request's rubric.
    assert set(response["out"]["probabilities"]) == {"0", "1", "2"}
    assert response["out"]["legend"] == {"0": "low", "1": "mid", "2": "high"}


@then("each question name has a typed answer")
def mixed_typed(response):  # noqa: ANN001
    answers = response["out"]["answers"]
    assert {k: v["type"] for k, v in answers.items()} == {
        "n": "noul", "c": "choice", "s": "score",
    }


@then("usage counts one output token per question")
def mixed_usage(response):  # noqa: ANN001
    assert response["out"]["usage"]["output_tokens"] == 3


@then(parsers.parse("the response status is {code:d}"))
def status_is(response, code):  # noqa: ANN001
    assert response["out"].status_code == code
