"""Steps for limits.feature: the rejection contract.

All three scenarios run against the `real_client` mount (conftest): the
slot-ceiling rejection happens inside SystemOneEngine.evaluate, which the
plain `client` route never reaches. Empty-criteria and state-type checks
are pydantic-level and would pass on either mount, but one mount keeps
the feature honest -- every 422 here comes from the real validation."""

from __future__ import annotations

from pytest_bdd import scenarios, given, when, then, parsers

scenarios("features/limits.feature")


@given("the server is mounted with a stub engine")
def the_server_is_mounted(real_client):  # noqa: ANN001 - fixture
    # "a stub engine": a weightless model behind the REAL engine and route.
    return real_client


@when("the client asks a choice question with 25 options")
def too_many_options(real_client, response):  # noqa: ANN001
    criteria = {f"opt_{i}": "" for i in range(25)}
    response["out"] = real_client.post(
        "/v1/systemone",
        json={
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": criteria}},
        },
    )


@when("the client posts a choice question with no criteria")
def no_criteria(real_client, response):  # noqa: ANN001
    response["out"] = real_client.post(
        "/v1/systemone",
        json={
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {}}},
        },
    )


@when("the client posts a request whose state is a number")
def state_number(real_client, response):  # noqa: ANN001
    response["out"] = real_client.post(
        "/v1/systemone",
        json={"state": 7, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    )


@then(parsers.parse("the response status is {code:d}"))
def status_is(response, code):  # noqa: ANN001
    assert response["out"].status_code == code
