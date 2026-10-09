"""Steps for cli_ask.feature: the ask command against a stubbed engine."""

from __future__ import annotations

import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from typer.testing import CliRunner

from strands_decider import cli
from tests.bdd.conftest import StubEngine

scenarios("features/cli_ask.feature")


@pytest.fixture
def cli_result():
    return {}


@given("the engine loader returns a stub engine")
def stub_loader(monkeypatch):  # noqa: ANN001
    import strands_decider.infer as infer_mod

    monkeypatch.setattr(infer_mod, "load_engine", lambda *a, **k: StubEngine())


@when(parsers.parse("the user runs ask with --noul \"{question}\""))
def run_noul(cli_result, question):  # noqa: ANN001
    runner = CliRunner()
    cli_result["out"] = runner.invoke(
        cli.app, ["ask", "ckpt", "--state", "s", "--noul", question]
    )


@when(parsers.parse("the user runs ask with --choice \"{spec}\""))
def run_choice(cli_result, spec):  # noqa: ANN001
    runner = CliRunner()
    cli_result["out"] = runner.invoke(
        cli.app, ["ask", "ckpt", "--state", "s", "--choice", spec]
    )


@when("the user runs ask with no question flags")
def run_none(cli_result):  # noqa: ANN001
    runner = CliRunner()
    cli_result["out"] = runner.invoke(cli.app, ["ask", "ckpt", "--state", "s"])


@then(parsers.parse("the exit code is {code:d}"))
def exit_code(cli_result, code):  # noqa: ANN001
    assert cli_result["out"].exit_code == code, cli_result["out"].output


@then(parsers.parse("the output names the question {name}"))
def names(cli_result, name):  # noqa: ANN001
    assert name in cli_result["out"].output
