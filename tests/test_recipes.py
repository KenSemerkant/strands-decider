"""Unit tests for the recipe converters in data/recipes.py.

The converters are pure functions over an iterable of rows; only `build_recipe`
touches `datasets`, and it is tested against a fake split so nothing here needs
the network. These tests pin the invariants the corpus design depends on:
option subsampling keeps the gold option, noul order is always [false, true],
ordinal levels keep their order, and ambiguous labels are dropped not forced.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from strands_decider.data import recipes as R


class _Feat:
    def __init__(self, names: list[str] | None) -> None:
        self.names = names


class _FakeDS:
    """Just enough of a HF split: features, iteration, and the select chain."""

    def __init__(self, rows: list[dict], features: dict[str, _Feat] | None = None) -> None:
        self.rows = rows
        self.features = features or {}

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def filter(self, fn: Any) -> "_FakeDS":
        return _FakeDS([r for r in self.rows if fn(r)], self.features)

    def shuffle(self, seed: int | None = None) -> "_FakeDS":
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return _FakeDS(rows, self.features)

    def select(self, idx: list[int]) -> "_FakeDS":
        return _FakeDS([self.rows[i] for i in idx], self.features)


def _labelled(rows: list[dict], names: list[str]) -> _FakeDS:
    return _FakeDS(rows, {"label": _Feat(names)})


# ---- _subsample_options --------------------------------------------------------


def test_subsample_passthrough_when_within_budget() -> None:
    options = [["a", ""], ["b", ""]]
    out, gold = R._subsample_options(options, 1, max_options=4, rng=random.Random(0))
    assert out is options and gold == 1


def test_subsample_keeps_gold_and_stays_in_range() -> None:
    options = [[str(i), ""] for i in range(10)]
    for gold in (0, 5, 9):
        out, new_gold = R._subsample_options(options, gold, max_options=5, rng=random.Random(1))
        assert R.MIN_SUBSAMPLED_OPTIONS <= len(out) <= 5
        assert out[new_gold] == options[gold]


# ---- _clip / _humanise ---------------------------------------------------------


def test_clip_handles_none_short_and_long() -> None:
    assert R._clip(None, 10) == ""
    assert R._clip("  hi  ", 10) == "hi"
    long = "x" * 20
    cut = R._clip(long, 10)
    assert len(cut) <= 10 + 4 and cut.endswith(" ...")


def test_humanise_replaces_separators() -> None:
    assert R._humanise("top-hat_shop ") == "top hat shop"


# ---- choice converters ---------------------------------------------------------


def test_choice_from_labelled_maps_gold_and_humanises() -> None:
    ds = _labelled([{"text": "hello", "label": 1}], ["top_hat", "plain"])
    out = R._choice_from_labelled(
        ds, task="t", label_column="label", text_column="text",
        instructions="pick", max_options=8, max_chars=100, rng=random.Random(0),
    )
    assert len(out) == 1
    ex = out[0]
    assert ex.kind == "choice" and ex.label == 1 and ex.task == "t"
    assert ex.options[0][0] == "top hat"  # _humanise applied to the label name


def test_choice_from_labelled_text_fn_wins() -> None:
    ds = _labelled([{"title": "T", "body": "B", "label": 0}], ["a", "b"])
    out = R._choice_from_labelled(
        ds, task="t", label_column="label", text_fn=lambda r: r["title"] + ":" + r["body"],
        instructions="pick", max_options=8, max_chars=100, rng=random.Random(0),
    )
    assert out[0].state == "T:B"


def test_choice_from_labelled_rejects_featureless_column() -> None:
    ds = _FakeDS([{"text": "x", "label": 0}])
    with pytest.raises(ValueError, match="no class names"):
        R._choice_from_labelled(
            ds, task="t", label_column="label", text_column="text",
            instructions="pick", max_options=8, max_chars=100, rng=random.Random(0),
        )


def test_choice_from_string_labels_sorts_and_maps() -> None:
    ds = _FakeDS([{"text": "a", "lab": "zeta"}, {"text": "b", "lab": "alpha"}])
    out = R._choice_from_string_labels(
        ds, task="t", text_column="text", label_column="lab",
        instructions="pick", max_options=8, max_chars=100, rng=random.Random(0),
    )
    assert [ex.label for ex in out] == [1, 0]  # alpha < zeta after sorting
    assert out[0].options[0][0] == "alpha"


# ---- noul converters -----------------------------------------------------------


def _claim_rows() -> list[dict]:
    return [
        {"ev": "e0", "claim": "c0", "lab": "ENTAILMENT"},
        {"ev": "e1", "claim": "c1", "lab": "NOT ENTAILMENT"},
        {"ev": "e2", "claim": "c2", "lab": "UNKNOWN"},  # dropped, not forced
    ]


def test_noul_from_claim_evidence_drops_unknown_and_orders_options() -> None:
    out = R._noul_from_claim_evidence(
        _claim_rows(), task="t", evidence_column="ev", claim_column="claim",
        label_column="lab", true_values={"ENTAILMENT"}, false_values={"NOT ENTAILMENT"},
        templates=["is {claim} true?", "does it follow that {claim}?"],
        true_desc="yes desc", false_desc="no desc", max_chars=100, rng=random.Random(0),
    )
    assert sorted(ex.label for ex in out) == [0, 1]
    for ex in out:
        assert ex.kind == "noul"
        assert ex.options == [["false", "no desc"], ["true", "yes desc"]]
        assert ex.instructions != ex.instruction_variants[0]  # both template forms present


def test_noul_from_pair_formats_template_per_row() -> None:
    rows = [{"state": "s", "claim": "the sky is blue", "lab": 1}]
    out = R._noul_from_pair(
        rows, task="t", state_column="state", claim_column="claim", label_column="lab",
        template="is it true that {claim}?", true_label=1,
        true_desc="y", false_desc="n", max_chars=100,
    )
    assert out[0].instructions == "is it true that the sky is blue?"
    assert out[0].label == 1


def test_noul_from_binary_canonical_option_order() -> None:
    rows = [{"text": "a", "lab": 0}, {"text": "b", "lab": 2}]
    out = R._noul_from_binary(
        rows, task="t", text_column="text", label_column="lab",
        instructions="q", true_desc="y", false_desc="n", true_label=2, max_chars=100,
    )
    assert [ex.label for ex in out] == [0, 1]  # only lab == true_label is true
    assert all(ex.options[0][0] == "false" and ex.options[1][0] == "true" for ex in out)


def test_noul_from_entailment_drops_neutral() -> None:
    rows = [
        {"premise": "p0", "hypothesis": "h0", "label": 0},  # entailment -> true
        {"premise": "p1", "hypothesis": "h1", "label": 1},  # neutral -> dropped
        {"premise": "p2", "hypothesis": "h2", "label": 2},  # contradiction -> false
    ]
    out = R._noul_from_entailment(rows, task="t", instructions_template="{hypothesis}?", max_chars=100)
    assert [ex.label for ex in out] == [1, 0]
    assert out[0].state == "p0"


@pytest.mark.parametrize(
    "ans,expected",
    [(True, 1), (False, 0), ("TRUE", 1), ("1", 1), ("yes", 1), ("no", 0), ("maybe", 0)],
)
def test_noul_from_passage_question_answer_coercion(ans: Any, expected: int) -> None:
    rows = [{"p": "passage", "q": "does it hold", "a": ans}]
    out = R._noul_from_passage_question(
        rows, task="t", passage_column="p", question_column="q",
        answer_column="a", max_chars=100,
    )
    assert out[0].label == expected
    assert out[0].instructions == "does it hold?"  # '?' normalised


# ---- score converters ----------------------------------------------------------


def test_score_from_ordinal_keeps_order_and_skips_out_of_range() -> None:
    rows = [{"text": "a", "lab": 0}, {"text": "b", "lab": 2}, {"text": "c", "lab": 9}]
    out = R._score_from_ordinal(
        rows, task="t", text_column="text", label_column="lab",
        instructions="rate", levels=["low", "mid", "high"], max_chars=100,
    )
    assert [ex.label for ex in out] == [0, 2]  # 9 out of range -> dropped
    assert [o[0] for o in out[0].options] == ["0", "1", "2"]  # order never shuffled


def test_rank_levels_bins_by_quantile_and_drops_dead_zone() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    out = R._rank_levels(values, n_levels=2, dead_zone=0.0)
    assert out  # dead_zone 0 keeps everything near cuts
    levels = dict(out)
    assert levels[0] == 0 and levels[7] == 1  # extremes land in the extreme bins

    dropped = R._rank_levels(values, n_levels=2, dead_zone=1.0)
    assert len(dropped) < len(values)  # wide dead zone removes boundary points


def test_score_from_continuous_groups_and_caps() -> None:
    rows = [
        {"text": "same", "v": 0.0, "g": 1},
        {"text": "same", "v": 1.0, "g": 1},  # mean 0.5 with the row above
        {"text": "other", "v": 1.0, "g": 2},
    ]
    out = R._score_from_continuous(
        rows, task="t", text_column="text", value_column="v",
        instructions="rate", levels=["low", "high"], max_chars=100,
        target=10, rng=random.Random(0), group_column="g", dead_zone=0.0,
    )
    states = {ex.state for ex in out}
    assert states == {"same", "other"}  # duplicate rows collapsed to one state


def test_score_from_stars_balances_and_skips_invalid() -> None:
    rows = (
        [{"text": "a0", "stars": "1"}, {"text": "a1", "stars": "1"}]  # level 0 x2
        + [{"text": "b", "stars": "2"}] * 4  # level 1 x4
        + [{"text": "c0", "stars": "3"}, {"text": "c1", "stars": "3"}]  # level 2 x2
        + [{"text": "out", "stars": "9"}, {"text": "bad", "stars": "n/a"}]  # dropped
    )
    out = R._score_from_stars(
        rows, task="t", text_column="text", star_column="stars",
        instructions="rate", levels=["1", "2", "3"], max_chars=100,
        target=30, rng=random.Random(0),
    )
    # capped at the rarest level count (2), so the x4 bucket loses two rows
    labels = sorted(ex.label for ex in out)
    assert labels == [0, 0, 1, 1, 2, 2]


# ---- registry / build_recipe ---------------------------------------------------


def test_available_recipes_sorted_and_registered() -> None:
    names = R.available_recipes()
    assert names == sorted(R.SPECS) and names


def test_build_recipe_rejects_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_datasets(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not load")))
    with pytest.raises(KeyError, match="unknown recipe"):
        R.build_recipe("no_such_recipe")


def _fake_datasets(monkeypatch: pytest.MonkeyPatch, fake_load: Any) -> None:
    """Stand in for `datasets` (an optional dep): build_recipe imports it lazily."""
    import sys
    import types

    mod = types.ModuleType("datasets")
    mod.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_build_recipe_end_to_end_with_fake_split(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, Any] = {}

    def fake_load(path, config, split, trust_remote_code):  # noqa: ANN001
        loaded["path"] = path
        return _labelled([{"text": "hello", "label": 1}], ["a", "b"])

    _fake_datasets(monkeypatch, fake_load)
    out = R.build_recipe("ag_news", max_examples=2)
    assert loaded["path"]  # spec reached load_dataset
    assert all(ex.task == "ag_news" and ex.kind == "choice" for ex in out)
    # tasks with a pool get canonical instructions + variants attached here, not in the converter
    pool = R.INSTRUCTION_POOLS["ag_news"]
    assert all(ex.instructions == pool[0] and ex.instruction_variants == pool[1:] for ex in out)


def test_build_recipe_row_filter_selects_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = R.SPECS["ruletaker_d0"]
    assert spec.row_filter is not None
    ds = _FakeDS(
        [{"config": "depth-0", "label": 0, "text": "keep"}, {"config": "depth-3", "label": 0, "text": "drop"}],
        {"label": _Feat(["a", "b"])},
    ).filter(spec.row_filter)
    assert [r["text"] for r in ds] == ["keep"]
