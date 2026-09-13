"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from bart_zero_shot_classification_pipeline import (
    HYPOTHESIS_TEMPLATE,
    INPUT_SCHEMA,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

LABELS = ["travel", "cooking", "dancing"]
TEXTS = ["one day I will see the world", "I will bake bread tonight."]


def _result(top: str) -> dict:
    ranked = sorted(LABELS, key=lambda label: label != top)
    return {
        "labels": [{"label": label, "score": 0.9 if label == top else 0.05} for label in ranked],
        "top_label": top,
        "multi_label": False,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(TEXTS, LABELS, multi_label=True, names=["t1", "t2"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["pair_tokens"] == [1, MAX_TEXT_TOKENS]
    assert manifest["schema"]["labels"] == [1, MAX_LABELS]
    assert manifest["schema"]["label_chars"] == [1, MAX_LABEL_CHARS]
    assert manifest["inputs"] == [{"id": "t1", "chars": 28}, {"id": "t2", "chars": 26}]
    assert manifest["labels"] == LABELS
    assert manifest["hypotheses"] == [
        "This example is travel.",
        "This example is cooking.",
        "This example is dancing.",
    ]
    assert manifest["multi_label"] is True
    assert manifest["hypothesis_template"] == HYPOTHESIS_TEMPLATE
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_settings() -> None:
    manifest = validate_inputs(["A sentence."], LABELS)
    assert [entry["id"] for entry in manifest["inputs"]] == ["text-0"]
    assert manifest["multi_label"] is False
    custom = validate_inputs(["A sentence."], ["x"], hypothesis_template="About {}.")
    assert custom["hypotheses"] == ["About x."]


def test_validate_inputs_rejects_like_the_core_method() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs("a bare string", LABELS)
    with pytest.raises(ValueError, match="at least one"):
        validate_inputs([], LABELS)
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs(["  "], LABELS)
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)], LABELS)
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(TEXTS, "travel")
    with pytest.raises(ValueError, match=f"MAX_LABELS={MAX_LABELS}"):
        validate_inputs(TEXTS, [f"l{i}" for i in range(MAX_LABELS + 1)])
    with pytest.raises(ValueError, match="MAX_LABEL_CHARS"):
        validate_inputs(TEXTS, ["x" * (MAX_LABEL_CHARS + 1)])
    with pytest.raises(ValueError, match="unique"):
        validate_inputs(TEXTS, ["a", "a"])
    with pytest.raises(ValueError, match="placeholder"):
        validate_inputs(TEXTS, LABELS, hypothesis_template="no slot")
    with pytest.raises(TypeError, match="multi_label"):
        validate_inputs(TEXTS, LABELS, multi_label=1)
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs(TEXTS, LABELS, names=["only-one"])


def test_evaluation_report_not_measurable_without_gold() -> None:
    report = evaluation_report([_result("travel"), _result("cooking")])
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_items"] == 2
    assert report["sample_kind"] == "synthetic"
    assert "no gold labels" in report["reason"]
    assert "one gold label per text" in report["needs"]
    assert "not a calibrated probability" in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_accuracy_with_gold() -> None:
    report = evaluation_report(
        [_result("travel"), _result("cooking"), _result("dancing")],
        ["travel", "cooking", "travel"],
        sample_kind="BYOD upload",
    )
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD upload"
    assert report["n_items"] == 3
    assert len(report["metrics"]) == 1
    metric = report["metrics"][0]
    assert metric["id"] == "accuracy"
    assert metric["value"] == pytest.approx(2 / 3)
    assert "no dispersion" in metric["estimation"]
    assert "plumbing check" in report["reason"]


def test_evaluation_report_rejects_mismatched_gold_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluation_report([_result("travel")], ["travel", "cooking"])
