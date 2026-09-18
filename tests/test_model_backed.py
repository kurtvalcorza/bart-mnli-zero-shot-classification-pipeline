"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a labelled
evaluation, a one-epoch adaptation of the last decoder layer and the head on a dozen messages, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from bart_zero_shot_classification_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    WEIGHT_FILE,
    BARTZeroShotClassificationPipeline,
)

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

MESSAGES = [
    ("My new card has not arrived yet, when will it come?", "card arrival"),
    ("Where is the card I ordered last week?", "card arrival"),
    ("Can you tell me when my card will be delivered?", "card arrival"),
    ("Is the card already on its way to me?", "card arrival"),
    ("I lost my card on the bus, please block it.", "a lost or stolen card"),
    ("Someone stole my wallet with the card inside.", "a lost or stolen card"),
    ("My card was stolen, what should I do now?", "a lost or stolen card"),
    ("I cannot find my card anywhere, I think it is lost.", "a lost or stolen card"),
    ("What exchange rate do you use for euros?", "the exchange rate"),
    ("How is the exchange rate calculated when I pay abroad?", "the exchange rate"),
    ("Which rate applies when I exchange dollars?", "the exchange rate"),
    ("Is your exchange rate the interbank one?", "the exchange rate"),
]
RECORDS = [{"id": f"m{i:02d}", "text": t, "label": lab} for i, (t, lab) in enumerate(MESSAGES)]
LABELS = sorted({r["label"] for r in RECORDS})
TEMPLATE = "This customer message is about {}."


@pytest.fixture(scope="module")
def pipe():
    return BARTZeroShotClassificationPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_gold_labels(pipe):
    metrics = pipe.evaluate(RECORDS[:6], LABELS, hypothesis_template=TEMPLATE)
    assert metrics["n"] == 6 and metrics["accuracy"] > 50.0 and metrics["adapted"] is False


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(
        RECORDS[:9],
        RECORDS[9:],
        labels=LABELS,
        epochs=1,
        trainable_decoder_layers=1,
        batch_size=6,
        hypothesis_template=TEMPLATE,
    )
    assert result["n_trainable"] == 17_849_347 and result["history"][0]["note"] == "frozen model"
    assert all(
        name.startswith("model.decoder.layers.11.") or name.startswith("classification_head.")
        for name in result["trainable_names"]
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = BARTZeroShotClassificationPipeline.from_artifact(artifact, device="cpu")
    a = [pipe.classify(r["text"], LABELS, hypothesis_template=TEMPLATE)["top_label"] for r in RECORDS[:3]]
    b = [reloaded.classify(r["text"], LABELS, hypothesis_template=TEMPLATE)["top_label"] for r in RECORDS[:3]]
    assert a == b
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
