"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a labelled
evaluation, a one-epoch adaptation of the last decoder layer and the head on a dozen messages, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

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


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, tmp_path):
    """Without a validation split the recorded policy is "final epoch": the state after the last of three
    epochs is what stays in memory and what the artifact carries."""
    import torch

    result = pipe.adapt(RECORDS[:9], None, labels=LABELS, epochs=3, trainable_decoder_layers=1, batch_size=6)
    assert result["best_epoch"] == 3 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 4
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = BARTZeroShotClassificationPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 3 and reloaded.adapter["trainable_decoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    """The loader derives the exact tensor set from the recorded layer count: a manifest listing fewer,
    more or other tensors — or one whose safetensors payload differs from its list — is refused."""
    import json as _json
    import shutil

    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:9], None, labels=LABELS, epochs=1, trainable_decoder_layers=1, batch_size=6)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(_json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BARTZeroShotClassificationPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(_json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        BARTZeroShotClassificationPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_decoder_layers": 2}
    (other_layers / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BARTZeroShotClassificationPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside training leaves the base exactly as it was, frozen, with no adapter attached."""
    import torch

    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(
            RECORDS[:9],
            None,
            labels=LABELS,
            epochs=2,
            trainable_decoder_layers=1,
            batch_size=6,
            progress=boom,
        )
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())
