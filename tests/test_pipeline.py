import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bart_zero_shot_classification_pipeline import (
    CONTRADICTION_INDEX,
    DECISION_RULE_MULTI,
    DECISION_RULE_SINGLE,
    DEFAULT_WEIGHTS_DIR,
    ENTAILMENT_INDEX,
    HYPOTHESIS_TEMPLATE,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    BARTZeroShotClassificationPipeline,
    accuracy,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
LABELS = ["travel", "cooking", "dancing"]


def _fake_runner(text: str, hypotheses: list[str]) -> tuple[np.ndarray, list[int]]:
    """Entailment logit grows with the hypothesis index, contradiction shrinks; 'cooking' wins if present."""
    logits = np.zeros((len(hypotheses), 3), dtype=np.float32)
    for i, hypothesis in enumerate(hypotheses):
        logits[i, CONTRADICTION_INDEX] = 1.0 - i
        logits[i, 1] = 0.0
        logits[i, ENTAILMENT_INDEX] = 4.0 if "cooking" in hypothesis else float(i)
    return logits, [len(text.split()) + len(h.split()) + 4 for h in hypotheses]


def _pipeline(runner=_fake_runner) -> BARTZeroShotClassificationPipeline:
    return BARTZeroShotClassificationPipeline(runner, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "facebook/bart-large-mnli"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert HYPOTHESIS_TEMPLATE.format("x") == "This example is x."


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    result = verify_snapshot(_write_snapshot(tmp_path).parent)
    assert result["revision"] == MODEL_REVISION and result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_size_missing_file_and_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        BARTZeroShotClassificationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_classify_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.classify(42, LABELS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.classify("   ", LABELS)
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.classify("a" * (MAX_TEXT_CHARS + 1), LABELS)
    with pytest.raises(TypeError, match="not a single string"):
        pipe.classify("text", "travel")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MAX_LABELS"):
        pipe.classify("text", [])
    with pytest.raises(ValueError, match="MAX_LABELS"):
        pipe.classify("text", [f"l{i}" for i in range(MAX_LABELS + 1)])
    with pytest.raises(TypeError):
        pipe.classify("text", ["ok", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="empty"):
        pipe.classify("text", ["ok", " "])
    with pytest.raises(ValueError, match="MAX_LABEL_CHARS"):
        pipe.classify("text", ["x" * (MAX_LABEL_CHARS + 1)])
    with pytest.raises(ValueError, match="unique"):
        pipe.classify("text", ["a", "a"])
    with pytest.raises(ValueError, match="placeholder"):
        pipe.classify("text", LABELS, hypothesis_template="no slot")
    with pytest.raises(TypeError, match="multi_label"):
        pipe.classify("text", LABELS, multi_label="yes")  # type: ignore[arg-type]


def test_classify_rejects_pairs_over_token_ceiling():
    def long_runner(text, hypotheses):
        return np.zeros((len(hypotheses), 3)), [MAX_TEXT_TOKENS + 1] * len(hypotheses)

    with pytest.raises(ValueError, match="MAX_TEXT_TOKENS"):
        _pipeline(long_runner).classify("text", LABELS)


def test_classify_single_label_mode_softmaxes_entailment_across_labels():
    result = _pipeline().classify("I will bake bread.", LABELS)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert result["multi_label"] is False and result["decision_rule"] == DECISION_RULE_SINGLE
    assert result["hypothesis_template"] == HYPOTHESIS_TEMPLATE
    assert [entry["label"] for entry in result["labels"]] == ["cooking", "dancing", "travel"]
    assert result["top_label"] == "cooking"
    scores = [entry["score"] for entry in result["labels"]]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1.0)
    assert set(result["labels"][0]) == {"label", "score", "entailment_logit", "contradiction_logit"}
    assert result["labels"][0]["entailment_logit"] == pytest.approx(4.0)
    assert result["n_tokens"] == 4 + 4 + 4  # words in text + longest hypothesis + specials


def test_classify_multi_label_mode_scores_each_label_independently():
    result = _pipeline().classify("I will bake bread.", LABELS, multi_label=True)
    assert result["multi_label"] is True and result["decision_rule"] == DECISION_RULE_MULTI
    scores = {entry["label"]: entry["score"] for entry in result["labels"]}
    # cooking: softmax([0, 4])[1]; travel: softmax([1, 0])[1]
    assert scores["cooking"] == pytest.approx(float(np.exp(4) / (1 + np.exp(4))))
    assert scores["travel"] == pytest.approx(float(1 / (1 + np.exp(1))))
    assert sum(scores.values()) > 1.0  # independent per-label probabilities do not sum to one


def test_classify_single_label_uses_the_per_label_rule():
    result = _pipeline().classify("I will bake bread.", ["cooking"])
    assert result["decision_rule"] == DECISION_RULE_MULTI
    assert 0.0 < result["labels"][0]["score"] < 1.0


def test_classify_rejects_backend_shape_mismatch():
    with pytest.raises(RuntimeError, match="expected"):
        _pipeline(lambda t, h: (np.zeros((len(h), 2)), [3] * len(h))).classify("text", LABELS)


def test_accuracy_helper():
    assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="same length"):
        accuracy(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        accuracy([], [])
