"""Offline tests for the labelled-text dataset contract, the pinned corpus reader, the classification
metrics and majority baseline, BYOD loaders, CSV export, artifact-manifest rejections and adapt() argument
validation. Nothing here imports torch or transformers; the corpus is two crafted CSV files served through
an injected fetcher."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from bart_zero_shot_classification_pipeline import (
    ARTIFACT_FORMAT,
    DECODER_LAYERS,
    LABEL_SET,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    BARTZeroShotClassificationPipeline,
    build_sample_dataset,
    check_split_disjoint,
    classification_metrics,
    dataset_digest,
    fetch_corpus,
    fetch_sample_dataset,
    filter_records,
    label_names,
    load_byod_dataset,
    majority_baseline,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from bart_zero_shot_classification_pipeline import pipeline as pl
from bart_zero_shot_classification_pipeline import samples as sm

INTENTS = list(LABEL_SET)  # the ten tutorial intents
PHRASES = list(LABEL_SET.values())
N_PER_INTENT = 4


def _rows(prefix):
    """Four texts per tutorial intent plus one text for each of 67 filler intents (77 in total)."""
    rows = []
    for intent in INTENTS:
        for k in range(N_PER_INTENT):
            rows.append(
                {"text": f"{prefix} message {k} about {intent.replace('_', ' ')}", "category": intent}
            )
    for k in range(sm.CORPUS_INTENTS - len(INTENTS)):
        rows.append({"text": f"{prefix} filler message {k}", "category": f"filler_{k}"})
    return rows


def _csv(rows):
    lines = ["text,category"] + [f'"{r["text"]}",{r["category"]}' for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _files():
    return {"train": _csv(_rows("train")), "test": _csv(_rows("test"))}


def _pin(monkeypatch, files):
    monkeypatch.setattr(
        sm, "CORPUS_FILES", {k: (f"{k}.csv", len(v), hashlib.sha256(v).hexdigest()) for k, v in files.items()}
    )
    monkeypatch.setattr(sm, "CORPUS_ROWS", {k: len(_rows(k)) for k in files})


def _records(prefix="r", n=12):
    return [
        {"id": f"{prefix}{i:03d}", "text": f"customer message number {i}", "label": PHRASES[i % 3]}
        for i in range(n)
    ]


def _echo_runner(text, hypotheses):
    """Fake NLI backend: the hypothesis sharing the most words with the text is entailed."""
    words = set(text.lower().split())
    logits = np.zeros((len(hypotheses), 3))
    for i, hyp in enumerate(hypotheses):
        overlap = len(words & set(hyp.lower().rstrip(".").split()))
        logits[i] = [1.0 - overlap, 0.0, float(overlap)]
    return logits, [len(text.split()) + len(h.split()) + 4 for h in hypotheses]


def _pipeline_without_model():
    return BARTZeroShotClassificationPipeline(_echo_runner)


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_BASE_URL.startswith(
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d"
    )
    assert {k: v[1] for k, v in sm.CORPUS_FILES.items()} == {"train": 839_073, "test": 239_961}
    assert all(len(v[2]) == 64 for v in sm.CORPUS_FILES.values())
    assert len(LABEL_SET) == 10 and len(set(LABEL_SET.values())) == 10
    assert sum(SAMPLE_SPLIT.values()) == 700 and all(v % 10 == 0 for v in SAMPLE_SPLIT.values())


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".csv")]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert len(calls) == 2 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_parses_and_checks_counts(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    assert len(corpus["train"]) == len(_rows("train")) and corpus["train"][0]["id"] == "train-00000"
    assert corpus["train"][0]["intent"] == INTENTS[0] and corpus["test"][0]["text"].startswith(
        "test message 0"
    )
    with pytest.raises(ValueError, match="missing the test file"):
        read_corpus({"train": files["train"]})
    monkeypatch.setattr(sm, "CORPUS_ROWS", {"train": 99, "test": len(_rows("test"))})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(files)


def test_filter_and_balanced_sample_split(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    kept = filter_records(corpus["train"])
    assert len(kept) == len(INTENTS) * N_PER_INTENT and {r["label"] for r in kept} == set(PHRASES)
    assert len(filter_records([*corpus["train"], {**corpus["train"][0], "id": "dup"}])) == len(kept)
    sizes = {"train": 20, "validation": 10, "test": 20}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert validate_dataset(splits["train"])["label_counts"] == {p: 2 for p in PHRASES}
    assert check_split_disjoint(splits) == sizes
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="not a multiple"):
        build_sample_dataset(corpus, sizes={"train": 15, "validation": 10, "test": 10})
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 40, "validation": 10, "test": 10})


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[url.rsplit("/", 1)[1].removesuffix(".csv")],
        sizes={"train": 20, "validation": 10, "test": 10},
    )
    assert validate_dataset(splits["train"])["n_records"] == 20


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert (
        report["n_records"] == 12 and report["unique_texts"] == 12 and report["labels"] == sorted(PHRASES[:3])
    )
    assert report["label_counts"] == {p: 4 for p in PHRASES[:3]}
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert label_names(_records()) == sorted(PHRASES[:3])
    good = _records()
    for bad, message, kwargs in (
        (good[:7], "8..20000", {}),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match", {}),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id", {}),
        ([{**good[0], "text": " "}, *good[1:]], "text is empty", {}),
        ([{**good[0], "text": "x" * 8_001}, *good[1:]], "MAX_TEXT_CHARS", {}),
        ([{**good[0], "label": ""}, *good[1:]], "non-empty string", {}),
        ([{**good[0], "label": "x" * 101}, *good[1:]], "MAX_LABEL_CHARS", {}),
        ([{"id": "a", "text": "b"}, *good[1:]], "missing 'label'", {}),
        (good, "not in the label set", {"labels": PHRASES[:2]}),
        (["not a mapping", *good[1:]], "must be a mapping", {}),
        ({"a": 1}, "must be a list", {}),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad, **kwargs)
    with pytest.raises(ValueError, match="at least 2 distinct"):
        label_names([{**r, "label": "same"} for r in good])


def test_split_dataset_is_stratified_and_seeded(forbid_model_imports):
    records = [*_records(n=30), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 30
    assert validate_dataset(splits["test"], min_records=1)["label_counts"] == {p: 2 for p in PHRASES[:3]}
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(_records(), val_fraction=0.0, test_fraction=0.9)


# --- metrics and baseline -----------------------------------------------------------------------------


def test_classification_metrics_and_majority_baseline(forbid_model_imports):
    gold = ["a", "a", "b", "c"]
    perfect = classification_metrics(gold, gold)
    assert perfect["accuracy"] == 100.0 and perfect["macro_f1"] == 100.0 and perfect["n_labels"] == 3
    partial = classification_metrics(["a", "a", "a", "z"], gold)
    assert partial["accuracy"] == 50.0 and partial["per_label"]["b"]["f1"] == 0.0
    assert partial["per_label"]["a"]["precision"] == pytest.approx(100.0 * 2 / 3)
    assert partial["predicted_outside_gold_vocabulary"] == 1
    with pytest.raises(ValueError, match="gold labels"):
        classification_metrics(["a"], ["a", "b"])
    majority = majority_baseline([{"label": g} for g in gold])
    assert majority["accuracy"] == 50.0 and "'a'" in majority["baseline"]


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "bad.csv").write_text("id,text\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="placeholder"):
        pipe.adapt(_records(), hypothesis_template="no placeholder")
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.adapt(_records(), trainable_decoder_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() only needs the classify path, so it works with an injected NLI backend
    records = [{"id": f"e{i}", "text": f"a message about {p}", "label": p} for i, p in enumerate(PHRASES)]
    metrics = pipe.evaluate(records, hypothesis_template="This is about {}.")
    assert metrics["n"] == 10 and metrics["labels"] == sorted(PHRASES) and metrics["adapted"] is False
    assert metrics["verdict"] == "measured-small-sample" and metrics["accuracy"] > 50.0
    with pytest.raises(ValueError, match="not in the label set"):
        pipe.evaluate(records, labels=PHRASES[:2])


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["classification_head.out_proj.weight"],
        "adapter": {},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
