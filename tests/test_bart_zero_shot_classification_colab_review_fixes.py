"""Regression tests for the 2026-10-02 notebook review of bart_zero_shot_classification_colab (ZSC-M1, ZSC-M2,
ZSC-m1..m5).

CI installs only pytest, ruff and numpy, so most tests use the carried modules with injected stand-ins: the
notebook's own Section 4 cell is executed from the notebook JSON with a fake upload and a whitespace "tokenizer".
The rerun journey (ZSC-M1) needs torch and transformers and a tiny randomly initialised BART built from the
snapshot's config with the snapshot's real tokenizer files; it is skipped where they are missing (as in CI). It
exercises the notebook's control flow, not BART-large-MNLI's numerics.
"""
# ruff: noqa: E501  -- notebook source literals are matched verbatim

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

from bart_zero_shot_classification_pipeline import (
    MIN_RECORDS_PER_LABEL,
    BARTZeroShotClassificationPipeline,
    byod_minimum_records,
    check_byod_tokens,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bart_zero_shot_classification_colab.ipynb"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in NB["cells"]]
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]
MODULE_CELLS = [i for i in CODE if NB["cells"][i]["metadata"].get("dimer", {}).get("embedded_module")]


def _cell(marker: str) -> int:
    found = [i for i in CODE if marker in CELLS[i]]
    assert len(found) == 1, (marker, found)
    return found[0]


SEC4 = _cell("USE_BYOD = False  # @param")
SEC6 = _cell("frozen_test = pipe.evaluate(")
SEC7 = _cell("adapt_result = pipe.adapt(")
SEC10 = _cell("MY_CATEGORY_ID = ''")
SEC11 = _cell("pipe.save_artifact(artifact_dir")


def _records(per_label: dict[str, int], long_words: dict[str, int] | None = None) -> list[dict]:
    out, k = [], 0
    for label, n in per_label.items():
        for j in range(n):
            text = f"message {k} about {label} number {j}"
            if long_words and long_words.get(label, 0) > j:
                # many short words: under the 8,000-character guard, over the token ceiling of a whitespace tokenizer
                text = f"m{k} " + " ".join(["w"] * long_words["words"])
            out.append({"id": f"r{k:05d}", "text": text, "label": label})
            k += 1
    return out


# ---- ZSC-M2: the BYOD split contract ---------------------------------------------------------------------


def test_stated_minimum_matches_the_split_arithmetic(forbid_model_imports):
    assert MIN_RECORDS_PER_LABEL == 4
    assert byod_minimum_records(2) == 12 and byod_minimum_records(10) == 40
    for n_labels in (2, 3, 10):
        minimum = byod_minimum_records(n_labels)
        split_dataset(_records({f"category {c}": minimum // n_labels for c in range(n_labels)}))
        with pytest.raises(ValueError, match=r"training split has \d+ records; at least 8|fewer than 4 distinct texts"):
            split_dataset(_records({f"category {c}": minimum // n_labels - 1 for c in range(n_labels)}))
    with pytest.raises(ValueError, match=r"training split has 6 records; at least 8 .*two balanced labels need at least 12"):
        split_dataset(_records({"category a": 5, "category b": 5}))


def test_twenty_record_two_label_file_is_accepted_with_every_label_in_every_split(forbid_model_imports):
    splits = split_dataset(_records({"category a": 10, "category b": 10}))
    for part in splits.values():
        assert {r["label"] for r in part} == {"category a", "category b"}


def test_a_label_too_small_to_reach_training_is_refused_by_name(forbid_model_imports):
    with pytest.raises(ValueError, match=r"fewer than 4 distinct texts.*'category rare': 1"):
        split_dataset(_records({"category a": 40, "category b": 40, "category rare": 1}))


def test_token_budget_refuses_over_ceiling_records_by_split_and_id_and_counts_training_cuts(forbid_model_imports):
    splits = {
        "train": [{"id": "t1", "text": "x " * 300, "label": "a"}, {"id": "t2", "text": "short", "label": "a"}],
        "validation": [{"id": "v1", "text": "fine", "label": "a"}],
        "test": [{"id": "s1", "text": "fine", "label": "a"}],
    }

    def count(text, label):
        n = len(text.split()) + 8
        return n, n

    budget = check_byod_tokens(splits, count, max_tokens=1024, train_max_tokens=256)
    assert budget == {"longest_pair_tokens": 308, "training_pairs_cut_at": 256, "training_records_cut": 1}
    splits["validation"].append({"id": "v-long", "text": "y " * 2700, "label": "a"})
    with pytest.raises(ValueError, match=r"validation:v-long \(2708 tokens\)"):
        check_byod_tokens(splits, count, max_tokens=1024, train_max_tokens=256)


# ---- ZSC-M1: adapt() refuses an adapted pipeline --------------------------------------------------------


def test_adapt_refuses_an_adapted_pipeline_before_any_model_import(forbid_model_imports):
    pipe = BARTZeroShotClassificationPipeline(lambda text, hyps: (None, []))
    pipe.adapter = {"trainable_decoder_layers": 2}
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt(_records({"a": 6, "b": 6}))


def test_count_pair_tokens_encodes_every_pair_without_truncation(forbid_model_imports):
    calls = []

    def tokenizer(texts, hypotheses, truncation):
        calls.append(truncation)
        return {"input_ids": [t.split() + h.split() for t, h in zip(texts, hypotheses, strict=True)]}

    pipe = BARTZeroShotClassificationPipeline(lambda text, hyps: (None, []), _model=object(), _tokenizer=tokenizer)
    counts = pipe.count_pair_tokens("one two three", ["x", "y z"], hypothesis_template="about {}")
    assert counts == {"x": 5, "y z": 6} and calls == [False]


def test_rerun_guards_are_in_the_notebook_cells():
    """Sections 6 and 7 reload the pretrained model before scoring / adapting; Section 6 refuses an adapted model;
    reload parity raises a message instead of a bare assert."""
    for index in (SEC6, SEC7):
        src = CELLS[index]
        assert src.index("reset_to_pretrained()\n") < src.index("pipe.evaluate(" if index == SEC6 else "pipe.adapt(")
    assert "if frozen_test['adapted']:\n    raise RuntimeError(" in CELLS[SEC6]
    assert "raise RuntimeError(f'Reload parity failed:" in CELLS[SEC11]
    for index in CODE:
        assert not re.search(r"^\s*assert ", CELLS[index], re.M), f"bare assert in cell {index}"
    closing = CELLS[-1]
    assert "set `TRAINABLE_DECODER_LAYERS = 1` in Section 7, select that cell and choose **Run after**" in closing
    assert "change `TEMPLATE` in Section 4" in closing and "re-run from that cell" not in "".join(CELLS)


# ---- ZSC-M2 / ZSC-m3: Section 4 and Section 11 executed from the notebook JSON ---------------------------


class _FakeFiles:
    payload: dict = {}

    @classmethod
    def upload(cls):
        return dict(cls.payload)


class _FakePipe:
    """Only the tokenizer-side call Section 4 makes: whitespace tokens + 4 per pair."""

    adapter = None

    def count_pair_tokens(self, text, labels, *, hypothesis_template):
        return {label: len(text.split()) + len(hypothesis_template.format(label).split()) + 4 for label in labels}


@pytest.fixture
def section4(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__main__", "os": os}
    for index in MODULE_CELLS:
        exec(compile(CELLS[index], f"<cell {index}>", "exec"), ns)
    ns["pipe"] = _FakePipe()

    def run(fields: dict, colab: bool = True):
        src = CELLS[SEC4]
        for name, value in fields.items():
            src, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m, v=value, k=name: f"{k} = {v!r}{m.group(1)}", src, flags=re.M)
            assert n == 1, name
        google = types.ModuleType("google")
        google.__path__ = []
        colab_mod = types.ModuleType("google.colab")
        colab_mod.files = _FakeFiles
        google.colab = colab_mod
        saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
        if colab:
            sys.modules["google"], sys.modules["google.colab"] = google, colab_mod
        else:
            sys.modules["google"] = types.ModuleType("google")
            sys.modules["google"].__path__ = []
            sys.modules.pop("google.colab", None)
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            exec(compile(src, f"<cell {SEC4}>", "exec"), ns)
        finally:
            sys.stdout = old
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        return ns, out.getvalue()

    return run


def _csv(records: list[dict]) -> bytes:
    rows = ["id,text,label"] + [f"{r['id']},\"{r['text']}\",{r['label']}" for r in records]
    return ("\n".join(rows) + "\n").encode("utf-8")


def test_section4_twenty_record_byod_file_passes_and_reports_its_token_budget(section4):
    _FakeFiles.payload = {"mine.csv": _csv(_records({"category a": 10, "category b": 10}))}
    ns, out = section4({"USE_BYOD": True})
    assert ns["disjoint"] == {"test": 4, "validation": 4, "train": 12}
    assert ns["token_budget"]["training_records_cut"] == 0 and "'token_budget'" in out


def test_section4_refuses_a_single_record_class_by_name(section4):
    _FakeFiles.payload = {"mine.csv": _csv(_records({"category a": 40, "category b": 40, "category rare": 1}))}
    with pytest.raises(ValueError, match="'category rare': 1"):
        section4({"USE_BYOD": True})


def test_section4_refuses_an_over_ceiling_record_by_id_and_counts_training_cuts(section4):
    long = _records({"category a": 30, "category b": 30}, {"category a": 1, "words": 2700})
    _FakeFiles.payload = {"mine.csv": _csv(long)}
    with pytest.raises(ValueError, match=r"(train|validation|test):r00000 \(27\d\d tokens\)"):
        section4({"USE_BYOD": True})
    cut = _records({"category a": 30, "category b": 30}, {"category a": 30, "words": 300})
    _FakeFiles.payload = {"mine.csv": _csv(cut)}
    ns, _out = section4({"USE_BYOD": True})
    assert ns["token_budget"]["training_records_cut"] == sum(r["label"] == "category a" for r in ns["train_records"])


def test_section4_empty_upload_and_non_colab_runtime_give_recovery_instructions(section4, tmp_path):
    _FakeFiles.payload = {}
    with pytest.raises(RuntimeError, match="Upload exactly one .csv, .json or .jsonl file .received 0.*BYOD_PATH"):
        section4({"USE_BYOD": True})
    with pytest.raises(RuntimeError, match="needs Google Colab.*BYOD_PATH"):
        section4({"USE_BYOD": True}, colab=False)
    path = tmp_path / "local.csv"
    path.write_bytes(_csv(_records({"category a": 6, "category b": 6})))
    ns, _out = section4({"USE_BYOD": True, "BYOD_PATH": str(path)}, colab=False)
    assert ns["data_source"] == "BYOD (local.csv)" and len(ns["train_records"]) == 8


def test_section11_says_byod_records_are_already_scored_test_records():
    md = CELLS[SEC11 - 1]
    assert "Under BYOD there is no pool of unused messages, so these are the first ten of your test records" in md
    assert "'BYOD test records (already scored in Section 8; not unseen messages)'" in CELLS[SEC11]


# ---- ZSC-m1: the isolated environment is the fleet's hash-locked managed-Python mechanism ---------------------


def test_install_cell_carries_the_lock_byte_for_byte_and_matches_the_pins():
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    lock = (ROOT / template["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, template), lock)
    install = CELLS[CODE[0]]
    literal = re.search(r"^LOCK_TEXT = r'''(.*?)'''$", install, re.M | re.S).group(1)
    assert literal == lock
    assert f"LOCK_SHA256 = {hashlib.sha256(lock.encode('utf-8')).hexdigest()!r}" in install
    assert f"LOCKED_PACKAGES = {len(build.lock_packages(lock))}" in install
    assert '"--managed-python", "--python", MANAGED_PYTHON' in install and "MANAGED_PYTHON = '3.12.12'" in install
    assert NB["cells"][CODE[0]]["metadata"].get("cellView") == "form"


def test_install_cell_refuses_non_linux_before_any_download(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    import platform
    import urllib.request

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("downloaded before the platform check"))
    with pytest.raises(RuntimeError, match="needs a Linux x86_64 runtime"):
        exec(compile(CELLS[CODE[0]], "<install>", "exec"), {"__name__": "__main__"})


# ---- ZSC-m2 / ZSC-m4 / ZSC-m5: learner-facing text ----------------------------------------------------------


def test_learner_text_has_no_stale_install_text_and_names_runtime_environments():
    markdown = "\n".join(CELLS[i] for i, c in enumerate(NB["cells"]) if c["cell_type"] == "markdown")
    for stale in ("installed directly — there is no repository clone", "stops with a restart instruction", "a dataset needs 8..20,000 records", "roughly ten minutes", "about a minute"):
        assert stale not in markdown, stale
    for figure in ("13.9 s", "48.1 s", "784 s", "320 s", "89 s"):
        assert figure in markdown, figure
    assert "Colab T4" in markdown and "CPU pre-flight" in markdown and "(an estimate)" in markdown


def test_guided_layer_has_troubleshooting_glossary_worked_answers_and_a_learner_change():
    markdown = "\n".join(CELLS[i] for i, c in enumerate(NB["cells"]) if c["cell_type"] == "markdown")
    assert "## Troubleshooting" in markdown and "## Glossary" in markdown
    for section_md in (SEC7 - 1, SEC7 + 1, SEC7 + 3):  # after the Section 6, 7 and 8 predictions
        assert "<details><summary>Check your reasoning</summary>" in CELLS[section_md], section_md
    assert "Predict → Change → Run → Observe → Explain" in CELLS[SEC10 - 1]
    assert "MY_DESCRIPTION = ''  # @param" in CELLS[SEC10] and "'learner_change': learner_change" in CELLS[SEC10]


# ---- ZSC-M1 journey on a tiny stand-in (torch + transformers only) ---------------------------------------


def _standin(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    snap = tmp_path / "weights" / "bart-large-mnli"
    snap.mkdir(parents=True)
    for name in ("README.md", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        shutil.copy2(ROOT / "weights" / "bart-large-mnli" / name, snap / name)
    cfg = transformers.BartConfig.from_pretrained(ROOT / "weights" / "bart-large-mnli")
    cfg.update({"d_model": 16, "encoder_ffn_dim": 32, "decoder_ffn_dim": 32, "encoder_attention_heads": 2,
                "decoder_attention_heads": 2, "encoder_layers": 1, "decoder_layers": 12, "dropout": 0.0})
    torch.manual_seed(0)
    transformers.BartForSequenceClassification(cfg).save_pretrained(snap, safe_serialization=True)
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(snap.iterdir()) if p.is_file()]
    manifest = {"format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": "bart-large-mnli", "modelId": "facebook/bart-large-mnli",
                "revision": "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce", "files": files, "totalBytes": sum(f["bytes"] for f in files)}
    (snap / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return torch, next(f["sha256"] for f in files if f["path"] == "model.safetensors")


def test_rerun_journeys_start_from_the_pretrained_model_on_a_tiny_standin(tmp_path, monkeypatch):
    """Default BYOD pass, then the two prescribed reruns: from Section 4 (TEMPLATE changed) and from Section 7
    (TRAINABLE_DECODER_LAYERS = 1). Stand-in model; labelled as such in the fix report."""
    torch, standin_sha = _standin(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIMER_NOTEBOOK_CI_PREINSTALLED", "1")
    ns: dict = {"__name__": "__main__", "os": os, "torch": torch, "display": print}
    for index in MODULE_CELLS:
        exec(compile(CELLS[index], f"<cell {index}>", "exec"), ns)
    ns["WEIGHT_SHA256"] = standin_sha  # stand-in substitution: the artifact boundary checks the base digest
    ns["WEIGHTS_DIR"] = ns["DEFAULT_WEIGHTS_DIR"]
    ns["fetched"] = []
    ns["snapshot"] = ns["verify_snapshot"](ns["WEIGHTS_DIR"])
    ns["pipe"] = ns["BARTZeroShotClassificationPipeline"].from_pretrained(weights_dir=ns["WEIGHTS_DIR"], device="cpu")
    ns.update({"NOTEBOOK_SOURCE": {"repository_revision": "test"}, "platform": __import__("platform"), "transformers": __import__("transformers")})
    data = tmp_path / "mine.csv"
    data.write_bytes(_csv(_records({"category a": 12, "category b": 12, "category c": 12})))

    def run(index, **fields):
        src = CELLS[index]
        for name, value in fields.items():
            src, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m, v=value, k=name: f"{k} = {v!r}{m.group(1)}", src, flags=re.M)
            assert n == 1, name
        exec(compile(src, f"<cell {index}>", "exec"), ns)

    byod = {"USE_BYOD": True, "BYOD_PATH": str(data)}
    fast = {"LEARNING_RATE": 1e-3, "EPOCHS": 1}
    base = ns["pipe"]._model.state_dict()
    base10 = base["model.decoder.layers.10.fc2.weight"].clone()
    for index in CODE[CODE.index(SEC4):]:
        run(index, **(byod if index == SEC4 else fast if index == SEC7 else {}))
    assert ns["frozen_test"]["adapted"] is False and ns["parity"]["identical_labels"] == ns["parity"]["of"]

    # Rerun from Section 4 with another template (the BYOD / template instructions).
    for index in (SEC4, CODE[CODE.index(SEC4) + 1], SEC6):
        run(index, **(dict(byod, TEMPLATE="This message is about {}.") if index == SEC4 else {}))
    fresh = ns["BARTZeroShotClassificationPipeline"].from_pretrained(weights_dir=ns["WEIGHTS_DIR"], device="cpu")
    reference = fresh.evaluate(ns["test_records"], ns["labels"], hypothesis_template=ns["TEMPLATE"])
    assert ns["frozen_test"]["adapted"] is False
    assert [p["predicted"] for p in ns["frozen_test"]["predictions"]] == [p["predicted"] for p in reference["predictions"]]
    run(SEC7, **fast)
    assert ns["adapt_result"]["history"][0]["val"] == {k: v for k, v in fresh.evaluate(ns["val_records"], ns["labels"], hypothesis_template=ns["TEMPLATE"]).items() if k in ("accuracy", "macro_f1", "n")}

    # Rerun from Section 7 with one trainable block: the artifact equals the evaluated model and parity holds.
    for index in CODE[CODE.index(SEC7):]:
        run(index, **(dict(fast, TRAINABLE_DECODER_LAYERS=1) if index == SEC7 else {}))
    assert torch.equal(ns["pipe"]._model.state_dict()["model.decoder.layers.10.fc2.weight"], base10)
    artifact = json.loads((Path("outputs/bart_zero_shot_classification_adapter") / "manifest.json").read_text(encoding="utf-8"))
    assert sorted({int(m.group(1)) for t in artifact["tensors"] for m in [re.match(r"model\.decoder\.layers\.(\d+)\.", t)] if m}) == [11]
    assert ns["parity"]["identical_labels"] == ns["parity"]["of"] and ns["parity"]["metrics_identical"]
