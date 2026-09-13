"""Zero-shot text classification by NLI entailment over the pinned ``facebook/bart-large-mnli``.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``classify``: the text is the NLI
premise, each caller-supplied label is turned into a hypothesis with ``HYPOTHESIS_TEMPLATE``, and the
entailment/contradiction logits are converted to one score per label (Yin et al., arXiv:1909.00161).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "facebook/bart-large-mnli"
MODEL_REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
MODEL_LICENSE = "mit"
MODEL_KEY = "bart-large-mnli"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Upstream hypothesis template (snapshot README "With manual PyTorch"): ``f'This example is {label}.'``.
HYPOTHESIS_TEMPLATE = "This example is {}."
# NLI head layout from the snapshot config.json ``label2id``: contradiction 0, neutral 1, entailment 2.
CONTRADICTION_INDEX = 0
ENTAILMENT_INDEX = 2
NUM_NLI_LABELS = 3
# Ceilings. 1024 is max_position_embeddings in config.json and model_max_length in tokenizer_config.json;
# a premise+hypothesis pair past it is rejected (not truncated) so a label is never scored on a cut premise.
MAX_TEXT_TOKENS = 1024
MAX_TEXT_CHARS = 8_000  # pre-tokenisation guard on the premise; ~4 chars per BPE token on English text
MAX_LABELS = 32  # hypotheses scored per classify() call (one forward pass, batched)
MAX_LABEL_CHARS = 100
DECISION_RULE_SINGLE = (
    "multi_label=False: softmax over the entailment logit across labels, argmax picks the label; "
    "no minimum score"
)
DECISION_RULE_MULTI = (
    "multi_label=True (or a single label): per label, softmax over [contradiction, entailment] logits, "
    "entailment probability is the score; no threshold applied"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _check_text(text: Any, name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError(f"{name} is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"{name} has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    return text


def _check_labels(labels: Any) -> list[str]:
    """``classify``'s label contract; raise naming the first violated ceiling."""
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a list of str, not a single string")
    if not 1 <= len(labels) <= MAX_LABELS:
        raise ValueError(f"labels must hold 1..MAX_LABELS={MAX_LABELS} items, got {len(labels)}")
    clean = []
    for i, label in enumerate(labels):
        if not isinstance(label, str):
            raise TypeError(f"labels[{i}] must be str, got {type(label).__name__}")
        if not label.strip():
            raise ValueError(f"labels[{i}] is empty")
        if len(label) > MAX_LABEL_CHARS:
            raise ValueError(f"labels[{i}] has {len(label)} chars; ceiling MAX_LABEL_CHARS={MAX_LABEL_CHARS}")
        clean.append(label)
    if len(set(clean)) != len(clean):
        raise ValueError("labels must be unique")
    return clean


def _check_template(template: Any) -> str:
    if not isinstance(template, str):
        raise TypeError("hypothesis_template must be str")
    if template.count("{}") != 1:
        raise ValueError("hypothesis_template must contain exactly one '{}' placeholder")
    return template


def _check_input_tokens(n_tokens: Sequence[int]) -> int:
    """The pair-token ceiling, applied once the tokenizer has counted every premise+hypothesis pair."""
    longest = max(int(n) for n in n_tokens)
    if longest > MAX_TEXT_TOKENS:
        raise ValueError(
            f"a premise+hypothesis pair is {longest} tokens; ceiling is MAX_TEXT_TOKENS={MAX_TEXT_TOKENS}"
        )
    return longest


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str (the NLI premise) and a list of unique non-empty str labels",
    "text_chars": [1, MAX_TEXT_CHARS],
    "pair_tokens": [1, MAX_TEXT_TOKENS],
    "labels": [1, MAX_LABELS],
    "label_chars": [1, MAX_LABEL_CHARS],
    "hypothesis_template": HYPOTHESIS_TEMPLATE,
    "multi_label": [False, True],
    "decision_rule": {"single": DECISION_RULE_SINGLE, "multi": DECISION_RULE_MULTI},
    "preprocessing": (
        "each label is inserted into hypothesis_template; every (premise, hypothesis) pair is BPE-encoded "
        "as one sequence with no truncation — a pair over MAX_TEXT_TOKENS is rejected, never cut"
    ),
}


def validate_inputs(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    multi_label: bool = False,
    hypothesis_template: str = HYPOTHESIS_TEMPLATE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    ``classify`` takes one text per call, so ``texts`` is the batch the notebook will loop over; every
    entry and the shared ``labels``/``hypothesis_template`` go through the same private checks the
    method uses (``_check_text``, ``_check_labels``, ``_check_template``), so a rejection here is a
    rejection there. ``MAX_TEXT_TOKENS`` needs the loaded tokenizer and is enforced inside ``classify``.
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str, not a single string")
    if not texts:
        raise ValueError("texts must hold at least one item")
    checked = [_check_text(text, f"texts[{i}]") for i, text in enumerate(texts)]
    clean_labels = _check_labels(labels)
    template = _check_template(hypothesis_template)
    if not isinstance(multi_label, bool):
        raise TypeError("multi_label must be a bool")
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"text-{i}", "chars": len(text)} for i, text in enumerate(checked)
        ],
        "labels": clean_labels,
        "hypotheses": [template.format(label) for label in clean_labels],
        "multi_label": multi_label,
        "hypothesis_template": template,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of items whose top label equals the gold label (exact string match)."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not gold:
        raise ValueError("accuracy needs at least one item")
    return float(sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold))


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    gold_labels: Sequence[str] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: ``accuracy`` of the top label against caller-supplied gold labels, else
    ``not-measurable``. The number is a sample-sanity observation on however many items were passed,
    never a benchmark; the score behind it is entailment-derived and not calibrated."""
    predicted = [str(result.get("top_label")) for result in results]
    supplied = gold_labels is not None
    report: dict[str, Any] = {
        "task": "zero-shot text classification by NLI entailment (caller-supplied label set)",
        "score_semantics": (
            "score is an entailment-derived softmax — over labels when multi_label=False, over "
            "[contradiction, entailment] per label otherwise — a ranking signal, not a calibrated "
            "probability; the decision rule is argmax over score and no threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_items": len(predicted),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": "no gold labels were supplied, so the top labels cannot be scored",
        "needs": (
            "one gold label per text from the deployment's own label set over enough texts to state a "
            "dispersion; the `accuracy` helper then scores exact top-label matches, and a calibration set "
            "is needed before any score is read as a probability"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if supplied:
        value = accuracy(predicted, list(gold_labels))
        report["metrics"] = [
            {
                "id": "accuracy",
                "value": value,
                "estimation": f"single pass over {len(predicted)} item(s); no dispersion",
                "decision_rule": "top_label == gold label (exact string match)",
            }
        ]
        report["verdict"] = "sample-sanity"
        report["reason"] = (
            f"gold labels were supplied for {len(predicted)} item(s); the accuracy is a plumbing check on "
            "that sample, not a benchmark"
        )
    return report


@dataclass
class BARTZeroShotClassificationPipeline:
    """``_runner(text, hypotheses)`` -> (NLI logits ``(n_hypotheses, 3)``, per-pair token counts). Injectable
    so tests run offline."""

    _runner: Callable[[str, list[str]], tuple[np.ndarray, list[int]]]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BARTZeroShotClassificationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), {"local_files_only": True}, "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, {}, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoTokenizer, BartForSequenceClassification

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            location, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = BartForSequenceClassification.from_pretrained(
            location, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(text: str, hypotheses: list[str]) -> tuple[np.ndarray, list[int]]:
            batch = tokenizer(
                [text] * len(hypotheses), hypotheses, return_tensors="pt", padding=True, truncation=False
            )
            n_tokens = [int(n) for n in batch["attention_mask"].sum(dim=1)]
            _check_input_tokens(n_tokens)
            with torch.inference_mode():
                logits = model(**batch.to(resolved_device)).logits
            return logits.float().cpu().numpy(), n_tokens

        return cls(runner, resolved_device, source)

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        multi_label: bool = False,
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
    ) -> dict[str, Any]:
        """Score every label against ``text``; ``labels`` in the result are sorted by descending score."""
        text = _check_text(text)
        clean = _check_labels(labels)
        template = _check_template(hypothesis_template)
        if not isinstance(multi_label, bool):
            raise TypeError("multi_label must be a bool")
        hypotheses = [template.format(label) for label in clean]
        logits, n_tokens = self._runner(text, hypotheses)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (len(clean), NUM_NLI_LABELS):
            raise RuntimeError(f"backend returned {logits.shape}, expected ({len(clean)}, {NUM_NLI_LABELS})")
        longest = _check_input_tokens(n_tokens)
        # Upstream ZeroShotClassificationPipeline rule: a single label always takes the per-label path.
        per_label = multi_label or len(clean) == 1
        if per_label:
            pair = logits[:, [CONTRADICTION_INDEX, ENTAILMENT_INDEX]]
            shifted = np.exp(pair - pair.max(axis=1, keepdims=True))
            scores = (shifted / shifted.sum(axis=1, keepdims=True))[:, 1]
        else:
            entail = logits[:, ENTAILMENT_INDEX]
            shifted = np.exp(entail - entail.max())
            scores = shifted / shifted.sum()
        order = np.argsort(-scores, kind="stable")
        ranked = [
            {
                "label": clean[i],
                "score": float(scores[i]),
                "entailment_logit": float(logits[i, ENTAILMENT_INDEX]),
                "contradiction_logit": float(logits[i, CONTRADICTION_INDEX]),
            }
            for i in order
        ]
        return {
            "labels": ranked,
            "top_label": ranked[0]["label"],
            "multi_label": multi_label,
            "hypothesis_template": template,
            "decision_rule": DECISION_RULE_MULTI if per_label else DECISION_RULE_SINGLE,
            "n_tokens": longest,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
