"""Zero-shot text classification by NLI entailment over the pinned ``facebook/bart-large-mnli``.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``classify``: the text is the NLI
premise, each caller-supplied label is turned into a hypothesis with ``HYPOTHESIS_TEMPLATE``, and the
entailment/contradiction logits are converted to one score per label (Yin et al., arXiv:1909.00161).

The adaptation contract (``evaluate``, ``adapt``, ``save_artifact``, ``from_artifact``) fine-tunes the last
decoder blocks and the NLI classification head on a validated ``{id, text, label}`` dataset — every labelled
text becomes one entailment pair (its gold label's hypothesis) and one contradiction pair (a seeded wrong
label's hypothesis) — with validation-accuracy epoch selection, and exports the trained tensors as a
safetensors adapter bound to the pinned base weights. The inference contract above is unchanged by it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "cfbb687dbbd9df99fe865e1860350a22aebac4d26ee4bcb50217f1df606a018e"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 407_344_131
DECODER_LAYERS = 12  # config.json decoder_layers
DEFAULT_TRAINABLE_DECODER_LAYERS = 2  # the last two decoder blocks plus the NLI head (34,646,019 parameters)
MAX_TRAIN_PAIR_TOKENS = 256  # premise+hypothesis truncation ceiling during adaptation (never at inference)
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.bart-large-mnli.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

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

        return cls(runner, resolved_device, source, _model=model, _tokenizer=tokenizer)

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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        labels: Sequence[str] | None = None,
        *,
        multi_label: bool = False,
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
    ) -> dict[str, Any]:
        """Classify every record's text over `labels` (default: the dataset's own label vocabulary) and
        score the top labels against the gold labels (accuracy, macro-F1)."""
        from .metrics import classification_metrics
        from .samples import label_names, validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS, labels=labels)[
            "records"
        ]
        label_list = list(labels) if labels is not None else label_names(checked)
        started = time.perf_counter()
        predicted = [
            self.classify(
                r["text"], label_list, multi_label=multi_label, hypothesis_template=hypothesis_template
            )["top_label"]
            for r in checked
        ]
        metrics = classification_metrics(predicted, [r["label"] for r in checked])
        metrics.update(
            {
                "labels": label_list,
                "multi_label": multi_label,
                "hypothesis_template": hypothesis_template,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        if (
            not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_decoder_layers
        prefixes = tuple(f"model.decoder.layers.{k}." for k in range(first, DECODER_LAYERS)) + (
            "classification_head.",
        )
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        labels: Sequence[str] | None = None,
        epochs: int = 2,
        lr: float = 2e-5,
        batch_size: int = 16,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning of the NLI classifier on a validated labelled-text dataset.

        Every training record becomes two NLI pairs: (text, template(gold label)) labelled entailment and
        (text, template(a seeded wrong label from `labels`)) labelled contradiction. Only the last
        `trainable_decoder_layers` decoder blocks and the classification head train (2 blocks by default;
        the encoder, the embeddings and the earlier decoder blocks stay frozen). Cross-entropy over the three
        NLI logits, AdamW at a fixed learning rate with gradient clipping at 1.0, pairs truncated to
        MAX_TRAIN_PAIR_TOKENS **during training only**. Epoch 0 records the frozen model's validation
        accuracy; every epoch is scored on the validation split with `evaluate` under the same labels and
        template, and the epoch with the highest validation accuracy is kept."""
        from .samples import label_names, validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        template = _check_template(hypothesis_template)
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train, labels=labels)["records"]
        label_list = _check_labels(list(labels) if labels is not None else label_names(train_checked))
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS, labels=label_list)["records"]
            if val
            else []
        )
        import torch

        torch.manual_seed(seed)
        rng = random.Random(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked, label_list, hypothesis_template=template).items()
                if k in ("accuracy", "macro_f1", "n")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_acc = entry["val"]["accuracy"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                pairs: list[tuple[str, str, int]] = []
                for record in train_checked:
                    wrong = rng.choice([lab for lab in label_list if lab != record["label"]])
                    pairs.append((record["text"], template.format(record["label"]), ENTAILMENT_INDEX))
                    pairs.append((record["text"], template.format(wrong), CONTRADICTION_INDEX))
                order = torch.randperm(len(pairs), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [pairs[i] for i in order[start : start + batch_size]]
                    encoded = tokenizer(
                        [b[0] for b in batch],
                        [b[1] for b in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_TRAIN_PAIR_TOKENS,
                    )
                    out = model(
                        input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device),
                        labels=torch.tensor([b[2] for b in batch], device=device),
                    )
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["accuracy"] if entry["val"] else math.inf
                if current > best_acc or not entry["val"]:
                    best_acc = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation accuracy"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "max_train_pair_tokens": MAX_TRAIN_PAIR_TOKENS,
            "labels": label_list,
            "hypothesis_template": template,
            "pairs_per_record": 2,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder-block and head tensors as safetensors with a manifest naming the base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported
        format and version, the pinned base (id, revision, weight file, digest), exactly one file entry
        named `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_decoder_layers` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not
        authenticity against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_decoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int):
            raise ValueError("artifact manifest does not record an integer trainable_decoder_layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_decoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not (
                key.startswith("model.decoder.layers.") or key.startswith("classification_head.")
            ):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder or head tensor of the base model"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BARTZeroShotClassificationPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
