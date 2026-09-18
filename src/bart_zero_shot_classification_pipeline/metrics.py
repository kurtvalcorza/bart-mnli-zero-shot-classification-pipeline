"""Classification metrics over a labelled dataset (accuracy, macro-F1, per-label precision/recall/F1) and the
majority-class baseline.

Metrics are exact-match comparisons of the predicted top label against the gold label string; macro-F1
averages the per-label F1 over the label vocabulary of the gold labels (a label never predicted scores 0).
Both are reported in percent. Neither is a calibration measure: the score behind a top label is an
entailment-derived softmax, not a probability, and nothing here calibrates it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

METRIC_DEFINITIONS = {
    "accuracy": (
        "fraction of items whose predicted top label equals the gold label (exact string match); percent"
    ),
    "macro_f1": (
        "unweighted mean over the gold label vocabulary of the per-label F1 (precision and recall of "
        "predicting that label); a label never predicted contributes 0; percent"
    ),
}


def classification_metrics(predicted: Sequence[str], gold: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 and per-label counts over parallel predicted and gold labels."""
    if len(predicted) != len(gold):
        raise ValueError(f"{len(predicted)} predictions but {len(gold)} gold labels")
    if not gold:
        raise ValueError("no items to score")
    labels = sorted(set(gold))
    per_label = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold, strict=True) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "f1": 100.0 * f1,
        }
    correct = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g)
    return {
        "n": len(gold),
        "accuracy": 100.0 * correct / len(gold),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(per_label),
        "n_labels": len(labels),
        "per_label": per_label,
        "predicted_outside_gold_vocabulary": sum(1 for p in predicted if p not in per_label),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def majority_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict the most frequent gold label for every item (ties broken alphabetically)."""
    gold = [str(r["label"]) for r in records]
    counts = Counter(gold)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    result = classification_metrics([top] * len(gold), gold)
    result["baseline"] = f"majority class ({top!r} for every item)"
    return result
