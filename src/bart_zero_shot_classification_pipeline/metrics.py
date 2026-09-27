"""Classification metrics over a labelled dataset (accuracy, macro-F1, per-label precision/recall/F1,
confusion counts), the majority-class baseline, and paired comparisons of two systems on the same items.

Metrics are exact-match comparisons of the predicted top label against the gold label string. When the
caller declares the label vocabulary, every declared label is scored (a declared label with no gold support
scores F1 0 and is reported), so the macro average never silently drops a category; otherwise the vocabulary
is the gold labels. Both metrics are reported in percent. Neither is a calibration measure: the score behind a
top label is an entailment-derived softmax, not a probability, and nothing here calibrates it.
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
        "unweighted mean over the declared label vocabulary (the gold labels when none is declared) of the "
        "per-label F1 = 2TP / (2TP + FP + FN); a label never predicted correctly contributes 0; percent"
    ),
    "precision": (
        "TP / (TP + FP); undefined when the label was never predicted (TP + FP = 0) — reported as 0 with "
        "precision_defined = False and a note, never as a perfect score"
    ),
    "recall": "TP / (TP + FN); undefined when the label has no gold support (reported as 0 with a note)",
}


def classification_metrics(
    predicted: Sequence[str], gold: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-label counts and the confusion matrix over parallel predicted and gold labels.

    ``labels`` declares the vocabulary (its order is the confusion-matrix order); every gold label must be in
    it. Without it the vocabulary is the sorted gold labels.
    """
    if len(predicted) != len(gold):
        raise ValueError(f"{len(predicted)} predictions but {len(gold)} gold labels")
    if not gold:
        raise ValueError("no items to score")
    if labels is None:
        vocabulary = sorted(set(gold))
    else:
        vocabulary = list(dict.fromkeys(str(label) for label in labels))
        unknown = sorted(set(gold) - set(vocabulary))
        if unknown:
            raise ValueError(f"gold labels outside the declared vocabulary: {unknown}")
    per_label = {}
    for label in vocabulary:
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold, strict=True) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if not tp + fp:
            note = "never predicted: precision undefined, reported as 0"
        elif not tp + fn:
            note = "no gold support: recall undefined, reported as 0"
        else:
            note = None
        per_label[label] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "f1": 100.0 * f1,
            "precision_defined": bool(tp + fp),
            "recall_defined": bool(tp + fn),
            "note": note,
        }
    index = {label: i for i, label in enumerate(vocabulary)}
    matrix = [[0] * len(vocabulary) for _ in vocabulary]
    outside = 0
    for p, g in zip(predicted, gold, strict=True):
        if p in index:
            matrix[index[g]][index[p]] += 1
        else:
            outside += 1
    correct = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g)
    return {
        "n": len(gold),
        "accuracy": 100.0 * correct / len(gold),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(per_label),
        "n_labels": len(vocabulary),
        "per_label": per_label,
        "confusion": {
            "labels": vocabulary,
            "rows": "gold label",
            "columns": "predicted label",
            "matrix": matrix,
        },
        "predicted_outside_gold_vocabulary": outside,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def majority_baseline(
    records: Sequence[Mapping[str, Any]],
    *,
    train: Sequence[Mapping[str, Any]] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Predict one label for every item of ``records``: the most frequent gold label of ``train`` (of
    ``records`` itself when no training split is given). Ties are broken by alphabetical label order."""
    fit = train if train is not None else records
    counts = Counter(str(r["label"]) for r in fit)
    if not counts:
        raise ValueError("the majority baseline needs at least one labelled record to fit")
    top_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == top_count)
    top = tied[0]
    gold = [str(r["label"]) for r in records]
    result = classification_metrics([top] * len(gold), gold, labels)
    result["baseline"] = f"majority class ({top!r} for every item)"
    result["fitted_on"] = "training labels" if train is not None else "the scored records' own labels"
    result["tie_rule"] = (
        f"{len(tied)} labels share the top training count {top_count}; the alphabetically first is used"
        if len(tied) > 1
        else "no tie"
    )
    return result


def paired_changes(
    ids: Sequence[str], gold: Sequence[str], before: Sequence[str], after: Sequence[str]
) -> dict[str, Any]:
    """Compare two systems' predictions on the same items: corrected errors, newly introduced errors,
    unchanged correct, unchanged incorrect and incorrect-to-different-incorrect. Item order is preserved."""
    if not len(ids) == len(gold) == len(before) == len(after):
        raise ValueError("ids, gold, before and after must have the same length")
    groups: dict[str, list[str]] = {
        "corrected": [],
        "new_error": [],
        "unchanged_correct": [],
        "unchanged_incorrect": [],
        "changed_incorrect_to_incorrect": [],
    }
    for item, g, b, a in zip(ids, gold, before, after, strict=True):
        if b != g and a == g:
            groups["corrected"].append(item)
        elif b == g and a != g:
            groups["new_error"].append(item)
        elif b == g:
            groups["unchanged_correct"].append(item)
        elif b == a:
            groups["unchanged_incorrect"].append(item)
        else:
            groups["changed_incorrect_to_incorrect"].append(item)
    changed = sum(1 for b, a in zip(before, after, strict=True) if b != a)
    return {
        "n": len(ids),
        "changed_predictions": changed,
        "changed_fraction": changed / len(ids) if ids else 0.0,
        "counts": {name: len(members) for name, members in groups.items()},
        "ids": groups,
    }


def confusion_markdown(metrics: Mapping[str, Any], title: str = "") -> str:
    """A markdown table of a ``classification_metrics`` confusion matrix (rows gold, columns predicted)."""
    labels = metrics["confusion"]["labels"]
    matrix = metrics["confusion"]["matrix"]
    header = " | ".join(f"P{i + 1}" for i in range(len(labels)))
    lines = [f"**{title}**", ""] if title else []
    lines.append(f"| gold \\ predicted | {header} | support |")
    lines.append("|---|" + "---:|" * (len(labels) + 1))
    for i, label in enumerate(labels):
        cells = " | ".join(str(v) for v in matrix[i])
        lines.append(f"| P{i + 1} {label} | {cells} | {sum(matrix[i])} |")
    return "\n".join(lines)
