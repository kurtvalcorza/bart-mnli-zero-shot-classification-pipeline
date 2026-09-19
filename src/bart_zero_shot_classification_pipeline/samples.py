"""Labelled-text dataset contract for adapting the zero-shot classifier: the pinned Banking77 sample,
validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and out of the NLI model's domain: Banking77 (Casanueva et al., 2020;
CC BY 4.0), 13,083 customer-support queries labelled with 77 fine-grained banking intents. Two CSV files
(`train.csv`, `test.csv`) are fetched from the PolyAI `task-specific-datasets` repository at a pinned
commit and refused on any byte-size or SHA-256 mismatch. The tutorial keeps ten intents (`LABEL_SET`) with
a readable phrase per intent so the zero-shot hypothesis reads as English ("This customer message is about a
lost or stolen card."); training and validation queries are drawn from `train.csv`, test queries from
`test.csv` — the release's own partition.

A record is ``{id, text, label}``: a customer message and the phrase of its gold intent.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_LABEL_CHARS, MAX_LABELS, MAX_TEXT_CHARS, MODEL_ID

CORPUS_NAME = "Banking77"
CORPUS_RELEASE = "PolyAI-LDN/task-specific-datasets @ 57ec275d8078af65b7731c2a98be812d844a6d6b"
CORPUS_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
)
CORPUS_FILES = {
    "train": ("train.csv", 839_073, "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test": ("test.csv", 239_961, "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
}
CORPUS_LICENSE = "CC BY 4.0 (Casanueva et al. 2020; PolyAI-LDN/task-specific-datasets)"
CORPUS_ROWS = {"train": 10_003, "test": 3_080}
CORPUS_INTENTS = 77
DEFAULT_CACHE_DIR = Path("weights") / "banking77"
# The ten intents the tutorial keeps, each with the phrase the hypothesis template receives.
LABEL_SET: dict[str, str] = {
    "card_arrival": "card arrival",
    "lost_or_stolen_card": "a lost or stolen card",
    "exchange_rate": "the exchange rate",
    "change_pin": "changing the PIN",
    "declined_card_payment": "a declined card payment",
    "transfer_not_received_by_recipient": "a transfer not received by the recipient",
    "atm_support": "ATM support",
    "age_limit": "the age limit",
    "terminate_account": "closing the account",
    "top_up_failed": "a failed top-up",
}
DEFAULT_HYPOTHESIS_TEMPLATE = "This customer message is about {}."
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 400, "validation": 100, "test": 200}  # balanced over the ten intents
MIN_RECORDS = 8
MAX_RECORDS = 20_000
MIN_LABEL_KINDS = 2
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return the two pinned Banking77 CSV files (bytes) from the cache or the repository, verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, size, digest) in CORPUS_FILES.items():
        local = cache / name
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = CORPUS_BASE_URL + name
            if fetcher is not None:
                data = fetcher(url)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    """Parse the CSV members (columns `text`, `category`) into flat records keeping the raw intent name."""
    out = {}
    for split in CORPUS_FILES:
        if split not in files:
            raise ValueError(f"corpus is missing the {split} file")
        rows = list(csv.DictReader(io.StringIO(files[split].decode("utf-8"))))
        if not rows or {"text", "category"} - set(rows[0]):
            raise ValueError(f"{split}: expected columns text and category")
        if len(rows) != CORPUS_ROWS[split]:
            raise ValueError(f"{split}: {len(rows)} rows, expected {CORPUS_ROWS[split]}")
        out[split] = [
            {"id": f"{split}-{i:05d}", "text": r["text"].strip(), "intent": r["category"].strip()}
            for i, r in enumerate(rows)
        ]
        intents = {r["intent"] for r in out[split]}
        if len(intents) != CORPUS_INTENTS:
            raise ValueError(f"{split}: {len(intents)} intents, expected {CORPUS_INTENTS}")
    return out


def filter_records(
    records: Sequence[Mapping[str, Any]], *, label_set: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    """Keep records whose intent is in the label set, mapped to its phrase; drop repeated texts."""
    label_set = dict(label_set or LABEL_SET)
    seen: set[str] = set()
    kept = []
    for record in records:
        intent = str(record.get("intent", record.get("label", "")))
        if intent not in label_set:
            continue
        text = str(record["text"]).strip()
        key = text.lower()
        if not text or key in seen or len(text) > MAX_TEXT_CHARS:
            continue
        seen.add(key)
        kept.append({"id": record["id"], "text": text, "label": label_set[intent], "intent": intent})
    return kept


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    label_set: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Balanced seeded draws: training and validation from `train` (disjoint texts), test from `test`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    label_set = dict(label_set or LABEL_SET)
    n_labels = len(label_set)
    for name, size in sizes.items():
        if size % n_labels:
            raise ValueError(f"{name} size {size} is not a multiple of the {n_labels} labels")
    rng = random.Random(seed)
    pools = {
        "train": filter_records(corpus["train"], label_set=label_set),
        "test": filter_records(corpus["test"], label_set=label_set),
    }
    by_label = {
        split: {phrase: [r for r in pool if r["label"] == phrase] for phrase in label_set.values()}
        for split, pool in pools.items()
    }
    for split in by_label.values():
        for records in split.values():
            rng.shuffle(records)
    cursor = {phrase: 0 for phrase in label_set.values()}
    out: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        source = "test" if name == "test" else "train"
        per_label = size // n_labels
        picked = []
        for phrase in label_set.values():
            pool = by_label[source][phrase]
            start = cursor[phrase] if source == "train" else 0
            chunk = pool[start : start + per_label]
            if len(chunk) < per_label:
                raise ValueError(
                    f"{name}: only {len(chunk)} records available for {phrase!r}, need {per_label}"
                )
            picked.extend(chunk)
            if source == "train":
                cursor[phrase] = start + per_label
        rng.shuffle(picked)
        out[name] = [
            {"id": f"{name}-{i:04d}", "text": r["text"], "label": r["label"], "intent": r["intent"]}
            for i, r in enumerate(picked)
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/text/label")
    for key in ("id", "text", "label"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, text, label = record["id"], record["text"], record["label"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if not isinstance(text, str):
        raise ValueError(f"{label_name}: text must be a string")
    if not text.strip():
        raise ValueError(f"{label_name}: text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"{label_name}: text has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}"
        )
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{label_name}: label must be a non-empty string")
    if len(label) > MAX_LABEL_CHARS:
        raise ValueError(
            f"{label_name}: label has {len(label)} chars; ceiling is MAX_LABEL_CHARS={MAX_LABEL_CHARS}"
        )
    item = {"id": rid, "text": text.strip(), "label": label.strip()}
    if "intent" in record:
        item["intent"] = str(record["intent"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Structural validation of a labelled-text dataset; raises ValueError before any model import.
    With `labels`, every record's label must be one of them."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, text, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    allowed = set(labels) if labels is not None else None
    checked = []
    ids: set[str] = set()
    texts: set[str] = set()
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        if allowed is not None and item["label"] not in allowed:
            raise ValueError(f"records[{index}]: label {item['label']!r} is not in the label set")
        ids.add(item["id"])
        texts.add(item["text"].lower())
        counts[item["label"]] = counts.get(item["label"], 0) + 1
        checked.append(item)
    if len(counts) > MAX_LABELS:
        raise ValueError(f"{len(counts)} distinct labels; ceiling is MAX_LABELS={MAX_LABELS}")
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_texts": len(texts),
        "labels": sorted(counts),
        "label_counts": dict(sorted(counts.items())),
        "text_chars": {
            "min": min(len(r["text"]) for r in checked),
            "max": max(len(r["text"]) for r in checked),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["text"], r["label"]] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def label_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted label vocabulary of a dataset (the `labels` argument for `classify`)."""
    names = sorted({str(r["label"]) for r in records})
    if len(names) < MIN_LABEL_KINDS:
        raise ValueError(f"a dataset needs at least {MIN_LABEL_KINDS} distinct labels")
    return names


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased text appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["text"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(f"text {record['text'][:60]!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified split of a BYOD dataset into train/validation/test after de-duplicating texts."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = record["text"].lower()
        if key not in seen:
            seen.add(key)
            by_label.setdefault(record["label"], []).append(record)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction))
        n_val = round(len(group) * val_fraction)
        splits["test"].extend(group[:n_test])
        splits["validation"].extend(group[n_test : n_test + n_val])
        splits["train"].extend(group[n_test + n_val :])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, text, label}` records from CSV (columns id, text, label), a JSON array or JSONL."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "text", "label"} - set(rows[0].keys() if rows else set())
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        return [{"id": r["id"], "text": r["text"], "label": r["label"]} for r in rows]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .csv, .json or .jsonl")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "label"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "text": record["text"], "label": record["label"]})
    return out
