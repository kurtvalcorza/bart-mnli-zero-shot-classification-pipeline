# BART-MNLI Zero-Shot Classification Pipeline

DIMER inference and fine-tuning wrapper for **`facebook/bart-large-mnli`** — zero-shot text classification by NLI entailment over a caller-supplied label set — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot, with a bounded adaptation contract: a digest-pinned real labelled corpus (Banking77, ten intents with readable phrases), accuracy and macro-F1 with a majority baseline, supervised fine-tuning of the last decoder blocks and the NLI head on entailment/contradiction pairs with validation-accuracy epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `facebook/bart-large-mnli` (BART-large fine-tuned on MultiNLI; 12+12 layers, `d_model` 1024, 407 M parameters, English)
- Revision: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`
- Upstream weight license: MIT
- Upstream task: natural-language inference, used zero-shot as a classifier by scoring `This example is {label}.` against the text (Yin et al., 2019; the recipe in the upstream README)
- Repository adaptation: bounded supervised fine-tuning of the last *k* decoder blocks plus the NLI classification head (`adapt`, default 2 of 12 = 34,646,019 of 407,344,131 parameters) on entailment/contradiction pairs built from caller-supplied or pinned Banking77 records; the encoder and embeddings are never modified; the adapted model remains an NLI scorer over any label set; the adapter carries only the trained tensors and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from bart_zero_shot_classification_pipeline import BARTZeroShotClassificationPipeline

pipe = BARTZeroShotClassificationPipeline.from_pretrained()          # cuda:0 if available, else cpu
result = pipe.classify("one day I will see the world", ["travel", "cooking", "dancing"])
print(result["top_label"], [(e["label"], round(e["score"], 4)) for e in result["labels"]])

multi = pipe.classify("one day I will see the world", ["travel", "cooking", "dancing"], multi_label=True)
```

`classify` returns the labels sorted by descending `score`. With `multi_label=False` (default) the score is a softmax over the entailment logits across labels (sums to one; `top_label` is the argmax); with `multi_label=True` each label is scored independently by a softmax over its own `[contradiction, entailment]` logits. Either way the score is entailment-derived and **not calibrated**. Ceilings: `MAX_TEXT_CHARS = 8000`, `MAX_TEXT_TOKENS = 1024` per premise+hypothesis pair (rejected, not truncated), `MAX_LABELS = 32`, `MAX_LABEL_CHARS = 100`. `HYPOTHESIS_TEMPLATE = "This example is {}."` is the upstream template and can be overridden per call. `evaluate(records, labels, hypothesis_template=...)` classifies a validated `{id, text, label}` dataset over an explicit label set and reports accuracy, macro-F1 and per-label F1 (`measured` / `measured-small-sample`); `adapt(train, val, *, labels, epochs=2, lr=2e-5, batch_size=16, trainable_decoder_layers=2, seed=0, hypothesis_template=...)` builds one entailment and one contradiction pair per record, fine-tunes the last decoder blocks and the head, and keeps the best-validation-accuracy epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`, `label_names`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_csv`) live in `samples.py`; `classification_metrics` and `majority_baseline` in `metrics.py`; records are 8–20,000 mappings with 2–32 distinct label phrases, and every inference ceiling is a refusal, never a silent cut (training truncates pairs to 256 tokens).

Adaptation on the pinned Banking77 sample (CPU, about five minutes including the validation passes):

```python
from bart_zero_shot_classification_pipeline import (
    BARTZeroShotClassificationPipeline, DEFAULT_HYPOTHESIS_TEMPLATE, fetch_sample_dataset, check_split_disjoint, label_names, majority_baseline,
)

splits = fetch_sample_dataset()            # two pinned CSV files (1.1 MB), digest-verified, cached under weights/banking77/
check_split_disjoint(splits)               # 400 / 100 / 200 balanced records over ten intents, no message shared between splits
labels = label_names(splits['train'])      # the ten readable phrases the hypotheses are built from
pipe = BARTZeroShotClassificationPipeline.from_pretrained()
print(majority_baseline(splits['test'])['accuracy'], pipe.evaluate(splits['test'], labels, hypothesis_template=DEFAULT_HYPOTHESIS_TEMPLATE)['accuracy'])   # 10.0, 82.50 in the recorded run
pipe.adapt(splits['train'], splits['validation'], labels=labels, hypothesis_template=DEFAULT_HYPOTHESIS_TEMPLATE)   # last 2 decoder blocks + head, 2 epochs
print(pipe.evaluate(splits['test'], labels, hypothesis_template=DEFAULT_HYPOTHESIS_TEMPLATE)['accuracy'])          # 97.00 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                    # adapter.safetensors (139 MB) + manifest.json
again = BARTZeroShotClassificationPipeline.from_artifact(artifact)                  # verifies base digest + artifact digest before applying
```

## Weights layout

```
weights/bart-large-mnli/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # BartForSequenceClassification architecture, 3 NLI labels
  tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
  model.safetensors          # 1629437147 bytes, git-ignored
  README.md                  # upstream card, listed in the manifest; not used by the loader
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any manifest file is missing or its SHA-256 differs. On a fresh clone (manifest committed, weights git-ignored) `from_pretrained(allow_download=True)` fetches only the missing files at the pinned revision; the default is to refuse. Without any manifest, `allow_download=True` loads from the Hub with `revision=d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed (35 tests + 5 notebook-parity tests; 2 model-backed tests run only with the staged snapshot)
```

Smoke (loads the verified snapshot on CPU; measured numbers are in `MODEL_CARD.md` → Runtime):

```python
from bart_zero_shot_classification_pipeline import BARTZeroShotClassificationPipeline

pipe = BARTZeroShotClassificationPipeline.from_pretrained(device="cpu")
print(pipe.classify("one day I will see the world", ["travel", "cooking", "dancing"])["labels"][0])
```

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/tutorials/bart_zero_shot_classification_colab.ipynb)

`tutorials/bart_zero_shot_classification_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path fetches the two pinned Banking77 CSV files (1.1 MB, CC BY 4.0, each refused on any digest mismatch), keeps ten intents with readable phrases and draws 400 / 100 / 200 balanced message-disjoint records with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, exercises the inference contract on three synthetic sentences in both score modes, scores the majority baseline and the frozen zero-shot model on the test split (accuracy 10.0 / 82.50 in the recorded run, with the per-label F1), fine-tunes the last two decoder blocks and the NLI head on 800 entailment/contradiction pairs for two epochs with validation-accuracy epoch selection (193.2 s on CPU), re-scores the test split (accuracy 97.00, macro-F1 96.99), classifies ten unseen messages with `sample-sanity` / `measured-small-sample` verdicts, exports a 139 MB safetensors adapter and reloads it with 8/8 identical top labels. Every number is one seeded split with no dispersion estimate. BYOD (`{id, text, label}` as CSV, JSON or JSONL) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Candidate** — the notebook source passes all static checks and one local CPU pre-flight execution of the committed blob is recorded; a clean run in a supported hosted runtime is still required (see `STATUS.md` and `docs/release-verification.md`). The earlier inference-only notebook's Kaggle pass does not carry over to the `E2E` blob.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; see `docs/WEIGHTS.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
