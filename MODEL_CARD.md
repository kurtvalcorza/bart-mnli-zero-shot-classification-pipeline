---
license: mit
model_card_spec: "1.1"
pipeline_tag: zero-shot-classification
base_model: facebook/bart-large-mnli
date_published: "2019-11"
date_published_source: "fairseq BART code+checkpoint release, examples/bart first commit 2019-11-09 (facebookresearch/fairseq#902); Hub history begins 2020-02-12"
---

# BART-large MNLI (DIMER package v0.1.0) — NLI Sequence Classifier (Zero-Shot Text Classification)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fbart--large--mnli-ffcc4d?style=flat)](https://huggingface.co/facebook/bart-large-mnli)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Ffairseq-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1910.13461-b31b1b.svg)](https://arxiv.org/abs/1910.13461)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/tutorials/bart_zero_shot_classification_colab.ipynb) [`bart_zero_shot_classification_colab.ipynb`](https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/tutorials/bart_zero_shot_classification_colab.ipynb)  
  *Zero-shot text classification with the pinned `facebook/bart-large-mnli` weights: `classify` scores caller-supplied labels by NLI entailment (an entailment-derived softmax, not a calibrated probability), the `accuracy` helper scores the top label when gold labels are supplied, and the results are exported with provenance.*

---

#### Description

`facebook/bart-large-mnli` is the BART-large checkpoint (Lewis et al., arXiv:1910.13461) after fine-tuning on the MultiNLI natural-language-inference dataset, published by Facebook AI on the Hugging Face Hub and pinned here to revision `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`. BART is a denoising sequence-to-sequence Transformer: a 12-layer bidirectional encoder and a 12-layer autoregressive decoder with `d_model` 1024, 16 attention heads, a 50 265-entry byte-level BPE vocabulary and 1024 positions (snapshot `config.json`), 407 344 133 float32 parameters in the pinned `model.safetensors` (counted from the SafeTensors header). The MNLI fine-tune replaced generation with a three-way classification head (`contradiction`, `neutral`, `entailment`) read from the decoder's final end-of-sequence state. Zero-shot classification follows Yin et al. (arXiv:1909.00161), as the upstream README describes: the text to classify is the NLI premise, each caller-supplied label is inserted into the hypothesis template `This example is {label}.` (`HYPOTHESIS_TEMPLATE`), and the entailment and contradiction logits of each pair become a per-label score. No adaptation happens in this repository — no training, fine-tuning or in-context conditioning; the pinned checkpoint is used as published. What this repository adds is packaging: the `BARTZeroShotClassificationPipeline` class in `src/bart_zero_shot_classification_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`, `stage_missing_files`), input validation against named ceilings, the score conversion in NumPy, and a fixed output contract.

#### Intended Use and Limitations

###### Primary Intended Uses

Zero-shot text classification: input one English string and a list of 1–`MAX_LABELS = 32` unique label names chosen by the caller at call time; output one score per label, sorted descending, plus the `top_label`. With `multi_label=False` (default) the entailment logits are soft-maxed across the labels so the scores sum to one and exactly one label is best; with `multi_label=True` each label is scored independently by a softmax over its own contradiction and entailment logits, for texts that may belong to several classes at once. The upstream README's own example — `one day I will see the world` against `travel`, `cooking`, `dancing` — is the shape of task intended: topic tagging, intent routing, content triage, or bootstrapping labels for a corpus before a supervised model exists, wherever the label names are ordinary English phrases the hypothesis template can carry. In a larger system the pipeline is a zero-configuration baseline and a labelling aid whose output a human or a downstream model consumes; it is not a decision engine.

###### Primary Intended Users

Machine-learning engineers, NLP researchers and application developers integrating an off-the-shelf zero-shot classifier into research prototypes, internal enterprise tooling or the DIMER model workbench. The pipeline assumes its users understand that the score is derived from an NLI entailment logit and is a ranking signal rather than a calibrated probability of class membership, that the quality of the result depends on how well the label wording fits the hypothesis template (`This example is sports.` reads naturally; a label such as `Q3 revenue variance` does not), that the checkpoint is English-only and 407 M parameters (~1.6 GB float32, roughly 8 s to load and 0.1–0.3 s per call on the CPU used here), and that the upstream MNLI fine-tuning data has not been audited for bias by this repository. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not text generation or summarisation — the classification head replaces the language-model head (the `bart-cnn-summarization-pipeline` sibling covers summarisation); not a supervised classifier trained on the caller's data and not a way to obtain one — nothing is fine-tuned here; not natural-language inference on arbitrary premise/hypothesis pairs, because `classify` always builds the hypothesis from the template and the raw three-way NLI logits are exposed only as `entailment_logit`/`contradiction_logit` per label; not sentence embeddings or token-level tagging.
2. **Input boundary:** `str` premise only (`TypeError` otherwise); empty or whitespace-only text rejected; more than `MAX_TEXT_CHARS = 8000` characters rejected before tokenisation and any premise+hypothesis pair over `MAX_TEXT_TOKENS = 1024` BPE tokens rejected after it — inputs are never silently truncated; label lists outside 1–32 entries, labels over `MAX_LABEL_CHARS = 100` characters, empty or duplicate labels, and a `hypothesis_template` without exactly one `{}` are rejected. Non-English text, code, and labels that are not English noun phrases degrade without any warning from the pipeline.
3. **Decision boundary:** not for autonomous or high-impact decisions — routing complaints, screening people, moderating content, triaging medical or legal text — without a human reviewing the output and a locally measured error rate on the deployment's own labels; not for inferring attributes of individuals from text.

#### Factors

###### Groups

The pipeline is human-centric in the sense that its inputs are natural-language text about, and often written by, people, and the labels a caller supplies can name social groups, attributes or intents. The upstream README states only that the checkpoint is `bart-large` "after being trained on the MultiNLI (MNLI) dataset" and carries no bias or group-level statement of its own; MultiNLI is crowd-written hypotheses over premises from ten written and spoken genres, and neither the upstream card nor this repository reports any group-level performance breakdown for it or for the BART pre-training corpus. This repository ran no fairness evaluation. The obligation therefore transfers to the operator: before any deployment that touches people, run paired probes that differ only in a group term (in the premise, in the labels, or both) and compare the ranked scores, evaluate top-label accuracy stratified by the groups relevant to the application on the operator's own labelled data, and treat a material gap as a blocker.

###### Instrumentation

The training data was captured by no physical sensor: it is text. The upstream README discloses the fine-tuning set by name only — the MultiNLI dataset — and the base model's pre-training data is disclosed in the BART paper (the same corpus as RoBERTa: books, news, web text and stories). The instrument characteristics that reach the model are therefore the text-collection and crowd-annotation procedures behind those corpora, none of which the pinned README describes. At inference the only instrument is the byte-level BPE tokenizer shipped in the snapshot (`vocab.json`, `merges.txt`, `tokenizer.json`), which encodes each premise+hypothesis pair as one sequence; unusual Unicode, OCR errors, code and URLs fragment into many sub-word pieces, and the pipeline reports the longest pair's token count (`n_tokens`) but does not detect encoding drift, language change or OCR noise.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `numpy==2.5.3` (exact pins in `pyproject.toml`), float32. This is a 407 M-parameter model whose `model.safetensors` is 1 629 437 147 bytes, so it needs roughly 1.7 GB of free RAM for the weights alone. `from_pretrained(device=None)` picks `cuda:0` when available, else CPU; this repository's smoke ran on CPU only (Windows venv, `CUDA_VISIBLE_DEVICES=-1`, `device="cpu"`): loading and digest-verifying the 1.63 GB snapshot took 7.84 s, one three-label `classify` call 0.334 s, a second call with `multi_label=True` 0.091 s. The CUDA path is untested in this repository. Data environment: inputs are assumed to be modern English prose of the kinds MultiNLI covers (fiction, letters, government reports, telephone speech transcripts, and similar written and spoken genres) and labels are assumed to be short English phrases that read naturally in `This example is {label}.`; behaviour on other languages, domain jargon, very long documents near the 1024-token ceiling, or label sets whose members overlap in meaning is not measured here and is expected to degrade.

#### Metrics

###### Performance Measures

The pipeline reports one performance measure, only when the caller supplies ground truth: `accuracy`, the fraction of texts whose `top_label` equals the caller's gold label by exact string match (`accuracy(predicted, gold)` in `pipeline.py`). Accuracy is the right first measure for a single-label decision over a caller-defined label set because it reads directly off the argmax rule the pipeline applies and needs no class prior; it is chosen over precision/recall per class because a zero-shot label set changes from call to call and per-class counts on a handful of items would be noise. What a reader loses by reading it alone is any view of ranking quality below the top label and of the `multi_label=True` case, where the argmax match is only a proxy — a caller who needs those must compute their own top-k hit rate or per-label precision/recall against multi-hot gold labels. Without gold labels no measure is reported: the public `evaluation_report()` helper then returns `verdict: not-measurable` with an empty `metrics` list and a `needs` field naming what would make the task measurable; with gold labels it returns `verdict: sample-sanity` and one `accuracy` entry. The upstream README reports no accuracy number for this checkpoint on MNLI or on any zero-shot benchmark, and this repository has measured none beyond the three-label smoke, which is one observation, not a result.

###### Decision thresholds

`classify` applies an implicit decision rule and names it in every result (`decision_rule`): with `multi_label=False` the entailment logits are soft-maxed across the labels and the `top_label` is the argmax — no minimum score is required, so a text that fits none of the labels still receives a winner; with `multi_label=True`, or whenever only one label is supplied (the upstream `ZeroShotClassificationPipeline` rule, reproduced here), each label's score is the entailment probability of a softmax over its own `[contradiction, entailment]` logits and no threshold is applied, so the caller sees independent scores and must decide which count as present. No acceptance threshold was set during development. A deployment that needs one must choose it on its own labelled data, trading the cost of a wrong confident label or a spurious extra tag (false positive) against a missed class (false negative) for its application, and must re-set it whenever the label wording or the hypothesis template changes, because both move the score scale.

###### Approaches to uncertainty and variability

`accuracy` is estimated by a single pass over whatever items the caller supplies, with no resampling and no dispersion; the `evaluation_report` records this (`estimation: single pass over n item(s); no dispersion`), and a caller who needs an interval must bootstrap over their own labelled set. Inference is deterministic given the same weights, device and library versions: dropout is disabled by `model.eval()`, there is no sampling, and no seed is required; small numeric differences between CPU and GPU kernels can reorder near-tied labels. The `score` field is not calibrated: it is a softmax over entailment logits whose scale depends on the label wording and the template — the smoke run gave `travel` 0.9939 against two unrelated labels, and the same premise would score differently against a different label set — so a caller who needs a probability of class membership must fit a calibration map on their own labelled data. The per-label `entailment_logit` and `contradiction_logit` are returned so that the caller can inspect the raw signal behind each score.

#### Ethical considerations and biases

###### Data

The upstream README discloses the fine-tuning data as the MultiNLI dataset and links the base model; the disclosure stops there — no per-genre composition, annotator demographics or licensing of the underlying premises is given in the pinned README, and the BART pre-training corpus (books, news, web text and stories per the BART paper) is described only in the paper. MultiNLI premises come from published and web text and crowd-written hypotheses, so the presence of personal data in the training corpora is not ruled out; whether any of it is sensitive is not known. This repository distributes code, tests and documentation; the 1.63 GB snapshot (`model.safetensors`, `config.json` and tokenizer files) is git-ignored and staged locally under `weights/bart-large-mnli/` with a manifest, and no sample data or datasets are shipped. The operator must audit the text they submit for personal, confidential or proprietary content and must not log or retain classifications of such text without the same controls they would apply to the text itself; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite (24 tests in `tests/test_pipeline.py`, `tests/test_role_helpers.py` and `tests/test_import_boundary.py`, plus the notebook parity tests) and the CPU smoke run recorded in this repository. Where a sensitive use is foreseeable — for example triaging patient messages by urgency, or flagging job applications by inferred category — it is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data with a documented bias probe stratified by the affected groups, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/bart_zero_shot_classification_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest naming another model or revision and fetches only manifest-listed files at the pinned revision, and `verify_snapshot` re-hashes every file in `weights/bart-large-mnli/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; both run before `torch` or `transformers` is imported (`tests/test_import_boundary.py`); the Hub path is taken only with `allow_download=True` and always with `revision=MODEL_REVISION`; `trust_remote_code=False` on both tokenizer and model. (2) Input integrity — `_check_text` rejects non-`str`, empty and over-`MAX_TEXT_CHARS` premises, `_check_labels` rejects non-list, empty, oversized (`MAX_LABELS`, `MAX_LABEL_CHARS`), non-string and duplicate labels, `_check_template` requires exactly one `{}`, `multi_label` must be a `bool`, and `_check_input_tokens` rejects any pair over `MAX_TEXT_TOKENS` instead of truncating. The public `validate_inputs()` helper applies those same checks through the same private functions and returns one input manifest (schema, ceilings, per-input observations, the hypotheses that will be scored, verdict, findings), so a caller can record exactly what was accepted or rejected without duplicating the validation logic. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, float32, and `model_id`/`model_revision`/`device`/`source`/`decision_rule` in every result. (4) Refusals — no fine-tuning, no generation and no raw NLI on caller-built hypotheses are exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (debiasing, re-weighting, calibration) is applied: the weights are redistributed unmodified.

###### Risks and harms

Confident wrong label: with `multi_label=False` the scores always sum to one, so a text that matches none of the labels still receives a high-scoring winner (mechanically, the least-contradicted hypothesis wins); the operator bears the harm when that label is acted on, and the likelihood is high whenever the label set is incomplete. Label-wording sensitivity: the same text scores differently under `sports` and `athletics`, because the hypothesis sentence changes; a deployment that tunes label names on a few examples can overfit them. Bias reproduction: entailment judgements were learned from crowd-written MNLI hypotheses and web-scale pre-training text, so labels that name social groups, occupations or attributes can rank differently for premises that differ only in a group term; data subjects and third parties bear the harm when such rankings feed a downstream system. Memorised or stereotyped associations surface as spurious entailment. Automation bias: a reviewer shown a scored label list checks it less carefully. Resource exhaustion: a 32-label call on a long premise runs 32 pairs of up to 1024 tokens through a 407 M-parameter model in one batch, which can exhaust memory on a small host; the ceilings bound but do not eliminate this. Magnitude ranges from a mis-filed ticket to a discriminatory triage of people.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — including classifying people's messages by inferred gender, ethnicity, religion, health status or political opinion with labels that name those attributes. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, for example by triaging applications or claims into categories that proxy for a protected characteristic. It must not power deceptive or manipulative applications such as targeting persuasion at people classified by vulnerability. Any use that violates the MIT terms of the upstream weights (the licence and copyright notice must accompany redistributed copies) or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's output is a ranked list of caller-supplied labels.

## Immutable provenance

- Model: `facebook/bart-large-mnli`
- Revision: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`
- Snapshot manifest: `weights/bart-large-mnli/dimer-base-manifest.json`, `totalBytes` 1632153123, 7 files
- `model.safetensors` SHA-256: `cfbb687dbbd9df99fe865e1860350a22aebac4d26ee4bcb50217f1df606a018e` (1629437147 bytes; 518 float32 tensors, 407 344 133 parameters)
- `config.json` SHA-256: `a0f9bcb245b680a96ccae0ad8d155f267ec3e3c971ef4a4937e52ea9ba368a86` (1154 bytes)
- `tokenizer.json` SHA-256: `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e` (1355863 bytes)
- Weight format: SafeTensors; loader `BartForSequenceClassification.from_pretrained(<snapshot dir>, local_files_only=True, trust_remote_code=False)` with `AutoTokenizer` from the same directory. The upstream `LICENSE` file is not part of the snapshot; the licence is declared as `mit` in the snapshot `README.md` front matter.

## Input/output contract

- `BARTZeroShotClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `classify(text, labels, *, multi_label=False, hypothesis_template=HYPOTHESIS_TEMPLATE)` — `text`: `str`, 1–8000 chars; `labels`: list of 1–32 unique non-empty `str` ≤ 100 chars each; every premise+hypothesis pair ≤ 1024 tokens. Returns `{"labels": [{"label", "score", "entailment_logit", "contradiction_logit"}, ...] (sorted by descending score), "top_label", "multi_label", "hypothesis_template", "decision_rule", "n_tokens", "device", "source", "model_id", "model_revision"}`.
- `validate_inputs(texts, labels, *, multi_label=False, hypothesis_template=..., names=None)` — input manifest for a batch of premises sharing one label set; raises exactly as `classify` does.
- `evaluation_report(results, gold_labels=None, *, sample_kind="synthetic")` — `accuracy` of `top_label` against `gold_labels` (`verdict: sample-sanity`) or `verdict: not-measurable` without them.
- `accuracy(predicted, gold)` — exact-match fraction.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.
- `stage_missing_files(path=None, *, allow_download=False, downloader=None)` — returns the relative paths fetched (`[]` if none were missing).

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`; Python 3.12.
- Precision: float32 on both CPU and CUDA (`dtype=torch.float32` in the loader).
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `device="cpu"`): source `local-snapshot`, load + verify 7.84 s; `classify("one day I will see the world", ["travel", "cooking", "dancing"])` 0.334 s → `travel` 0.9939, `dancing` 0.0033, `cooking` 0.0029 (longest pair 16 tokens), which agrees with the upstream README's own example (0.9939, 0.0033, 0.0029) to four decimals; the same call with `multi_label=True` 0.091 s → `travel` 0.9945, `dancing` 0.0057, `cooking` 0.0018 (upstream README, without `exploration`: 0.9945, 0.0057, 0.0018). The CUDA path was not run.
- Tests: `pytest -q -o addopts= tests` — 24 offline tests plus the notebook parity tests, no weights required; `ruff check src tests tools` clean.

## References

- Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. ACL 2020. https://arxiv.org/abs/1910.13461
- Yin, Hay, Roth. Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. EMNLP 2019. https://arxiv.org/abs/1909.00161
- Williams, Nangia, Bowman. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. NAACL 2018 (MultiNLI). https://arxiv.org/abs/1704.05426
- fairseq BART implementation. https://github.com/facebookresearch/fairseq/tree/main/examples/bart
- Upstream card: https://huggingface.co/facebook/bart-large-mnli
