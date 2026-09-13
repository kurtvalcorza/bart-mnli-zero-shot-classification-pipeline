"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bart_zero_shot_classification_pipeline",
    "repo_name": "bart-mnli-zero-shot-classification-pipeline",
    "stem": "bart_zero_shot_classification",
    "notebook_name": "bart_zero_shot_classification_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "BARTZeroShotClassificationPipeline",
    "weights_key": "bart-large-mnli",
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "BART-large MNLI — DIMER zero-shot text classification tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/tutorials/bart_zero_shot_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fbart--large--mnli-ffcc4d?style=flat",
            "https://huggingface.co/facebook/bart-large-mnli",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Ffairseq-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/fairseq/tree/main/examples/bart",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1910.13461-b31b1b.svg", "https://arxiv.org/abs/1910.13461"),
    ],
    "capability": "zero-shot text classification by NLI entailment (one text, a caller-supplied label set, one entailment-derived score per label, sorted; single-label softmax across labels or independent multi-label scores) using the pinned BART-large MNLI weights",
    "intro": (
        "At inference each caller-supplied label is inserted into the hypothesis template `This example is {label}.` "
        "(the upstream README's own recipe, exposed as `HYPOTHESIS_TEMPLATE`), every (text, hypothesis) pair is encoded as one "
        "sequence, and one batched forward pass of the 407 M-parameter BART-large encoder-decoder with its three-way MNLI "
        "classification head returns `contradiction`/`neutral`/`entailment` logits per pair. `classify` then converts them in "
        "NumPy: with `multi_label=False` the entailment logits are soft-maxed **across labels** (scores sum to one, `top_label` "
        "is the argmax); with `multi_label=True`, or when only one label is supplied, each label gets the entailment probability "
        "of a softmax over its own `[contradiction, entailment]` pair. **No adaptation occurs:** no training, fine-tuning, "
        "in-context conditioning, or preprocessing fitting — the pinned checkpoint is used as published. What the upstream "
        "checkpoint supplies is the encoder-decoder, the NLI head and the tokenizer; what the carried pipeline module adds is "
        "manifest verification, input validation and ceilings (over-long premise+hypothesis pairs are rejected, not truncated), "
        "the score conversion, a fixed output contract, the `accuracy` helper and the `validate_inputs` and `evaluation_report` "
        "stage helpers. **The `score` is entailment-derived and not a calibrated probability** of class membership; the pipeline "
        "ships no threshold."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author three synthetic sentences with a "
        "three-label set and one author-expected label each (or upload your own), stage and digest-verify the immutable upstream "
        "snapshot, surface the pipeline's ceilings and validate the batch into one input manifest, run `classify` per sentence "
        "and read the ranked scores correctly in single-label and multi-label mode, read from the machine-readable evaluation "
        "report what the `accuracy` on three author-labelled sentences does and does not mean, and export scores alongside "
        "identifiers plus provenance."
    ),
    "exclusions": (
        "supervised fine-tuning or a trained classifier, text generation or summarisation (the `bart-cnn-summarization-pipeline` "
        "sibling covers that), natural-language inference on caller-built premise/hypothesis pairs, sentence embeddings, "
        "token-level tagging, non-English text, or any calibrated probability. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline loads the checkpoint in float32 on both). This is a 407 M-parameter model: the model card's CPU smoke loaded and verified the 1.63 GB snapshot in 7.84 s and classified one sentence against three labels in 0.33 s, so the default path runs in well under a minute on a hosted CPU runtime once the ~1.6 GB `model.safetensors` download has finished. The pinned `torch==2.14.0` install and that download are the largest transfers of the run; allow ~2 GB of free RAM for the weights.",
        "- **Knowledge:** basic Python; what natural-language inference (entailment vs contradiction) is; why a softmax over entailment logits is a ranking signal and not a calibrated probability.",
        "- **Data:** the default sample is three synthetic sentences and a three-label set authored in code, each sentence with the label its author expects, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file whose first non-empty line is the comma-separated label set and whose remaining non-empty lines are the texts to classify, each optionally followed by a tab and its gold label. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**, written in this cell: three sentences — the upstream README's own "
                "example (`one day I will see the world`) and two more — against the label set `travel`, `cooking`, `dancing`, "
                "each sentence carrying the label its author expects and a stable identifier (`t1`, `t2`, `t3`) so every score "
                "can be mapped back to its text. The expected labels are the author's intent, not a labelled dataset: the "
                "`accuracy` the evaluation stage computes on them is a sample-sanity check that the code path works, never "
                "benchmark evidence. `MULTI_LABEL` is a Colab form parameter checked against the carried module in Section 5.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file whose first non-empty line "
                "is the comma-separated label set (1..`MAX_LABELS` unique labels, each at most `MAX_LABEL_CHARS` characters) and "
                "whose remaining non-empty lines are the texts to classify (each at most `MAX_TEXT_CHARS` characters and, "
                "paired with the longest hypothesis, at most `MAX_TEXT_TOKENS` BPE tokens — longer pairs are rejected by the "
                "pipeline, not truncated), each optionally followed by a tab and its gold label. If any line carries a gold "
                "label, every line must. The upload stays inside this runtime."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "MULTI_LABEL = False  # @param {{type:\"boolean\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    lines = [line.rstrip('\\n') for line in io.StringIO(uploaded[sample_name].decode('utf-8')) if line.strip()]\n"
                "    if len(lines) < 2:\n"
                "        raise ValueError(f'{{sample_name}}: expected a label line followed by at least one text line')\n"
                "    labels = [label.strip() for label in lines[0].split(',') if label.strip()]\n"
                "    rows = [line.split('\\t', 1) for line in lines[1:]]\n"
                "    texts = [row[0].strip() for row in rows]\n"
                "    golds = [row[1].strip() for row in rows if len(row) == 2]\n"
                "    if golds and len(golds) != len(texts):\n"
                "        raise ValueError(f'{{sample_name}}: {{len(golds)}} of {{len(texts)}} lines carry a gold label; all or none must')\n"
                "    gold_labels = golds or None\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    labels = ['travel', 'cooking', 'dancing']\n"
                "    texts = [\n"
                "        'one day I will see the world',\n"
                "        'Whisk the eggs and fold in the flour before baking.',\n"
                "        'The tango class meets every Thursday evening.',\n"
                "    ]\n"
                "    gold_labels = ['travel', 'cooking', 'dancing']\n"
                "    sample_name = 'synthetic_three_sentences'\n"
                "    sample_kind = 'synthetic (authored in this cell; the first sentence is the upstream README example)'\n"
                "text_ids = [f't{{index + 1}}' for index in range(len(texts))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join([','.join(labels), *texts]).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'labels': labels, 'texts': len(texts), 'gold_labels': gold_labels, 'multi_label': MULTI_LABEL, 'text_sha256': sample_sha256}})\n"
                "for text_id, text in zip(text_ids, texts, strict=True):\n"
                "    print(f'{{text_id}}: {{text[:100]}}')"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it takes the batch of texts `classify` will be "
                "called on one by one, the shared label set, `multi_label` and the hypothesis template, and runs the "
                "method's own checks — `_check_text`, `_check_labels`, `_check_template` — so a rejection here is a rejection "
                "there. `MAX_TEXT_CHARS` is the character guard applied before tokenisation; `MAX_TEXT_TOKENS` (1024, the "
                "checkpoint's position limit) applies to each premise+hypothesis pair after tokenisation and **rejects** "
                "longer pairs rather than truncating them, so it is enforced inside the pipeline and cannot be observed at "
                "this stage; `MAX_LABELS` and `MAX_LABEL_CHARS` bound the label set; `HYPOTHESIS_TEMPLATE` is the sentence "
                "each label is inserted into, and the manifest lists the resulting `hypotheses` verbatim so you can read "
                "whether your labels make natural sentences. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a label set "
                "with a duplicate entry and records the pipeline's own error message as a finding. The notebook never trims "
                "or alters the texts or labels."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'HYPOTHESIS_TEMPLATE': HYPOTHESIS_TEMPLATE, 'NUM_NLI_LABELS': NUM_NLI_LABELS, 'ENTAILMENT_INDEX': ENTAILMENT_INDEX, 'CONTRADICTION_INDEX': CONTRADICTION_INDEX}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(texts, labels, multi_label=MULTI_LABEL, names=text_ids)\n"
                "# Demonstrate the unique-labels rejection; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(texts, [*labels, labels[0]], multi_label=MULTI_LABEL)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-label-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': f'MAX_TEXT_TOKENS={{MAX_TEXT_TOKENS}} is checked per premise+hypothesis pair by the pipeline after tokenisation and rejects, never truncates'}})"
            ),
        },
        {
            "md": (
                "## 6. Classify each sentence and read the scores correctly\n\n"
                "**Input/output contract.** `classify(text, labels, multi_label=...)` takes one string and the label list and "
                "returns `labels` — one entry per supplied label ordered by descending `score`, each with the `label`, the "
                "`score`, and the raw `entailment_logit` and `contradiction_logit` behind it — plus `top_label`, "
                "`multi_label`, the `hypothesis_template` used, the `decision_rule` applied, `n_tokens` (the longest "
                "premise+hypothesis pair), the device and the model identity. **Score semantics:** with `multi_label=False` "
                "the score is a softmax over the entailment logits **across the labels** — it sums to one over the label set "
                "and the **default decision rule is `argmax`** (the first entry), so a text that fits none of the labels still "
                "gets a winner; with `multi_label=True` each score is the entailment probability of a softmax over that "
                "label's own `[contradiction, entailment]` pair, the scores do not sum to one, and no threshold is applied. "
                "In neither mode is the score a calibrated probability of class membership: it moves with the label wording "
                "and the template, and any acceptance threshold is owned by the caller and must be set on their own labelled "
                "data. The model card's CPU smoke on the first sentence gave `travel` 0.9939, `dancing` 0.0033, `cooking` "
                "0.0029 — the upstream README's own numbers to four decimals — which is one observation, not an expected "
                "value; near-tied labels can reorder between CPU and CUDA kernels. Inference is deterministic on a fixed "
                "device and dtype (`model.eval()`, no sampling, no seed needed). The cell also classifies the first sentence "
                "in the other mode so the two score conventions can be compared side by side."
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "elapsed = []\n"
                "for text_id, text in zip(text_ids, texts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.classify(text, labels, multi_label=MULTI_LABEL)\n"
                "    elapsed.append(time.perf_counter() - started)\n"
                "    results.append(result)\n"
                "    scores = [entry['score'] for entry in result['labels']]\n"
                "    checks = {{\n"
                "        'one_entry_per_label': sorted(entry['label'] for entry in result['labels']) == sorted(labels),\n"
                "        'scores_descending': all(a >= b for a, b in zip(scores, scores[1:], strict=False)),\n"
                "        'scores_in_unit_interval': all(0.0 <= s <= 1.0 for s in scores),\n"
                "        'single_label_scores_sum_to_one': MULTI_LABEL or len(labels) == 1 or abs(sum(scores) - 1.0) < 1e-6,\n"
                "        'top_label_is_first': result['top_label'] == result['labels'][0]['label'],\n"
                "        'n_tokens_within_ceiling': 1 <= result['n_tokens'] <= MAX_TEXT_TOKENS,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'classify output for {{text_id}} failed a sanity check: {{checks}}')\n"
                "    print(f'{{text_id}}: {{text[:60]}}')\n"
                "    print({{'top_label': result['top_label'], 'seconds': round(elapsed[-1], 3), 'n_tokens': result['n_tokens'], 'decision_rule': result['decision_rule'], 'checks': checks}})\n"
                "    for rank, entry in enumerate(result['labels'], start=1):\n"
                "        print(f\"  {{rank}}. {{entry['label']:<12}} score {{entry['score']:.4f}}  entailment {{entry['entailment_logit']:+.3f}}  contradiction {{entry['contradiction_logit']:+.3f}}\")\n"
                "other_mode = pipe.classify(texts[0], labels, multi_label=not MULTI_LABEL)\n"
                "print({{'first_text_other_mode': {{'multi_label': other_mode['multi_label'], 'scores': {{entry['label']: round(entry['score'], 4) for entry in other_mode['labels']}}, 'decision_rule': other_mode['decision_rule']}}}})\n"
                "classify_sanity = {{}}\n"
                "if gold_labels is not None:\n"
                "    classify_sanity = {{'top_label_matches_expected': [result['top_label'] == gold for result, gold in zip(results, gold_labels, strict=True)]}}\n"
                "    print({{'sanity_check': classify_sanity, 'note': 'falsifiable plumbing check on author-expected labels; the metric is computed in Section 7'}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When gold labels "
                "are supplied it reports one metric, **`accuracy`** — the fraction of texts whose `top_label` equals the gold "
                "label by exact string match, the repository's own `accuracy` helper — with verdict `sample-sanity`: on three "
                "author-labelled synthetic sentences the number is a plumbing check on this sample, estimated by a single pass "
                "with no dispersion, and the report says so in `estimation` and `reason`. With `multi_label=True` the argmax "
                "match is only a proxy, because the mode exists for texts that belong to several classes. Without gold labels "
                "the verdict is `not-measurable` and `needs` names what would make the task measurable: one gold label per "
                "text from the deployment's own label set over enough texts to state a dispersion, and a calibration set "
                "before any score is read as a probability. The sanity checks printed in Section 6 remain falsifiable "
                "plumbing checks, not results. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, gold_labels, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: supply one gold label per text to obtain the accuracy plumbing check; a real evaluation needs your own labelled set.')\n"
                "else:\n"
                "    print({{'accuracy': report['metrics'][0]['value'], 'note': 'sample-sanity on ' + str(report['n_items']) + ' author-labelled item(s); not a benchmark'}})"
            ),
        },
        {
            "md": (
                "## 8. Export scores alongside identifiers, and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report. The "
                "scores go to CSV (`outputs/{stem}_scores.csv`) with one row per (text, label) — `text_id`, `rank`, `label`, "
                "`score`, `entailment_logit`, `contradiction_logit`, `multi_label` — so every score stays attached to its "
                "text identifier for downstream use. One JSON record (`outputs/{stem}_result.json`) preserves every result "
                "(the identified texts, the ranked labels with scores and logits, `top_label`, the decision rule, "
                "`n_tokens`, seconds), the other-mode comparison, the sanity checks, the ceilings in force, the input "
                "manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, "
                "revision, embedded module digest, generator), the model identifier, the immutable model revision, the model "
                "licence, the verified snapshot summary, and the runtime identity (Python, `torch`, `transformers`, device, "
                "dtype). No credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "with open('outputs/{stem}_scores.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['text_id', 'rank', 'label', 'score', 'entailment_logit', 'contradiction_logit', 'multi_label'])\n"
                "    for text_id, result in zip(text_ids, results, strict=True):\n"
                "        for rank, entry in enumerate(result['labels'], start=1):\n"
                "            writer.writerow([text_id, rank, entry['label'], f\"{{entry['score']:.7f}}\", f\"{{entry['entailment_logit']:.5f}}\", f\"{{entry['contradiction_logit']:.5f}}\", result['multi_label']])\n"
                "payload = {{\n"
                "    'classify': [\n"
                "        {{\n"
                "            'text_id': text_id,\n"
                "            'text': text,\n"
                "            'expected_label': None if gold_labels is None else gold,\n"
                "            'top_label': result['top_label'],\n"
                "            'labels': [{{'rank': rank, **entry}} for rank, entry in enumerate(result['labels'], start=1)],\n"
                "            'decision_rule': result['decision_rule'],\n"
                "            'multi_label': result['multi_label'],\n"
                "            'hypothesis_template': result['hypothesis_template'],\n"
                "            'n_tokens': result['n_tokens'],\n"
                "            'seconds': round(seconds, 3),\n"
                "        }}\n"
                "        for text_id, text, gold, result, seconds in zip(text_ids, texts, gold_labels or [None] * len(texts), results, elapsed, strict=True)\n"
                "    ],\n"
                "    'score_semantics': 'entailment-derived softmax (across labels when multi_label=False, per label otherwise); not a calibrated probability; argmax rule; no threshold shipped',\n"
                "    'other_mode_first_text': {{'multi_label': other_mode['multi_label'], 'labels': other_mode['labels'], 'decision_rule': other_mode['decision_rule']}},\n"
                "    'plumbing_check': classify_sanity,\n"
                "    'scores_file': 'outputs/{stem}_scores.csv',\n"
                "    'ceilings': ceilings,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'labels': labels, 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The scores are entailment-derived: each label became the hypothesis `This example is {{label}}.`, the MNLI head "
        "judged it against the text, and the entailment logits were soft-maxed across the labels (single-label mode) or "
        "against each label's own contradiction logit (multi-label mode). In single-label mode the first entry is the "
        "argmax and the scores sum to one, so a text that matches none of the labels still receives a confident-looking "
        "winner; in multi-label mode nothing is thresholded. Neither score is a calibrated probability, both move with the "
        "label wording and the template, and the pipeline applies no threshold — the caller owns any cut-off and must set "
        "it on labelled data from their own label set. On the synthetic sample the `accuracy` in the evaluation report is a "
        "sample-sanity check over three author-labelled sentences with no dispersion, and a real evaluation needs the "
        "caller's own labelled texts over enough items to state one. Premise+hypothesis pairs over 1024 BPE tokens are "
        "rejected, not truncated; the checkpoint is English only and carries whatever associations MultiNLI and the BART "
        "pre-training corpus contain, which neither the upstream card nor this repository has audited; the pipeline exposes "
        "no fine-tuning, no generation and no raw NLI on caller-built hypotheses. Inference is deterministic on a fixed "
        "device and dtype, but CPU and CUDA kernels can reorder near-tied labels.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated inputs against the enforced "
        "ceilings, execute the public `classify` path in both modes, and emit the shown machine-readable outputs in the "
        "tested runtime — without the repository being reachable. It does **not** establish benchmark superiority, "
        "classification accuracy on any domain, a usable threshold, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "**Next experiments.** Flip `MULTI_LABEL` and compare the two score conventions on the same sentences; reword one "
        "label (`travel` → `holidays`) and watch every score move, which is what \"not calibrated\" means in practice; add a "
        "sentence that fits none of the labels and read the single-label winner it still receives; assemble a few dozen "
        "labelled texts of your own and compute the accuracy the evaluation report asks for, with a bootstrap interval. None "
        "of these turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/fairseq/tree/main/examples/bart\n"
        "- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension: https://arxiv.org/abs/1910.13461\n"
        "- Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach (Yin et al.): https://arxiv.org/abs/1909.00161"
    ),
}
