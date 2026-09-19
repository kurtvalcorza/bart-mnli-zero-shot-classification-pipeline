"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E zero-shot-classification workflow: the pinned BART-large MNLI snapshot
is digest-verified and loaded, a digest-pinned real labelled corpus (Banking77, ten intents) is fetched,
validated and split, three synthetic sentences are classified through the inference contract, the frozen
zero-shot model is scored against gold intents beside the majority-class baseline, a bounded fine-tuning
of the last decoder blocks and the NLI head on entailment/contradiction pairs runs in the kernel, the
held-out split is scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bart_zero_shot_classification_pipeline",
    "repo_name": "bart-mnli-zero-shot-classification-pipeline",
    "stem": "bart_zero_shot_classification",
    "notebook_name": "bart_zero_shot_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned BART-large MNLI snapshot (safetensors, 1.6 GB), fetches the two digest-pinned Banking77 CSV files from the "
        "project repository (1.1 MB, no credential), keeps ten intents and draws 400 / 100 / 200 balanced training, "
        "validation and test messages from the release's own partition, classifies three synthetic sentences through the "
        "inference contract with an input manifest and a rejection probe, scores the frozen zero-shot model on the test "
        "messages with accuracy and macro-F1 beside the majority-class baseline, runs a bounded fine-tuning of the last two "
        "decoder blocks and the NLI head on entailment/contradiction pairs built from the training messages with "
        "validation-accuracy epoch selection, scores the held-out split again, classifies new messages with the adapted "
        "model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to "
        "verify label parity. The default path needs no repository clone, no DIMER worker or service, no credential, no "
        "upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about six minutes of "
        "model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled texts as a CSV (columns `id`, `text`, `label`), a JSON array or a JSONL file of `{{id, text, label}}` records "
        "with 2..32 distinct labels written as short readable phrases. They pass through the same validation, seeded stratified "
        "text-disjoint split, baseline, fine-tuning, held-out evaluation, inference, artifact export and reload-parity cells as "
        "the Banking77 sample. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "BARTZeroShotClassificationPipeline",
    "weights_key": "bart-large-mnli",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "BART-large MNLI — DIMER E2E zero-shot classification fine-tuning tutorial (standalone)",
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
    "capability": "zero-shot text classification by NLI entailment (one text, a caller-supplied label set, one entailment-derived score per label, sorted; single-label softmax across labels or independent multi-label scores) and bounded supervised fine-tuning of the last decoder blocks and the NLI head on a labelled-text dataset, using the pinned BART-large MNLI weights",
    "intro": (
        "At inference each candidate label is written into a hypothesis (`This example is {{label}}.` by default), every "
        "(text, hypothesis) pair is encoded once as one BPE sequence, and the 407 M-parameter BART-large encoder-decoder "
        "with the three-way NLI head fine-tuned upstream on MultiNLI emits contradiction / neutral / entailment logits per "
        "pair; the carried module turns them into one score per label — a softmax over the entailment logits across the "
        "labels (`multi_label=False`, the default) or, per label, the entailment probability of its own "
        "[contradiction, entailment] pair (`multi_label=True`) — and the **default decision rule is `argmax`**. What the "
        "upstream checkpoint supplies is the NLI model and the tokenizer; what the carried pipeline module adds is manifest "
        "verification, input validation with named ceilings (a premise+hypothesis pair over `MAX_TEXT_TOKENS` is rejected, "
        "not truncated), the decision rule, a fixed output contract, and the `validate_inputs` and `evaluation_report` "
        "stage helpers. **The score is an entailment-derived softmax, not a calibrated probability**, and no threshold is "
        "shipped.\n\n"
        "What this notebook adds to inference is **adaptation with gold labels**. The dataset is real and far from MultiNLI: "
        "Banking77 (Casanueva et al., 2020; CC BY 4.0), 13,083 customer-support messages over 77 fine-grained banking "
        "intents, of which the tutorial keeps **ten** and gives each a readable phrase (`card arrival`, `a lost or stolen "
        "card`, `the exchange rate`, …) so the hypothesis reads as English. Zero-shot NLI already does well on these ten "
        "(the build record measured 82.5 % accuracy on the test split), and the fine-tuning question is whether a bounded "
        "adaptation — every labelled message becomes one **entailment pair** with its gold phrase and one **contradiction "
        "pair** with a seeded wrong phrase, and only the last two decoder blocks and the NLI head train — closes the rest "
        "of the gap on held-out messages. Two metrics are implemented in the carried `metrics.py` (accuracy and "
        "**macro-F1** with per-label precision/recall) and the **majority-class baseline** shows where a classifier that "
        "does nothing sits. Nothing here is a quality claim about your labels: it is one seeded split of one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned labelled corpus and validate and split it "
        "without leakage; classify through the public API with an explicit label set and hypothesis template and read the "
        "ranked scores correctly in single-label and multi-label mode; score the frozen zero-shot model against gold labels "
        "beside the majority baseline and read the per-label F1; run a bounded fine-tuning on entailment/contradiction "
        "pairs with explicit hyperparameters and validation-based epoch selection; evaluate on an independent test split; "
        "classify new messages; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "a trained classification head over a fixed label vocabulary (the adapted model is still an NLI scorer over any "
        "label set), text generation or summarisation (the `bart-cnn-summarization-pipeline` sibling covers that), "
        "natural-language inference on caller-built premise/hypothesis pairs, sentence embeddings, token-level tagging, "
        "full-model or encoder fine-tuning, non-English text, any calibrated probability, and any claim that a Banking77 "
        "split stands in for your label set. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured 6 s to load and digest-verify the 1.6 GB snapshot, about 0.3 s per message for a ten-label classification (one minute for the 200-message test split) and about 65 s per training epoch over 800 NLI pairs plus a 100-message validation pass per epoch. The pinned `torch==2.14.0` install and the 1.6 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what natural-language inference (entailment vs contradiction) is; why a softmax over entailment logits is a ranking signal and not a calibrated probability; what accuracy and macro-F1 measure and why a balanced test split makes the majority baseline equal to one over the number of labels.",
        "- **Data contract:** records are `{{id, text, label}}` — a message and the readable phrase of its gold label — the text 1..8,000 characters and, paired with the longest hypothesis, at most 1,024 BPE tokens at inference, the label 1..100 characters, 2..32 distinct labels, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; texts are de-duplicated case-insensitively before splitting so the same message never sits in two splits; during training only, pairs are truncated to 256 BPE tokens (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a label phrase describes its messages or that the labels are mutually exclusive — a mislabelled corpus is fine-tuned on without complaint, and a label phrase the model cannot read as English will score badly zero-shot.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a customer-support log with its intent labels is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches two pinned objects (`train.csv` 839,073 bytes, SHA-256 `b06e26ac…`; `test.csv` 239,961 bytes, SHA-256 `d12d6e3b…`) from `raw.githubusercontent.com` at the pinned `PolyAI-LDN/task-specific-datasets` commit over HTTPS, each refused on any mismatch before it is read; Banking77 is CC BY 4.0 (attribution: PolyAI; Casanueva et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Labelled corpus, validation and split\n\n"
                "`fetch_corpus` downloads the two pinned Banking77 CSV files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` checks the columns, the row "
                "counts and the 77 intents. `build_sample_dataset` keeps the ten intents in `LABEL_SET`, maps each to its "
                "readable phrase, drops repeated messages, and draws **balanced** seeded samples — 40 training and 10 "
                "validation messages per intent from `train.csv` (disjoint), 20 test messages per intent from `test.csv` — "
                "the release's own partition. `validate_dataset` then checks every record against the contract and reports "
                "the label counts, `label_names` derives the ten-phrase label set every `classify` call will score, "
                "`check_split_disjoint` asserts no message appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 10,003 + 3,080 raw rows, three digests, splits 400 / 100 / 200 with 40 / 10 / 20 per label, and "
                "four refusal probes — a duplicate id, a label outside the label set, a missing field and a dataset too small "
                "to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "TEMPLATE = 'This customer message is about {{}}.'  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/banking77'))\n"
                "    raw_rows = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}}), {{len(LABEL_SET)}} of {{CORPUS_INTENTS}} intents'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "labels = label_names(train_records)\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'labels': labels, 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_texts': manifest['unique_texts'], 'label_counts': manifest['label_counts'], 'text_chars': manifest['text_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{k: train_records[0][k] for k in ('id', 'text', 'label')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'label outside the label set': ([{{**train_records[0], 'label': 'something else'}}, *train_records[1:8]], labels),\n"
                "    'missing field': [{{'id': r['id'], 'text': r['text']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(*probe) if isinstance(probe, tuple) else validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Classify through the inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on three synthetic sentences "
                "and a three-label set authored in this cell (the upstream README's `one day I will see the world` and two "
                "more). `validate_inputs` applies exactly the checks `classify` applies — text types and character ceilings, "
                "1..`MAX_LABELS` unique labels under `MAX_LABEL_CHARS`, one `{{}}` in the template — and returns an input "
                "manifest; the pair-token ceiling `MAX_TEXT_TOKENS` needs the real tokenizer and is enforced inside "
                "`classify`, which **rejects with a `ValueError` naming the count, never truncates**. A duplicate-label probe "
                "is validated too and its rejection recorded as a finding. `classify` returns one entry per label ordered by "
                "descending `score` with the raw entailment and contradiction logits, plus `top_label`, `multi_label`, the "
                "template, the `decision_rule`, `n_tokens` and the model identity. **Score semantics:** with `multi_label=False` "
                "the scores sum to one over the label set and the **default decision rule is `argmax`**, so a text that fits "
                "none of the labels still gets a winner; with `multi_label=True` each score is that label's own entailment "
                "probability and they do not sum to one. In neither mode is the score a calibrated probability of class "
                "membership. Whether the top labels are *right* is what Section 6 measures on 200 gold-labelled messages, not "
                "what three authored sentences can tell you."
            ),
            "code": (
                "import time\n\n"
                "MULTI_LABEL = False  # @param {{type:\"boolean\"}}\n\n"
                "demo_labels = ['travel', 'cooking', 'dancing']\n"
                "texts = ['one day I will see the world', 'Whisk the eggs and fold in the flour before baking.', 'The tango class meets every Thursday evening.']\n"
                "expected = ['travel', 'cooking', 'dancing']\n"
                "text_ids = [f't{{index + 1}}' for index in range(len(texts))]\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'HYPOTHESIS_TEMPLATE': HYPOTHESIS_TEMPLATE, 'NUM_NLI_LABELS': NUM_NLI_LABELS, 'ENTAILMENT_INDEX': ENTAILMENT_INDEX, 'CONTRADICTION_INDEX': CONTRADICTION_INDEX}}\n"
                "print(ceilings)\n"
                "print({{'decision_rules': {{'single': DECISION_RULE_SINGLE, 'multi': DECISION_RULE_MULTI}}}})\n"
                "input_manifest = validate_inputs(texts, demo_labels, multi_label=MULTI_LABEL, names=text_ids)\n"
                "try:\n"
                "    validate_inputs(texts, [*demo_labels, demo_labels[0]], multi_label=MULTI_LABEL)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-label-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "results = []\n"
                "for text_id, text in zip(text_ids, texts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.classify(text, demo_labels, multi_label=MULTI_LABEL)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    results.append(result)\n"
                "    scores = [entry['score'] for entry in result['labels']]\n"
                "    checks = {{\n"
                "        'one_entry_per_label': sorted(entry['label'] for entry in result['labels']) == sorted(demo_labels),\n"
                "        'scores_descending': all(a >= b for a, b in zip(scores, scores[1:], strict=False)),\n"
                "        'scores_in_unit_interval': all(0.0 <= s <= 1.0 for s in scores),\n"
                "        'single_label_scores_sum_to_one': MULTI_LABEL or abs(sum(scores) - 1.0) < 1e-6,\n"
                "        'top_label_is_first': result['top_label'] == result['labels'][0]['label'],\n"
                "        'n_tokens_within_ceiling': 1 <= result['n_tokens'] <= MAX_TEXT_TOKENS,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'classify output for {{text_id}} failed a sanity check: {{checks}}')\n"
                "    print({{'id': text_id, 'text': text[:60], 'top_label': result['top_label'], 'scores': {{e['label']: round(e['score'], 4) for e in result['labels']}}, 'seconds': round(elapsed, 3), 'n_tokens': result['n_tokens'], 'checks': checks}})\n"
                "other_mode = pipe.classify(texts[0], demo_labels, multi_label=not MULTI_LABEL)\n"
                "print({{'first_text_other_mode': {{'multi_label': other_mode['multi_label'], 'scores': {{e['label']: round(e['score'], 4) for e in other_mode['labels']}}}}}})\n"
                "print({{'top_label_matches_expected': [r['top_label'] == g for r, g in zip(results, expected, strict=True)], 'findings': len(input_manifest['findings'])}})"
            ),
        },
        {
            "md": (
                "## 6. Baseline and the frozen zero-shot model's score on the test split\n\n"
                "Two numbers frame the adaptation. The **majority-class baseline** predicts the most frequent gold label for "
                "every message — on a balanced ten-label split that is exactly 10 % accuracy, the floor any classifier must "
                "clear. The **frozen zero-shot model** classifies the 200 test messages over the ten phrases with the template "
                "from Section 4 and is scored with the same two metrics: **accuracy** (exact top-label match) and **macro-F1** "
                "(the unweighted mean of the per-label F1, so a rarely predicted label counts as much as a popular one). "
                "Expect the frozen model to do well already — these intents have readable names — and read the per-label F1 "
                "to see which phrases the NLI model reads badly (the build record found `ATM support` and `card arrival` "
                "far below the rest). About one minute on CPU."
            ),
            "code": (
                "baseline_majority = majority_baseline(test_records)\n"
                "print({{'majority_baseline': {{'accuracy': round(baseline_majority['accuracy'], 2), 'macro_f1': round(baseline_majority['macro_f1'], 2), 'n': baseline_majority['n'], 'rule': baseline_majority['baseline']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, labels, hypothesis_template=TEMPLATE)\n"
                "print({{'frozen_model_test': {{'accuracy': round(frozen_test['accuracy'], 2), 'macro_f1': round(frozen_test['macro_f1'], 2), 'n': frozen_test['n'], 'n_labels': frozen_test['n_labels'], 'verdict': frozen_test['verdict']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'per_label_f1': {{label: round(v['f1'], 1) for label, v in frozen_test['per_label'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record in test_records[:2]:\n"
                "    item = pipe.classify(record['text'], labels, hypothesis_template=TEMPLATE)\n"
                "    print({{'text': record['text'][:80], 'frozen': item['top_label'], 'score': round(item['labels'][0]['score'], 3), 'gold': record['label']}})\n"
                "assert frozen_test['accuracy'] > baseline_majority['accuracy']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder blocks and the NLI head\n\n"
                "`pipe.adapt` turns every training message into two NLI pairs — (message, template(gold phrase)) labelled "
                "*entailment* and (message, template(a seeded wrong phrase)) labelled *contradiction* — and trains only the "
                "last `TRAINABLE_DECODER_LAYERS` decoder blocks plus the classification head: two blocks by default, "
                "34,646,019 of 407,344,131 parameters; the encoder, the shared embeddings and the earlier decoder blocks "
                "stay frozen. Cross-entropy over the three NLI logits, AdamW at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling and no scheduler; pairs are truncated to 256 BPE tokens **during training only**. "
                "Epoch 0 records the frozen model's validation accuracy and macro-F1; every epoch is scored on the "
                "validation split with the same labels and template, and the epoch with the highest validation accuracy is "
                "kept.\n\n"
                "Watch validation accuracy climb into the high nineties over two epochs (about a minute of training plus a "
                "validation pass per epoch on CPU). The adapted model is still an NLI scorer: it answers any label set, but "
                "it has been pulled towards these ten phrases and this template."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_accuracy'] = round(entry['val']['accuracy'], 2)\n"
                "        row['val_macro_f1'] = round(entry['val']['macro_f1'], 2)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, labels=labels, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, hypothesis_template=TEMPLATE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'pairs_per_record': adapt_result['pairs_per_record'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no message in it appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, "
                "and the three numbers are put side by side with the per-label F1 before and after. Look for an accuracy gain "
                "of ten points or more — the cell asserts the adapted accuracy is above the frozen accuracy — and for the "
                "weakest phrases of Section 6 recovering. Two hundred messages from one seeded split of one corpus give no "
                "dispersion estimate; the deltas are sample-sanity evidence that the adaptation contract works, not a "
                "benchmark, and a gain on ten banking intents says nothing about your label set until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, labels, hypothesis_template=TEMPLATE)\n"
                "adapted_val = pipe.evaluate(val_records, labels, hypothesis_template=TEMPLATE)\n"
                "comparison = {{\n"
                "    'accuracy': {{'majority': round(baseline_majority['accuracy'], 2), 'frozen': round(frozen_test['accuracy'], 2), 'adapted': round(adapted_test['accuracy'], 2)}},\n"
                "    'macro_f1': {{'majority': round(baseline_majority['macro_f1'], 2), 'frozen': round(frozen_test['macro_f1'], 2), 'adapted': round(adapted_test['macro_f1'], 2)}},\n"
                "    'delta_vs_frozen': {{'accuracy': round(adapted_test['accuracy'] - frozen_test['accuracy'], 2), 'macro_f1': round(adapted_test['macro_f1'] - frozen_test['macro_f1'], 2)}},\n"
                "    'per_label_f1': {{label: {{'frozen': round(frozen_test['per_label'][label]['f1'], 1), 'adapted': round(adapted_test['per_label'][label]['f1'], 1)}} for label in labels}},\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "for record in test_records[:2]:\n"
                "    item = pipe.classify(record['text'], labels, hypothesis_template=TEMPLATE)\n"
                "    print({{'text': record['text'][:80], 'adapted': item['top_label'], 'score': round(item['labels'][0]['score'], 3), 'gold': record['label']}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'labels': labels,\n"
                "    'hypothesis_template': TEMPLATE,\n"
                "    'baselines': {{'majority': baseline_majority}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['accuracy'] > frozen_test['accuracy']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Classify new messages, export the adapter and reload it\n\n"
                "Ten messages that were in none of the splits (one per intent, drawn from the training file beyond the "
                "sample) are classified by the adapted model through the same `classify` contract as Section 5 and scored "
                "with the batch `evaluation_report` — the inference-stage helper, whose `accuracy` on ten gold-labelled "
                "items is a `sample-sanity` observation, never a benchmark — and with `pipe.evaluate` (`measured-small-sample`).\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder blocks and the NLI head, about 139 "
                "MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration (labels, template, hyperparameters) and the epoch history (OUT8). "
                "`BARTZeroShotClassificationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact "
                "manifest and digest **before** deserialising, refuses any tensor that is not an adaptable decoder or head "
                "tensor, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical top labels (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate(test_records[:10])]\n"
                "else:\n"
                "    used = {{r['text'].lower() for part in splits.values() for r in part}}\n"
                "    spare = [r for r in filter_records(corpus['train']) if r['text'].lower() not in used]\n"
                "    new_records = [{{**next(r for r in spare if r['label'] == label), 'id': f'new-{{i:02d}}'}} for i, label in enumerate(labels)]\n"
                "new_results = [pipe.classify(r['text'], labels, hypothesis_template=TEMPLATE) for r in new_records]\n"
                "new_report = evaluation_report(new_results, [r['label'] for r in new_records], sample_kind='ten unseen Banking77 messages' if not USE_BYOD else 'BYOD test records')\n"
                "new_metrics = pipe.evaluate(new_records, labels, hypothesis_template=TEMPLATE)\n"
                "for record, item in zip(new_records, new_results, strict=True):\n"
                "    print({{'id': record['id'], 'text': record['text'][:70], 'adapted': item['top_label'], 'score': round(item['labels'][0]['score'], 3), 'gold': record['label']}})\n"
                "print({{'new_messages': {{'verdict': new_report['verdict'], 'accuracy': new_report['metrics'][0]['value'], 'reason': new_report['reason']}}, 'evaluate_verdict': new_metrics['verdict'], 'macro_f1': round(new_metrics['macro_f1'], 2)}})\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['id', 'text', 'top_label', 'score', 'gold', 'n_tokens'])\n"
                "    writer.writeheader()\n"
                "    for record, item in zip(new_records, new_results, strict=True):\n"
                "        writer.writerow({{'id': record['id'], 'text': record['text'], 'top_label': item['top_label'], 'score': item['labels'][0]['score'], 'gold': record['label'], 'n_tokens': item['n_tokens']}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BARTZeroShotClassificationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.classify(r['text'], labels, hypothesis_template=TEMPLATE)['top_label'] for r in test_records[:8]]\n"
                "after = [reloaded.classify(r['text'], labels, hypothesis_template=TEMPLATE)['top_label'] for r in test_records[:8]]\n"
                "parity = {{'identical_labels': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_labels'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE, 'label_set': LABEL_SET}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'demo_results': [{{'id': i, 'top_label': r['top_label'], 'scores': {{e['label']: e['score'] for e in r['labels']}}, 'n_tokens': r['n_tokens'], 'decision_rule': r['decision_rule']}} for i, r in zip(text_ids, results, strict=True)], 'expected': expected}},\n"
                "    'labels': labels,\n"
                "    'hypothesis_template': TEMPLATE,\n"
                "    'comparison': comparison,\n"
                "    'new_messages': new_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen zero-shot model already reads most of the ten banking intents from their phrases — its test accuracy "
        "sits far above the 10 % majority floor — and a bounded fine-tuning of the last two decoder blocks and the NLI head "
        "on 800 entailment/contradiction pairs built from 400 messages lifts held-out accuracy and macro-F1 by more than ten "
        "points in a few minutes on CPU, recovering the phrases the frozen model read badly, with a 139 MB adapter that "
        "reloads to identical labels. That is the claim: the adaptation contract works end to end on a real labelled corpus, "
        "and the numbers it produces are read against the majority baseline and the frozen model rather than in isolation.\n\n"
        "The test split is 200 messages over ten balanced intents from one seeded split of one corpus, the metrics are "
        "exact-match accuracy and macro-F1 (neither a calibration measure), and Banking77 messages are short, English and "
        "single-intent. So a gain here says the contract works, not that the adapted model is better on your label set, that "
        "it separates intents whose phrases overlap, or that its scores mean anything as probabilities — they remain "
        "entailment-derived softmaxes. Fine-tuning towards ten phrases and one template also pulls the model away from "
        "general NLI: the adapted scorer still accepts any label set, but its zero-shot behaviour on other labels is "
        "changed, and nothing here measures that.\n\n"
        "Three things to carry to real data. **Baseline first:** the majority baseline and the frozen model's accuracy and "
        "per-label F1 on *your* labels, with *your* phrases and template, are the numbers to read before any adapted one — "
        "phrase wording moves zero-shot scores as much as the messages do. **Leakage:** de-duplicate messages across splits "
        "(the contract does this case-insensitively) and split by customer or conversation when several messages come from "
        "one. **Ceilings:** a message paired with a hypothesis over `MAX_TEXT_TOKENS` is refused at inference and truncated "
        "to 256 tokens only during training — long-document classification is out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled corpus, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against a trivial baseline and the frozen model on an independent split, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, classification quality on any other label set, a calibrated score or acceptance threshold, or "
        "production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_DECODER_LAYERS = 1` and compare the "
        "artifact size and the test scores; change `TEMPLATE` (for example `This message is about {{}}.`) and watch the "
        "frozen per-label F1 move before any training; set `MULTI_LABEL = True` in Section 5 and read the independent "
        "scores; or bring your own labelled messages through BYOD and read the majority baseline before the adapted "
        "number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/fairseq/tree/main/examples/bart\n"
        "- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (Lewis et al., 2019): https://arxiv.org/abs/1910.13461\n"
        "- Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach (Yin et al., EMNLP 2019): https://arxiv.org/abs/1909.00161\n"
        "- Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020; Banking77, CC BY 4.0): https://arxiv.org/abs/2003.04807\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
