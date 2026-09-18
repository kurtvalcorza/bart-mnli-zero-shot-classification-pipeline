# Release verification

`tutorials/bart_zero_shot_classification_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate**
until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable
release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 7-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned corpus
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BARTZeroShotClassificationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `label_names`, `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` with the duplicate-label refusal
  probe, `pipe.classify` with the sanity checks in both score modes, `majority_baseline`, `pipe.evaluate` on the
  frozen model and on the validation and test splits after adaptation with the accuracy assertions, `pipe.adapt` with
  its explicit hyperparameters and template, `evaluation_report` and `pipe.evaluate` on the unseen messages,
  `pipe.save_artifact`, `BARTZeroShotClassificationPipeline.from_artifact` and the reload-parity assertion, and the
  provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected `outputs/` paths, the
  learner-facing statements (an entailment-derived softmax is not a calibrated probability, the argmax rule,
  adaptation with gold labels, the entailment and contradiction pairs, the majority baseline, macro-F1, no dispersion
  estimate, pairs above the token ceiling rejected not truncated, named exclusions, the CC BY 4.0 corpus licence)
  and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository
  import on the primary path, a mutable `revision='main'`, direct `from transformers import` / `AutoTokenizer` /
  `BartForSequenceClassification` / `softmax(` / `from huggingface_hub import` / `urllib.request` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected NLI backend and corpus fetcher, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/bart-large-mnli/` or the corpus cache `weights/banking77/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub, and fetches the two pinned Banking77 files from the
   project repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `TEMPLATE = 'This customer message is about {}.'`, `MULTI_LABEL = False`,
   `EPOCHS = 2`, `LEARNING_RATE = 2e-5`, `BATCH_SIZE = 16`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BARTZeroShotClassificationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `accuracy`, `fetch_corpus`, `read_corpus`,
     `filter_records`, `build_sample_dataset`, `validate_dataset`, `label_names`, `check_split_disjoint`,
     `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `classification_metrics`, `majority_baseline`,
     `LABEL_SET` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `facebook/bart-large-mnli` at the immutable revision, and `verify_snapshot` returning its dict (7 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the two pinned files (839,073 / 239,961 bytes) from
     `raw.githubusercontent.com` into `weights/banking77/`, 10,003 + 3,080 raw rows read, and the balanced seeded
     draw of 400 / 100 / 200 records (40 / 10 / 20 per intent) with `check_split_disjoint` reporting no shared
     message and the three dataset digests `25a4a21e…` / `ab2f1c0a…` / `fe46d678…`; the ten label phrases printed;
     `outputs/…_train.csv` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_TEXT_CHARS` 8000, `MAX_TEXT_TOKENS` 1024, `MAX_LABELS` 32, `MAX_LABEL_CHARS` 100)
     and both decision rules surfaced; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`,
     one recorded rejection finding from the duplicate-label probe); `pipe.classify` on the three synthetic sentences
     with every sanity check `True` in both score modes (the card-pass smoke gave `travel` 0.9939 on the first);
   - Section 6: the majority baseline (accuracy 10.0, macro-F1 1.82 on the balanced split) and the frozen zero-shot
     model's test score (accuracy ≈ 82.50, macro-F1 ≈ 81.39 on CPU float32; `ATM support` and `card arrival`
     the weakest phrases), with the cell's assertion that the frozen accuracy beats the baseline;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 34,646,019 trainable of 407,344,131 parameters,
     2 pairs per record, and a two-epoch history with validation accuracy rising (≈ 84.00 → 96.00 → 98.00 in the
     recorded run; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the three-way comparison and per-label F1 and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test accuracy exceeds the frozen one —
     on the sample ≈ 97.00 versus ≈ 82.50, macro-F1 ≈ 96.99 versus ≈ 81.39);
   - Section 9: ten unseen training-file messages (one per intent) classified with `evaluation_report` returning
     `sample-sanity` and `pipe.evaluate` returning `measured-small-sample`, `outputs/…_predictions.csv` written;
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (56 tensors, about 139 MB)
     and `BARTZeroShotClassificationPipeline.from_artifact` reloading it with 8/8 identical top labels (the cell
     asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_format`, `weight_sha256`), the `corpus` block with the label set, the inference-contract
     items, the labels and template, the comparison, the artifact digest, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `bart_zero_shot_classification_colab.ipynb` (`E2E`) | `2559a76` / `9c599285` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `bart_zero_shot_classification_colab.ipynb` (`TASK-INFERENCE`, superseded) | `34098a7` / `f8a761ce5419` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-bart-zero-shot-classification` v1) | PASS — 8/8 cells ok (1 restart after install cell), 4 outputs verified; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bart_zero_shot_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bart_zero_shot_classification_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include the model download where it occurred; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `2559a76` / `9c599285` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 7 entries because the snapshot was pre-staged → `verify_snapshot` 7 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 400 / 100 / 200 balanced records drawn with `check_split_disjoint` clean and digests `25a4a21e…` / `ab2f1c0a…` / `fe46d678…` → four dataset refusals → input manifest + duplicate-label refusal probe → three synthetic sentences classified in both score modes with every sanity check `True` → majority baseline → frozen evaluation → `adapt` → validation + test evaluation → ten unseen messages → adapter export → reload parity) | 342.1 s | **PASSED** — 11/11 code cells; majority accuracy 10.0 / macro-F1 1.82; frozen zero-shot test 82.5 / 81.39 (54.2 s; per-label F1 from 33.3 on `ATM support` to 100 on three phrases); `adapt` 34,646,019 of 407,344,131 params, 800 pairs from 400 messages, 2 epochs, 192.6 s, validation accuracy 84.0 → 96.0 → 98.0 (`best_epoch` 2, train loss 0.289 → 0.167); **adapted test accuracy 97.0 / macro-F1 96.99 (Δ +14.5 / +15.6; every per-label F1 ≥ 92.3)**; ten unseen messages `sample-sanity` accuracy 0.8, `measured-small-sample`; adapter 138,590,612 B / 56 tensors, SHA-256 `30e53f03…`; reload parity 8/8; six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `34098a7` / `f8a761ce5419` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-bart-zero-shot-classification` v1) | Default sample path of the inference-only notebook: three synthetic sentences, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot`, `classify` in both modes, `sample-sanity` report | 280.7 s | **PASSED** — 8/8 code cells (1 restart after the install cell), 4 outputs verified, 1632 MB staged; does not cover the `E2E` blob |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and balanced split, the inference contract in both score modes, the majority
baseline, two epochs of decoder-and-head fine-tuning on NLI pairs, held-out evaluation, unseen-message
classification, adapter export and reload parity — which catches defects but is **not** a supported runtime under
REL1/REL10, and it ran with the snapshot and the two Banking77 files pre-staged, so neither the 1.6 GB Hub fetch nor
the corpus download has been exercised by this notebook end to end; the earlier `TASK-INFERENCE` Kaggle run did
exercise the Hub fetch and digest check of the same snapshot. The repository stays at **Candidate** until a Colab or
fresh-container run of the exact `E2E` release revision is recorded above.
