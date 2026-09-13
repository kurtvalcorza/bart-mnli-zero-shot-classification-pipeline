# Release verification

`tutorials/bart_zero_shot_classification_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact notebook
revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for
the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/bart_zero_shot_classification_pipeline/pipeline.py` after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 7-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BARTZeroShotClassificationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `classify(text, labels, multi_label=MULTI_LABEL)`,
  the other-mode comparison call, `evaluation_report`), the ceiling print, the duplicate-label rejection probe,
  the descending-score and sum-to-one sanity checks, the identifier-keyed CSV export, the
  exported score semantics, the four exports, the learner-facing statements (no adaptation, score not a calibrated
  probability, argmax rule, sample-sanity and not-measurable verdicts, reject-not-truncate, the `classify` contract, never
  benchmark evidence) and the gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`,
  direct `transformers` or `huggingface_hub` calls **outside the carried module cell**, `BartForSequenceClassification`,
  `AutoTokenizer`, `softmax(`, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way); ~2 GB free RAM for the 407 M-parameter weights | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache, no pre-staged `model.safetensors` under `weights/bart-large-mnli/` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with **no repository checkout**, an empty Hugging Face
   cache, and no pre-staged `model.safetensors` under `weights/bart-large-mnli/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `MULTI_LABEL = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= the
   `pyproject.toml` pins (`torch==2.14.0`,
   `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executing (defining `BARTZeroShotClassificationPipeline`, `validate_inputs`, `evaluation_report`,
     `accuracy` and the ceilings) with no import of the repository package;
   - the three synthetic sentences, the three-label set and the expected labels authored in code with the text SHA-256 printed;
   - the inline `MANIFEST` asserted against the module identity and written to `weights/bart-large-mnli/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reporting `['model.safetensors']` fetched from
     `facebook/bart-large-mnli` at the immutable revision, `verify_snapshot` returning the
     7-entry manifest, and `BARTZeroShotClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory with `source 'local-snapshot'`;
   - ceilings `MAX_TEXT_CHARS = 8000`, `MAX_TEXT_TOKENS = 1024`, `MAX_LABELS = 32`, `MAX_LABEL_CHARS = 100`,
     `HYPOTHESIS_TEMPLATE = 'This example is {}.'`, `NUM_NLI_LABELS = 3`, `ENTAILMENT_INDEX = 2`, `CONTRADICTION_INDEX = 0`
     printed, and `validate_inputs` writing `outputs/bart_zero_shot_classification_input_manifest.json` (verdict `accepted`, three
     inputs `t1`–`t3`, the three hypotheses listed, one recorded rejection finding from the duplicate-label probe) before model
     execution;
   - `classify` returning three ranked entries per sentence with descending scores summing to one, all six sanity checks
     true for each sentence, the expected labels `travel`, `cooking`, `dancing` as the top labels (an observation, not a
     metric), and the multi-label comparison printed for the first sentence;
   - `evaluation_report` writing `outputs/bart_zero_shot_classification_evaluation_report.json` with verdict `sample-sanity`,
     one `accuracy` metric on 3 items, and the sample-sanity line printed;
   - `outputs/bart_zero_shot_classification_scores.csv` written with nine identified (text, label) rows and
     `outputs/bart_zero_shot_classification_result.json` written with the per-text results, score semantics, the other-mode
     comparison, `NOTEBOOK_SOURCE`, model identifier, immutable model
     revision, model licence, snapshot summary, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, NumPy,
   device), model identifier and immutable revision, whether the model cache and weights directory
   were clean, outcome, produced outputs, the per-sentence top label and scores (as
   observations, not a metric), and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/bart_zero_shot_classification_colab.ipynb` | | | | pending — queued to the GPU lane |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bart_zero_shot_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bart_zero_shot_classification_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

No execution of the notebook has been recorded. The only runtime measurements that exist for this
repository are the pipeline smoke run documented in `MODEL_CARD.md` (Windows venv, CPU float32,
`HF_HUB_OFFLINE=1`: load and verify 7.84 s; `classify` of the first sample sentence against the three labels 0.334 s →
`travel` 0.9939, `dancing` 0.0033, `cooking` 0.0029; the same call with `multi_label=True` 0.091 s → `travel` 0.9945,
`dancing` 0.0057, `cooking` 0.0018). That run exercised the package, not this
notebook, and is not notebook execution evidence.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | — | — | Default sample path | — | pending — queued to the GPU lane |

## Current status

The notebook source is complete and passes the static checks above; **no clean-runtime execution
has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
**The standalone carrier itself — executing the carried module cell in a runtime that has no repository
checkout — has been validated statically only (parity PASS) and never run end-to-end.** A carrier probe
did exec the install, carried-module and identity-assert cells in a fresh interpreter with the repository
package blocked on `sys.meta_path`, which confirms the cells define the public API without the package;
it fetched nothing and loaded no model. The clean run will therefore be the first execution of the
standalone path and of the staging path.
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row.
