# BART-large MNLI Zero-Shot Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/bart-mnli-zero-shot-classification-pipeline`  
**Notebook:** `tutorials/bart_zero_shot_classification_colab.ipynb`  
**Reviewed commit:** `68933e2104152b4300e8d5d695e979432deafd91` (`main`, confirmed with `gh api repos/kurtvalcorza/bart-mnli-zero-shot-classification-pipeline/commits/main`)  
**Notebook Git blob:** `f652fd671322b9ff0a7ca975ce26a0c33f729f38`, the same blob as at `6141d4f` (the only later commit, `b64fbf0`, changes `docs/release-verification.md` only)  
**Finding prefix:** `ZSC`

## Executive assessment

This is one of the stronger notebooks in the fleet. It already runs the pinned stack in an isolated `uv` environment, and the one hosted run of this exact blob (Colab T4, 2026-09-27) completed `Run all` in one pass, 15/15 code cells, with no restart. It carries its three modules byte for byte (generator `--check` exits 0, release validator passes), pins and digest-verifies the model and a real licensed corpus, fits the majority baseline on training labels, keeps validation for epoch selection, reports per-class results with confusion matrices and paired changes, runs a controlled category-wording comparison on a freshly loaded pretrained model, and checks reload parity on all 200 test messages with a tolerance. Its orientation (driving question, Input → Model → Output, how to use, roadmap, predictions, "What to notice") and its limits prose are careful.

Two problems stand in the way of `Ready for intended use`:

1. Every rerun the notebook prescribes reuses the already-adapted model. The BYOD instruction ("re-run from that cell") and the template experiment ("watch the pretrained per-class F1 move before any training") make Section 6 score the adapted model under the label "pretrained", and Section 7 then fine-tunes on top of it under an epoch-0 "frozen model" label. The `TRAINABLE_DECODER_LAYERS = 1` experiment ends in a bare `AssertionError` at the reload-parity check, because the exported artifact omits a layer the evaluated model still carries (ZSC-M1).
2. The BYOD contract's stated minimum (8 records) is not the real one (40–48 in the probes), the rejection message names neither the split nor the real minimum, a category too small to reach the training split passes Section 4 and fails in Section 6, and a text under the 8,000-character guard but over the 1,024-token ceiling passes Section 4 and fails inside Section 7's epoch-0 validation without naming the record (ZSC-M2).

The default path is not shown to be wrong. The problems are the validity of every non-default run and the BYOD promise, plus stale install text, a few runtime and guided-layer gaps, and a fleet-pattern deviation in how the isolated environment is built.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.2** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` `b1cfe13` |
| Intended audience | "Level 2 — Applied tasks", self-paced, "basic Python and Colab familiarity and no prior machine-learning experience" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12", CPU float32 by default, CUDA when present |
| Promised outcomes | One-pass pinned install into an isolated environment; carried modules; staged, digest-verified snapshot; Banking77 fetch, ten intents, 400/100/200 balanced disjoint split, four refusal probes; one text-hypothesis pair with three NLI logits; inference contract in both score modes; training-fitted majority baseline and pretrained model on the test split; bounded fine-tuning of the last two decoder blocks and the NLI head with validation selection; three-way comparison with per-class F1, confusion matrices and paired changes; category-wording activity (sets A/B, pretrained, validation); ten unseen messages; adapter export with its classification setup and full-test reload parity; BYOD "through the same … cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`, `isolated_runtime` template key) + `tools/notebook_template.py`; carried modules from `src/bart_zero_shot_classification_pipeline/` @ `4864c5a` |

### Evidence actually obtained

- **Source inspection:** all 33 cells (15 code), the three carried modules (`pipeline.py` `from_pretrained`, `classify`, `evaluate`, `adapt`, `save_artifact`, `load_artifact`; `samples.py` and `metrics.py` in full), the generator and template, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, `README.md`, the CI workflow.
- **Documented execution evidence:** `docs/release-verification.md`, recorded-executions row of 2026-09-27: Google Colab T4, commit `6141d4f` / **blob `f652fd671322`, the reviewed blob**, `Run all` with no restart, 15/15 code cells, kernel Python 3.13.15, isolated environment `torch 2.14.0+cu130` / `transformers 4.57.6`, `cuda:0`; test accuracy / macro-F1 majority 10.0 / 1.82, pretrained 82.5 / 81.39, adapted 96.0 / 95.96; reload parity 200/200. The executed file itself is not archived in the workspace; this review relies on the record. Also recorded: the preceding blob's Colab failure at the in-kernel install (the reason for the isolated runtime) and local pre-flights (not promotion evidence).
- **Direct execution (this review):** `run_probes.py`, Windows, Python 3.12 (`eo-notebook-test` env, torch 2.13.0+cpu, transformers 4.57.6), CPU. **No BART-large weights, no downloads.** Static: JSON parse, compile of all 15 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0), `uv venv` behaviour on Windows. Model probes use a **tiny randomly initialised `BartForSequenceClassification`** built from the snapshot's own `config.json` (12 decoder layers kept, widths shrunk) with the snapshot's real tokenizer files, staged with its own manifest. The notebook's carried cells (9, 11, 13) and learner cells (7, 17–31) are executed **verbatim from the notebook JSON** in one namespace, as the isolated worker would; only form-field literals are substituted (as an executor sets fields; `LEARNING_RATE = 1e-3` so the random stand-in moves), Section 3's download cell is replaced by loading the stand-in, and the carried `WEIGHT_SHA256` constant is set to the stand-in's digest. The real Banking77 files came from the local digest-verified cache. The stand-in exercises the carried control flow, not BART-large-MNLI's numerics.
- **Not verified:** the Colab executed file itself; any Kaggle or hosted-CPU run of this blob; any real BART-large forward pass in this review; BYOD through the real upload widget and the real model; the optional experiments on the real model; learner understanding.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Strong on the default path: immutable revision, per-file digests, digest-pinned corpus, no remote code, validation before model work, transactional `adapt`, digest- and tensor-set-checked artifact loading, full-test parity with tolerance. One-pass install verified on Colab. Defects: `adapt` and the Section 6 "pretrained" evaluation are not idempotent across prescribed reruns, and a changed `TRAINABLE_DECODER_LAYERS` exports an artifact that is not the evaluated model (ZSC-M1); BYOD split-size, class-coverage and token-ceiling checks happen after Section 4 or not at all (ZSC-M2); the isolated environment inherits the kernel's interpreter, is not hash-locked and is POSIX-only (ZSC-m1). |
| Scientific / experimental validity | Good default design: release partition, balanced draw, text-disjoint check, training-fitted baseline, validation-only selection, test used once per system, paired changes, wording comparison on the pretrained checkpoint with only the wording changed, explicit no-dispersion caveat. Weakness: every non-default run silently replaces the pretrained reference with the adapted model (ZSC-M1). |
| Promise fulfilment | Every listed default stage ran in the documented Colab run. Not met: BYOD "through the same … cells" at the stated minimum and its token ceiling (ZSC-M2); the optional experiments and BYOD rerun give invalid "pretrained" numbers or crash (ZSC-M1); BYOD Section 11 "in none of the splits" (ZSC-m3). |
| Learner experience | Good: orientation, roadmap, Input → Model → Output, Infrastructure labels, predictions before principal stages, "What to notice" after them, three worked hints, conclusion scaffold, transfer guidance. Gaps: no troubleshooting section or glossary, no worked answers at Sections 6–8, optional experiments without rerun instructions (ZSC-m5); Section 1's "Record the runtime" text still describes an in-kernel install and restart (ZSC-m2); runtime figures do not name their environment (ZSC-m4). |
| Spec conformance | `Run all` / RUN1, RUN10, ENV6: met on the documented Colab run. Unmet applicable `MUST`s: DAT13, DAT14, UX7, RUN9 (ZSC-M1); DAT12, DAT19, VAL1, VAL5, VAL6, VAL7, REL12 (ZSC-M2); SRC3 (ZSC-m2); INF2 (ZSC-m3); UX12 (ZSC-m4); ENV1/ENV9 partly (ZSC-m1). `SHOULD` gaps: GDL6, GDL9, GDL10, GDL13, UX10, EXE2. |

## 3. Promise and objective tracing

| Claim (opening / section text) | Implementation | Observable result | Learner interpretation | Holds? |
|---|---|---|---|---|
| `Run all` in a fresh runtime, no restart, no configuration edit | Cells 3 and 5 (uv environment + router) | Colab T4 record: 15/15, no restart, pinned versions in the isolated env | Clear | Yes (documented) |
| Snapshot staged and digest-verified, no fallback | Cell 15 | Colab record: 7 files verified | Clear | Yes (documented) |
| Corpus digest-pinned, split without leakage | Cell 17 | Colab record: 400/100/200, three digests; probe P3a: verbatim cell runs on the real cached files | Clear | Yes |
| Baseline fitted on training labels; pretrained model scored on test | Cell 21 | Colab record: 10.0 / 1.82 and 82.5 / 81.39 | Clear | Yes (default path); **No** after a prescribed rerun (ZSC-M1) |
| Bounded adaptation with validation selection | Cell 23, `adapt` | Colab record: adapted 96.0 / 95.96, best epoch 2 | Clear | Yes (default path) |
| Wording activity changes only the wording, on the pretrained checkpoint | Cell 29, fresh `from_pretrained` | Probe P3a: checkpoint `pretrained (not adapted)`; Colab: A 84.0 / B 82.0, 17 changed | Clear | Yes |
| Adapter reloads with prediction, score and metric parity | Cell 31 | Colab: 200/200, max diff 0.0 | Clear | Yes (default path); **No** after the suggested 1-layer experiment (ZSC-M1) |
| Optional experiments "do not affect the default path" | Interpretation cell 32 | Probe P3b/P4: reruns score adapted as pretrained; 1-layer rerun stops at `AssertionError` | — | **No** (ZSC-M1) |
| BYOD passes "through the same validation, … reload-parity cells"; "a dataset needs 8..20,000 records" | Cell 17 BYOD branch, `split_dataset`, per-split `validate_dataset` | Probe P5: 8–47 records rejected; rare class fails in Section 6; over-ceiling text fails in Section 7 | — | **No** (ZSC-M2) |
| Section 11 messages "were in none of the splits" | Cell 31 | Under BYOD they are `test_records[:10]` (probe P6) | — | **No** under BYOD (ZSC-m3) |

| Learning objective (opening cell) | Learner activity | Evidence the objective is exercised |
|---|---|---|
| (1) Explain text + descriptions → model → scores and prediction | Read the Input → Model → Output contract; run cell 19 | Exercised by reading and observation; worked hint in Section 5 |
| (2) Explain zero-shot classification through NLI | Predict the winner and the wrong pair's logit; read three logits | Exercised (prediction + worked interpretation) |
| (3) Distinguish zero-shot inference from task adaptation | Read Sections 5–7 prose | Exercised by reading; no checkpoint asks the learner to state the difference |
| (4) Compare baseline, pretrained and adapted models on the same messages | Predict before Sections 6 and 8; read comparison, per-class table, paired changes | Exercised on the default path; invalid on any rerun (ZSC-M1) |
| (5) Measure sensitivity to category wording through a controlled experiment | Predict changed count; read set A vs B and paired changes | Exercised as observation; the learner does not make the change themselves (ZSC-m5) |
| (6) Interpret accuracy, macro-F1, per-class results, confusion and limits | Conclusion scaffold | Exercised; worked answers only in Sections 5, 9, 10 |

## 4. Prioritized findings

### ZSC-M1 — Major: prescribed reruns score the adapted model as "pretrained", stack a second adaptation, and the suggested 1-layer experiment stops at the reload-parity assert

**Cell/section:** Opening BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell", `tools/notebook_template.py` line 45); "Optional experiments" in Interpretation (line 670: `TRAINABLE_DECODER_LAYERS = 1`, change `TEMPLATE` "and watch the pretrained per-class F1 move before any training", BYOD); Sections 6, 7, 11 (cells 21, 23, 31; template lines 333, 384, 614); carried `adapt` and `save_artifact` (`src/bart_zero_shot_classification_pipeline/pipeline.py` lines 496–651).

**Observed issue:** `pipe` is built once, in Section 3. `adapt()` trains from the model's **current** weights: `initial_state` is only used to roll back on an exception, nothing restores the base at entry, and it still writes `"note": "frozen model"` on epoch 0. After the default run, rerunning as the notebook instructs gives:
- Section 6's `pretrained_test` scores the adapted model (only the unprinted `adapted: True` field says so), so Section 8's comparison and `evaluation_report.json` call adapted-versus-re-adapted "pretrained versus adapted", and the template experiment's "pretrained per-class F1 before any training" is not pretrained;
- Section 7 fine-tunes on top of the previous adaptation under the "frozen model" epoch-0 label;
- with `TRAINABLE_DECODER_LAYERS = 1` rerun from Section 7, decoder layer 10 keeps the first run's trained weights in `pipe`, but `save_artifact` writes only the latest `trainable_names` (layer 11 + head). The reloaded artifact lacks the layer that produced the evaluated scores, and cell 31 stops at `assert parity[...]` with an empty `AssertionError` after the artifact has been written. The cell-4 note "to start over, restart the session and choose Run all" is the only route to a valid experiment, and none of the experiment or BYOD instructions says so.

The wording activity is not affected: it loads a fresh pretrained pipeline.

**Consequence:** A learner who follows the BYOD instruction or any suggested experiment gets "pretrained" and "adapted" numbers that do not mean what the headings and exported report say, or a bare assertion failure. The conclusion the unit teaches (what adaptation adds over the pretrained model) becomes invalid for every non-default run, including the BYOD transfer path.

**Evidence:** Source inspection of `adapt` and `save_artifact`. Direct execution, stand-in model, cells run verbatim (probes P3b, P4, P4c): after the default path, rerunning Sections 4–6 with the template experiment gives `evaluate(...)['adapted'] = True` for Section 6's "pretrained" score, with 166 of 200 predictions differing from a freshly loaded pretrained model under the same template; the second `adapt` labels epoch 0 "frozen model" and starts from adapted weights. Rerunning Sections 7–11 with `TRAINABLE_DECODER_LAYERS = 1`: `pipe` layer 10 still holds the first run's weights, the artifact contains decoder layer `[11]` only, the reloaded layer 10 equals the base, parity is 160/200 with max score difference 0.0123, and cell 31 raises `AssertionError`. Control: the same 1-layer setting from a fresh start passes parity 200/200, max difference 0.0. Real-model behaviour is inferred from the same control flow, **not executed**.

**Recommended correction:** Make each pass start from the pretrained base: have Section 4 rebuild `pipe` with `BARTZeroShotClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` from the already-verified files (as cell 29 already does), or add a `reset_adapter()` that restores base tensors and clears `self.adapter`, called before Sections 6 and 7. Alternatively, have `adapt()` refuse when `self.adapter is not None` and Section 6 refuse to label an adapted pipe "pretrained". Make `save_artifact` refuse, or include every tensor that differs from the base, when the adapted set and `trainable_names` disagree, and give the parity assert a message naming the cause. State, for each experiment and for BYOD, exactly which section to rerun from. Generator: `tools/notebook_template.py` lines 45, 333, 384, 614, 670; `pipeline.py` `adapt` / `save_artifact`.

**Acceptance check:** On a stand-in or the real model, running the default path and then rerunning from Section 4 as instructed gives Section 6 `adapted: False` and predictions equal to a freshly loaded pretrained pipeline; the second `adapt` starts from base weights; after a `TRAINABLE_DECODER_LAYERS = 1` rerun as instructed, cell 31 passes parity and the artifact's tensors equal `pipe`'s on every decoder tensor that differs from the base.

**Spec:** DAT13, DAT14, UX7, RUN9 (MUST); VER5 (context); GDL10, UX5 (SHOULD).

### ZSC-M2 — Major: the BYOD contract's stated limits are not the enforced ones, and three classes of bad input fail after Section 4 or with misleading messages

**Cell/section:** Prerequisites data contract ("a dataset needs 8..20,000 records", template line 144) and Section 4 BYOD branch (cell 17, template line 179); `split_dataset`, `validate_dataset` (`samples.py`); `label_names(train_records)`; the token check in `classify` (`pipeline.py` `_check_input_tokens`); the 256-token training truncation in `adapt`.

**Observed issue:**
1. Cell 17 calls `split_dataset` (test 20 %, validation 15 %, stratified, rounded per label) and then `validate_dataset(part)` on **each split**, which requires at least 8 records per split. A BYOD file that meets the stated "8..20,000 records" is rejected with messages such as `2 records; 8..20000 are required` or `split leaves 4 training records; at least 8 are required`, which name neither the split nor the real minimum.
2. Classes are not checked for coverage. A label with too few records lands only in the test split; `labels = label_names(train_records)` omits it; Section 4 passes, and Section 6 fails with `gold labels outside the declared vocabulary: ['category rare']`.
3. Section 4 checks the 8,000-character guard but not the 1,024-token pair ceiling. A text under 8,000 characters can exceed 1,024 tokens. In the validation or test split it fails inside `pipe.evaluate` in Section 6 or in Section 7's epoch-0 validation, after the pretrained evaluation, with `a premise+hypothesis pair is 2756 tokens; ceiling is MAX_TEXT_TOKENS=1024` and no record id. In the training split it is silently cut to 256 tokens; the Prerequisites state the rule, but the run never reports how many pairs were cut.
4. An empty or cancelled upload raises a bare `StopIteration`.

**Consequence:** A learner who follows the stated contract with a small dataset is rejected with a message that contradicts it; one with an unevenly labelled or long-text dataset fails mid-run without being told which record or class to fix. The BYOD promise is not met at its own boundaries.

**Evidence:** Direct execution, cell 17 verbatim with `USE_BYOD = True` and a fake upload (probe P5): for 2 balanced labels, every balanced file of 8 to 46 records is rejected and 48 is the smallest accepted; for 10 balanced labels, 40 is the smallest accepted (10, 20 and 30 rejected). An 81-record file with one single-record class passes cell 17 and 19 and fails cell 21 with the error above. A 6,000-character record of 2,756 tokens passes cell 17 and lands in validation; cell 21 passes; cell 23 fails with the token error, `error_names_record_id: false`; five similar records in the training split are truncated with no report in cell 23's output. Empty upload: `StopIteration`. Positive control: a 3-label × 30-record BYOD file runs cells 17–31 to the end on the stand-in (probe P6). Real upload widget and real model: **not verified**.

**Recommended correction:** In Section 4, before any model call: state and enforce the real minimum (or validate validation/test with `min_records=1` while keeping the training minimum), and name the split in the message; require every label to reach the training split (or report and drop it); count tokens for every BYOD text against `MAX_TEXT_TOKENS` with the longest hypothesis (the tokenizer is loaded in Section 3) and reject over-ceiling validation/test texts by id; report the number of training pairs that will be truncated to 256 tokens; guard the empty upload with a recovery message. Generator: cell 17 in `tools/notebook_template.py` (around line 179) and the Prerequisites text (line 144).

**Acceptance check:** A 20-record, 2-label BYOD file either runs through Section 4 or is rejected in Section 4 with a message naming the real minimum and the split; a file with a single-record class is rejected in Section 4 naming the class; a file with one 2,700-token record in validation is rejected in Section 4 naming its id; a training split with long texts prints the number of truncated pairs; an empty upload prints a recovery instruction.

**Spec:** DAT12, DAT19, VAL1, VAL5, VAL6, VAL7, REL12 (MUST); UX10 (SHOULD).

### ZSC-m1 — Minor: the isolated environment inherits the kernel's interpreter, is not hash-locked and is POSIX-only

**Cell/section:** Section 1 install cell (cell 3) and router (cell 5); `_ISOLATED_INSTALL` and `_ISOLATED_ROUTER` in `tools/build_notebook.py` (lines 78–130).

**Observed issue:** The restart claim holds: the Colab record shows one-pass `Run all` with no restart, so the uv approach does what it is for. The build differs from the fleet's reference uv isolated-environment pattern in three ways. (a) `uv venv --python sys.executable` reuses the kernel's interpreter, so on Colab the isolated environment ran on the kernel's Python 3.13.15 (inferred from the command plus the record's kernel version) while the notebook states "Python 3.12" as its runtime. (b) Only the eight principal pins are fixed; transitive dependencies resolve at run time, with no hash lock and no `--only-binary`. (c) `ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"`, and the router uses `pass_fds` and `start_new_session`, all POSIX-only, so a Windows Jupyter kernel cannot run the notebook although "Jupyter" is a stated runtime.

**Consequence:** The resolved environment can drift between runs and platforms beyond what the notebook says it controls, and the stated Python version is not the one that ran on Colab. A Jupyter user on Windows gets an install or `Popen` failure in Section 1.

**Evidence:** Source inspection (probe P1). Direct execution: `uv venv --python <python>` on Windows creates `Scripts\python.exe`, not `bin/python` (P1 `windows_uv_venv`). Documented execution: Colab record, kernel Python 3.13.15. The isolated environment's Python version on Colab is **inferred**, not recorded.

**Recommended correction:** Adopt the fleet's uv isolated-environment pattern as implemented on `main` in `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`): create a managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), install a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and run the stages in that environment. Either support `Scripts\python.exe` and a Windows-safe channel, or state that the supported Jupyter runtime is Linux/macOS. Change `_ISOLATED_INSTALL` / `_ISOLATED_ROUTER` in `tools/build_notebook.py` and regenerate; re-qualify with a one-pass hosted `Run all`.

**Acceptance check:** Section 1 creates the environment from a pinned managed interpreter and installs from a hash-locked file; the runtime record prints the pinned Python version on Colab; the stated runtime matches what ran; on Windows Jupyter the notebook either runs or the opening states it is unsupported.

**Spec:** ENV1, ENV9 (MUST, partly); ENV3 (the recorded Python is the kernel's, not stated as the isolated one).

### ZSC-m2 — Minor: stale install and status text survives the isolated-runtime change

**Cell/section:** "Record the runtime" markdown (cell 6, `tools/build_notebook.py` lines 544–552); cell 7's in-kernel install guard; `tutorials/README.md` table "Run-all" column; `docs/release-verification.md` "Automatic coverage"; CI step label.

**Observed issue:** Cell 6 says the dependency set "is installed directly" and that the cell "stops with a restart instruction" if a pin replaces a loaded distribution; in the routed path the cell skips installation, and the restart guard only runs when an executor disables routing. `tutorials/README.md` still says Run-all "pending — the revised notebook has no hosted run yet", while `docs/release-verification.md` records the Colab T4 PASS of this blob. The validator description still lists "the pinned-install cell with its restart-on-stale-import guard"; the CI step is labelled "NOTEBOOK_SPEC 2.0 PAR3".

**Consequence:** A learner reading Section 1 is told a restart may be needed, which the isolated runtime exists to prevent; a maintainer reading the tutorials index thinks no hosted run exists.

**Evidence:** Source inspection; probe P2.

**Recommended correction:** Rewrite the cell-6 text for the routed path (records versions; the guard applies only when `DIMER_NOTEBOOK_CI_PREINSTALLED=1` runs the notebook in the executor's kernel); update the `tutorials/README.md` Run-all cell to cite the 2026-09-27 Colab run; refresh the release-verification coverage bullet and the CI label.

**Acceptance check:** No learner-facing cell on the routed path mentions a restart instruction from installing; `tutorials/README.md` and `docs/release-verification.md` agree on the hosted-run state of the current blob.

**Spec:** SRC3 (MUST: knowingly stale instructions).

### ZSC-m3 — Minor: under BYOD, Section 11's "new" messages are the first ten test records

**Cell/section:** Section 11 markdown ("Ten messages that were in none of the splits", template line 555) and cell 31 (template line 575).

**Observed issue:** With `USE_BYOD = True`, `new_records` are `test_records[:10]` renamed `new-00…`; the heading still says they were in none of the splits. The `sample_kind` string is honest (`BYOD test records`); the prose is not.

**Consequence:** A BYOD learner is told the inference demonstration is on unseen data when it reuses scored test data.

**Evidence:** Source inspection; probe P6 (`section11_new_records_are_test_records: true`).

**Recommended correction:** Hold ten BYOD records out of the split for Section 11, or make the markdown say that under BYOD these are test records.

**Acceptance check:** Under BYOD, Section 11's text and `sample_kind` agree with where the records came from.

**Spec:** INF2 (MUST where task semantics permit).

### ZSC-m4 — Minor: runtime figures do not name their environment

**Cell/section:** Opening ("on CPU the model stages take roughly ten minutes after the downloads", template line 42) and Prerequisites ("about 0.3 s" per message, "one training epoch over 800 NLI pairs about a minute", line 143).

**Observed issue:** The figures are consistent with the recorded local pre-flights (Windows CPU: pretrained test 54.2 s for 200 messages, adaptation 192.6 s for two epochs plus three validation passes; Linux CPU: adaptation 320 s, total 784 s), but no environment is named and none is labelled an estimate. The Linux pre-flight implies about 90 s per epoch, not "about a minute". No hosted CPU run exists.

**Consequence:** A learner on a hosted CPU runtime cannot judge whether a slow run is normal.

**Evidence:** Source inspection; `docs/release-verification.md` recorded-executions table.

**Recommended correction:** Label each figure with its environment (e.g. "local Windows CPU pre-flight"), add the Colab T4 figures, and mark hosted-CPU times as estimates.

**Acceptance check:** Every runtime figure in the notebook names its environment or is labelled an estimate and agrees with a recorded run.

**Spec:** UX12 (MUST).

### ZSC-m5 — Minor: guided-layer gaps — no troubleshooting or glossary, worked answers missing at Sections 6–8, and the "change one thing" activity changes nothing the learner chooses

**Cell/section:** Whole notebook; Sections 6, 7, 8 and 10; Interpretation "Optional experiments".

**Observed issue:** The guided layer is largely present (see Executive assessment). Missing: a troubleshooting section for expected hosted failures (memory, download, isolated-worker exit, BYOD rejections); a glossary collecting the recurring terms (premise, hypothesis, entailment, macro-F1, confusion matrix, adaptation, epoch); collapsible worked answers after the predictions in Sections 6, 7 and 8 (the three `<details>` blocks are in Sections 5, 9 and 10). Section 10 is controlled and well bounded, but set B is fixed, so the learner predicts and observes without making a change. The optional experiments are one paragraph without rerun instructions or interpretation guidance, and following them gives invalid results (ZSC-M1).

**Consequence:** A self-paced learner who hits a failure has no recovery guide, and the principal comparisons offer no answer to check a prediction against.

**Evidence:** Source inspection of all 18 markdown cells.

**Recommended correction:** Add a troubleshooting section and a short glossary; add "Check your reasoning" answers after Sections 6–8; after ZSC-M1 is fixed, turn one optional experiment (e.g. editing one set-B description, or the template) into a Predict → Change → Run → Observe → Explain activity with exact rerun instructions. Generator: `tools/notebook_template.py`.

**Acceptance check:** The regenerated notebook has a troubleshooting section, a glossary, a worked answer after each principal prediction, and at least one activity naming the change, the cells to rerun and what to compare.

**Spec:** GDL6, GDL9, GDL10, GDL13, UX5 (SHOULD).

### Suggestions

- **ZSC-S1:** Render the confusion matrices with `display(Markdown(...))` instead of printing raw pipe-table text, and print the column key (P1…P10 → description) once.
- **ZSC-S2:** Note the GPU-versus-CPU difference in the adapted score (Colab/Kaggle T4 96.0 versus CPU pre-flights 97.0) in Section 8, so learners do not read a one-point difference as a defect (ENV8 context).
- **ZSC-S3:** Print a bootstrap interval over the 200 test messages for the pretrained-versus-adapted accuracy and macro-F1 deltas; it makes the no-dispersion caveat concrete.
- **ZSC-S4:** When the router cell is rerun it silently replaces the worker and every variable created so far; print a warning that a full `Run all` is now required.

## 5. Readiness

**Needs revision.** Two Major findings are open. Applicable `MUST`s unmet: DAT13/DAT14/UX7/RUN9 (ZSC-M1); DAT12/DAT19/VAL1/VAL5/VAL6/VAL7/REL12 (ZSC-M2); ENV1/ENV9 in part (ZSC-m1); SRC3 (ZSC-m2); INF2 (ZSC-m3); UX12 (ZSC-m4).

Met: one-pass `Run all` on Colab T4 for this exact blob (RUN1, RUN10, ENV6, REL1, REL2, REL10 recorded), so the default path is not blocked.

Remaining gates after fixes: a one-pass hosted `Run all` of the regenerated blob (and, if ZSC-m1 is adopted, of the managed-Python environment); a BYOD positive and negative run through Section 11 on a hosted runtime (REL12); a rerun of an optional experiment, as instructed, showing a pretrained baseline equal to a fresh load; the maintainer's review of description set B (already named in `STATUS.md`).

## 6. Verified versus inferred

- **Verified by direct execution (CPU, this review):** notebook parse and compile (15/15 code cells), blob `f652fd67`, generator `--check` exit 0, release validator exit 0; Windows `uv venv` layout (P1); stale text (P2); default path cells 17–31 verbatim on the stand-in, including the fresh pretrained checkpoint for the wording activity and 200/200 parity (P3a); rerun state and parity failure, with a fresh-start control (P3b, P4, P4c); BYOD minimum, rare-class, token-ceiling and empty-upload behaviour (P5); BYOD positive path and Section 11 records (P6).
- **Verified from documented execution:** the Colab T4 one-pass `Run all` of this exact blob and its figures, as recorded in `docs/release-verification.md` (the executed file was not inspected).
- **Inferred, not executed:** real-model behaviour for ZSC-M1 (scale of the "pretrained" error on BART-large; the parity failure follows from the same tensor set) and ZSC-M2 through the upload widget; the isolated environment's Python version on Colab (ZSC-m1); learner understanding.
- **Finding most likely to be wrong:** ZSC-m1's Python-version point. `uv venv --python sys.executable` should reproduce the kernel's 3.13.15, but the record does not print the isolated interpreter's version, and Colab's kernel version may differ from the one recorded.

Probe ZIP: `bart_zero_shot_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
