# Weight provenance and DIMER hosting

- Upstream: `facebook/bart-large-mnli`
- Immutable revision: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`
- Weight format: SafeTensors (`model.safetensors`, 1629437147 bytes, SHA-256 `cfbb687dbbd9df99fe865e1860350a22aebac4d26ee4bcb50217f1df606a018e`; 518 float32 tensors, 407 344 133 parameters)
- Upstream weight license: MIT (`license: mit` in the snapshot `README.md` front matter; the snapshot carries no separate `LICENSE` file)
- Local snapshot: `weights/bart-large-mnli/` with `dimer-base-manifest.json` (7 files, per-file bytes + SHA-256, `totalBytes` 1632153123); the Git repository does not vendor the checkpoint.
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/bart_zero_shot_classification_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch. Both run before `torch`/`transformers` are imported.
- DIMER hosting: MIT permits use, modification, distribution and commercial use provided the licence and copyright notice accompany copies; DIMER may mirror the pinned checkpoint in its model store under the upstream license with that notice.
- Loader trust boundary: `transformers==4.57.6` built-in `BartForSequenceClassification` + `AutoTokenizer`, `trust_remote_code=False`, `local_files_only=True` from the snapshot directory. Hub download is opt-in and pinned to the revision above.

## Adaptation corpus (not redistributed)

- Source: Banking77 (Casanueva et al., NLP4ConvAI 2020) — two CSV files from `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/`: `train.csv` 839,073 bytes SHA-256 `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` (10,003 rows) and `test.csv` 239,961 bytes SHA-256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` (3,080 rows); columns `text`, `category`; 77 intents, of which the tutorial keeps the ten in `LABEL_SET`.
- Licence: CC BY 4.0 (the repository licence). Attribution is carried in `CORPUS_LICENSE` and in every result export's `corpus` block.
- Handling: `fetch_corpus` verifies the byte count and SHA-256 of each file before it is parsed and caches the verified files under `weights/banking77/` (git-ignored). The repository ships no copy of the corpus; only the digests, the seed (`SAMPLE_SEED = 42`), the split sizes (`SAMPLE_SPLIT`) and the label set are committed, which reproduce the balanced 400 / 100 / 200 records and their digests exactly.
- Adapter artifacts: `save_artifact` writes `adapter.safetensors` (the trained `model.decoder.layers.*` and `classification_head.*` tensors only, 138,590,612 bytes for the default two blocks plus the head) plus `manifest.json` carrying `format` `org.valcorza.bart-large-mnli.adapter.v1`, the base `model_id` / `model_revision` / `weight_sha256`, the training labels and template, the tensor list and the file SHA-256; `load_artifact` refuses any mismatch. Adapters are derived works of the MIT-licensed weights and of whatever corpus trained them.
