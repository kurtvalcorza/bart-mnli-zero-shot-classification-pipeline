# Weight provenance and DIMER hosting

- Upstream: `facebook/bart-large-mnli`
- Immutable revision: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`
- Weight format: SafeTensors (`model.safetensors`, 1629437147 bytes, SHA-256 `cfbb687dbbd9df99fe865e1860350a22aebac4d26ee4bcb50217f1df606a018e`; 518 float32 tensors, 407 344 133 parameters)
- Upstream weight license: MIT (`license: mit` in the snapshot `README.md` front matter; the snapshot carries no separate `LICENSE` file)
- Local snapshot: `weights/bart-large-mnli/` with `dimer-base-manifest.json` (7 files, per-file bytes + SHA-256, `totalBytes` 1632153123); the Git repository does not vendor the checkpoint.
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/bart_zero_shot_classification_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch. Both run before `torch`/`transformers` are imported.
- DIMER hosting: MIT permits use, modification, distribution and commercial use provided the licence and copyright notice accompany copies; DIMER may mirror the pinned checkpoint in its model store under the upstream license with that notice.
- Loader trust boundary: `transformers==4.57.6` built-in `BartForSequenceClassification` + `AutoTokenizer`, `trust_remote_code=False`, `local_files_only=True` from the snapshot directory. Hub download is opt-in and pinned to the revision above.
