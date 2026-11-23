# HeartSounds v2 (`ultrasound-heart-v2`)

Modular, auditable fetal Doppler ultrasound classification system predicting **Abnormal** vs. **Normal** fetal heart rate/rhythm conditions, built to jointly maximize sensitivity and specificity across the operating frontier.

## Data Provenance (`project=heartsounds`)

All curated datasets, split manifests, extracted DSP features, and model predictions are stored transparently in Google Cloud Storage under **`gs://data-processed-dualregion/v2_curated/`**:

- `README.md` & `curation_report.json` — Exact deduplication, case-insensitive label repair, conflict quarantine rules, split statistics, and SHA-256 checksums.
- `manifest.csv` & `manifest.parquet` — One row per unique recording (`recording_id`, `waveform_md5`, `primary_raw_gcs_uri`, `all_duplicate_raw_gcs_uris`, `original_parquet_labels`, `curated_label`, `curated_subtype`, `label_change_reason`, `patient_group`, `duration_s`, `sample_rate`, `split`).
- `raw_row_mapping.csv` — Maps every row in the legacy 8,281-row Parquet dataset to its deduplicated `recording_id` and `split`.
- `train_val_folds.parquet` — Leak-free 85% development set (`fold_0`..`fold_4` via patient-grouped `StratifiedGroupKFold`) at 4,000 Hz.
- `test.parquet` — Held-out 15% patient-grouped test set at 4,000 Hz.
- `quarantine.parquet` — Quarantined `NON_DIAGNOSTIC` and cross-folder conflict recordings.
- `dsp_features.parquet` — Extracted multi-band DSP, confidence-gated periodicity, HRV/Poincaré, and local ectopy features.
- `predictions/` — Standardized per-recording out-of-fold (`oof_predictions.csv`) and held-out test (`test_predictions.csv`) predictions, metrics JSONs, and plots for both **Evaluation 1 (100%-coverage binary)** and **Evaluation 2 (Green / Yellow: Re-record / Red triage)**.

## Package Architecture (`src/heartsounds/`)

The codebase follows strict acyclic layering (`schema/config` → `storage` → `data` / `dsp` → `models` → `training` / `eval` → `scripts`):

- `schema.py` & `config.py` — Typed enums, table contracts, and frozen YAML configuration dataclasses.
- `storage/gcs.py` — GCS upload/download and SHA-256 checksum helpers.
- `data/` — Pure label repair (`labels.py`), patient-folder grouping & splitting (`grouping.py`), dataset curation (`curate.py`), waveform augmentations (`augment.py`), and PyTorch masked sequence datasets (`torch_dataset.py`).
- `dsp/` — Multi-band Butterworth/Hilbert/Shannon envelopes (`filters.py`), confidence-gated autocorrelation & beat segmentation (`periodicity.py`), and interpretable HRV/ectopy tabular features (`tabular_features.py`).
- `models/` — Shared masked gated-attention MIL pooling (`pooling.py`), focal + pairwise ROC-AUC loss (`losses.py`), Family 1 GBDT (`gbdt.py`), Family 2 1D Multi-Scale Waveform+Envelope MIL (`waveform_mil.py`), Family 3 2D Log-Mel+Tempogram ConvNeXt MIL (`spec_tempogram_mil.py`), and model registry (`registry.py`).
- `training/` — Single-fold PyTorch AMP trainer (`trainer.py`) and 5-fold CV + test runner (`run_cv.py`).
- `eval/` — Pure binary ROC/PR/Youden-$J$/bootstrap CI metrics (`metrics.py`), 3-state clinical triage calibration (`triage.py`), out-of-fold stacking/fusion (`ensemble.py`), and diagnostic plots (`plots.py`).
- `deploy/` — Beaker `<= 2 GPUs` budget guard (`beaker_guard.py`) and 1-GPU smoke probe (`hello.py`).

## Reproducible Workflow

```bash
# 1. Install dependencies & run unit tests
uv sync --python 3.12 --group dev
uv run pytest -v
uv run ruff check .

# 2. Curate 4 kHz dataset, upload to gs://data-processed-dualregion/v2_curated/, and mirror to Beaker
uv run python scripts/curate_dataset.py --upload-gcs --create-beaker-dataset

# 3. Extract DSP features & train Family 1 (Logistic, XGBoost, LightGBM) on CPU
uv run python scripts/train_gbdt.py --upload-gcs

# 4. Verify Beaker GPU environment (1 GPU) & launch Neural Families 2 and 3 (<= 2 GPUs total)
uv run python scripts/launch.py --hello --submit --wait
uv run python scripts/launch.py --config configs/waveform_mil_5fold.yaml --submit
uv run python scripts/launch.py --config configs/spec_tempogram_mil_5fold.yaml --submit

# 5. Run ensemble fusion + Evaluation 1 (Binary) + Evaluation 2 (3-State Triage) & upload to GCS
uv run python scripts/evaluate_all.py --upload-gcs
```
