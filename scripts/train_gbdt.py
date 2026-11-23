#!/usr/bin/env python3
"""CLI to extract multi-band DSP/HRV features and train Model Family 1 (GBDT + Linear)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heartsounds.dsp.tabular_features import extract_recording_dsp_features  # noqa: E402
from heartsounds.models.gbdt import serialize_tabular_summary, train_cv_tabular_models  # noqa: E402
from heartsounds.storage.gcs import (  # noqa: E402
    CURATED_GCS_ROOT,
    DEFAULT_GCP_PROJECT,
    upload_file_to_gcs,
)


def _extract_worker(item: tuple[str, np.ndarray, int]) -> tuple[str, dict[str, float]]:
    rec_id, samples, sr = item
    return rec_id, extract_recording_dsp_features(samples, sr=sr)


def extract_table_features(waveform_df: pd.DataFrame, workers: int = 24) -> pd.DataFrame:
    """Extract DSP features in parallel across all rows of a curated waveform table."""
    tasks = [
        (str(row.recording_id), np.asarray(row.samples, dtype=np.float32), int(row.sample_rate))
        for row in waveform_df.itertuples(index=False)
    ]
    rows_by_id: dict[str, dict[str, float]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for rec_id, feats in tqdm(
            pool.map(_extract_worker, tasks, chunksize=16),
            total=len(tasks),
            desc="DSP features",
        ):
            rows_by_id[rec_id] = feats

    meta_cols = [
        "recording_id",
        "waveform_md5",
        "primary_raw_gcs_uri",
        "curated_label",
        "curated_subtype",
        "patient_group",
        "sample_rate",
        "split",
    ]
    meta = waveform_df[meta_cols].reset_index(drop=True)
    feat_df = pd.DataFrame([rows_by_id[rid] for rid in meta["recording_id"]])
    return pd.concat([meta, feat_df], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract DSP features & train Family 1 GBDT models."
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "v2_curated")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "outputs" / "runs" / "family1_dsp"
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--upload-gcs", action="store_true")
    parser.add_argument("--gcs-dest", default=CURATED_GCS_ROOT)
    parser.add_argument("--gcp-project", default=DEFAULT_GCP_PROJECT)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dsp_pq = args.data_dir / "dsp_features.parquet"

    if dsp_pq.is_file() and not args.force_extract:
        print(f"[family1] Loading cached DSP features from {dsp_pq}", flush=True)
        all_feats = pd.read_parquet(dsp_pq)
    else:
        print(
            "[family1] Extracting DSP features for train_val_folds, test, and quarantine...",
            flush=True,
        )
        dev_wav = pd.read_parquet(args.data_dir / "train_val_folds.parquet")
        test_wav = pd.read_parquet(args.data_dir / "test.parquet")
        quar_wav = pd.read_parquet(args.data_dir / "quarantine.parquet")
        combined_wav = pd.concat([dev_wav, test_wav, quar_wav], ignore_index=True)
        all_feats = extract_table_features(combined_wav, workers=args.workers)
        all_feats.to_parquet(dsp_pq, index=False)
        print(f"[family1] Saved {len(all_feats):,} feature rows to {dsp_pq}", flush=True)

    if args.upload_gcs:
        upload_file_to_gcs(
            dsp_pq,
            f"{args.gcs_dest.rstrip('/')}/dsp_features.parquet",
            project=args.gcp_project,
        )

    dev_feats = all_feats[all_feats["split"].str.startswith("fold_")].reset_index(drop=True)
    test_feats = all_feats[all_feats["split"] == "test"].reset_index(drop=True)
    quar_feats = all_feats[all_feats["split"] == "quarantine"].reset_index(drop=True)

    results = train_cv_tabular_models(dev_feats, test_feats, quarantine_features_df=quar_feats)
    summary = serialize_tabular_summary(results)
    print(json.dumps(summary, indent=2), flush=True)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, res in results.items():
        model_dir = args.out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        res.oof_df.to_parquet(model_dir / "oof_predictions.parquet", index=False)
        res.test_df.to_parquet(model_dir / "test_predictions.parquet", index=False)
        if res.quarantine_df is not None:
            res.quarantine_df.to_parquet(model_dir / "quarantine_predictions.parquet", index=False)
        res.feature_importances.to_csv(model_dir / "feature_importances.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
