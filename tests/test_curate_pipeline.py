"""End-to-end curation pipeline tests verifying deduplication, provenance, and schemas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heartsounds.data.curate import RawRecord, curate_records, write_curated_bundle
from heartsounds.schema import validate_manifest_df, validate_waveform_df


def test_curate_deduplicates_and_writes_complete_provenance_bundle(tmp_path: Path):
    rng = np.random.default_rng(123)
    records: list[RawRecord] = []
    idx = 0

    # Create 28 normal subjects (2 clips each) + 14 abnormal subjects
    for s in range(28):
        for c in range(2):
            wav = rng.standard_normal(8000).astype(np.float32)
            records.append(
                RawRecord(
                    row_index=idx,
                    filename=f"heartsounds_data/normal/normals-20241116/subj_{s:02d}/clip_{c}.m4a",
                    legacy_label="NORMAL",
                    sample_rate=4000,
                    samples=wav,
                )
            )
            idx += 1

    # Add an exact duplicate normal row
    records.append(
        RawRecord(
            row_index=idx,
            filename="heartsounds_data/normal/normals-20241116/subj_00/clip_0_copy.m4a",
            legacy_label="NORMAL",
            sample_rate=4000,
            samples=records[0].samples.copy(),
        )
    )
    idx += 1

    # 14 abnormal subjects (including case-bug repaired & PAC)
    for s in range(14):
        wav = (rng.standard_normal(8000) * 2.0).astype(np.float32)
        if s == 0:
            fn = f"heartsounds_data/normal/normals-20241116/abn_subj_{s:02d}/clip ABNORMAL.m4a"
            leg = "NORMAL"
        else:
            fn = f"heartsounds_data/abnormal/allpacs/pac_{s:02d}_clip.m4a"
            leg = "PREMATURE_ATRIAL_CONTRACTION"
        records.append(
            RawRecord(row_index=idx, filename=fn, legacy_label=leg, sample_rate=4000, samples=wav)
        )
        idx += 1

    # 1 non-diagnostic recording -> quarantined
    records.append(
        RawRecord(
            row_index=idx,
            filename="heartsounds_data/abnormal/nondiagnostic.m4a",
            legacy_label="ABNORMAL_GENERIC",
            sample_rate=4000,
            samples=rng.standard_normal(8000).astype(np.float32),
        )
    )

    manifest_df, raw_map_df, wavs = curate_records(records, n_folds=5, seed=7)
    validate_manifest_df(manifest_df)
    assert len(raw_map_df) == len(records)
    assert len(manifest_df) == len(records) - 1  # 1 exact duplicate collapsed

    report = write_curated_bundle(manifest_df, raw_map_df, wavs, tmp_path)
    assert report["quarantined_recordings"] == 1
    assert (tmp_path / "manifest.csv").is_file()
    assert (tmp_path / "README.md").is_file()

    dev_df = pd.read_parquet(tmp_path / "train_val_folds.parquet")
    tst_df = pd.read_parquet(tmp_path / "test.parquet")
    quar_df = pd.read_parquet(tmp_path / "quarantine.parquet")
    validate_waveform_df(dev_df)
    validate_waveform_df(tst_df)
    validate_waveform_df(quar_df, allow_quarantined=True)
    assert set(dev_df["patient_group"]).isdisjoint(set(tst_df["patient_group"]))
