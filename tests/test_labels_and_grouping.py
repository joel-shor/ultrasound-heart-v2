"""Unit tests for pure label resolution (`labels.py`) and patient-group splitting (`grouping.py`)."""

from __future__ import annotations

import pandas as pd

from heartsounds.data.grouping import (
    assign_patient_grouped_splits,
    extract_patient_group_single,
    resolve_patient_group,
)
from heartsounds.data.labels import resolve_recording_label
from heartsounds.schema import BinaryLabel, LabelRepairReason, Subtype


def test_case_bug_repaired_to_abnormal():
    res = resolve_recording_label(
        [
            "heartsounds_data/normal/normals-20241116/006-19 sent to Joel and Carson/6.29.24 pm 24 wks ABNORMAL.mp3"
        ],
        legacy_parquet_labels=["NORMAL"],
    )
    assert res.binary_label == BinaryLabel.ABNORMAL
    assert res.subtype == Subtype.ABNORMAL_GENERIC
    assert res.reason == LabelRepairReason.CASE_BUG_REPAIRED_TO_ABNORMAL


def test_typo_abnomral_and_subtype_inference():
    res_typo = resolve_recording_label(
        ["heartsounds_data/normal/normals-20241116/006-19/7.2 24 aft 24 wks ABNOMRAL.mp3"],
        legacy_parquet_labels=["NORMAL"],
    )
    assert res_typo.binary_label == BinaryLabel.ABNORMAL

    res_pac = resolve_recording_label(
        ["gs://data-raw-dualregion/abnormal/allpacs/PAC_sample_1.m4a"],
        legacy_parquet_labels=["PREMATURE_ATRIAL_CONTRACTION"],
    )
    assert res_pac.binary_label == BinaryLabel.ABNORMAL
    assert res_pac.subtype == Subtype.PREMATURE_ATRIAL_CONTRACTION
    assert res_pac.reason == LabelRepairReason.UNCHANGED

    res_avb = resolve_recording_label(
        ["gs://data-raw-dualregion/abnormal/allavbfromsenttojoel5_24/case2.m4a"],
        legacy_parquet_labels=["ATRIOVENTRICULAR_BLOCK"],
    )
    assert res_avb.subtype == Subtype.ATRIOVENTRICULAR_BLOCK


def test_explicit_normal_in_abnormal_dir_repaired_to_normal():
    res = resolve_recording_label(
        ["heartsounds_data/abnormal/Normal Untitled video - Made with Clipchamp (2).mp4"],
        legacy_parquet_labels=["ABNORMAL_GENERIC"],
    )
    assert res.binary_label == BinaryLabel.NORMAL
    assert res.subtype == Subtype.NORMAL
    assert res.reason == LabelRepairReason.EXPLICIT_NORMAL_IN_ABNORMAL_DIR


def test_non_diagnostic_and_cross_folder_conflict_quarantined():
    res_nd = resolve_recording_label(
        ["heartsounds_data/abnormal/NON-DIAGNOSTIC Untitled video.mp4"],
        legacy_parquet_labels=["ABNORMAL_GENERIC"],
    )
    assert res_nd.binary_label == BinaryLabel.QUARANTINED
    assert res_nd.subtype == Subtype.NON_DIAGNOSTIC
    assert res_nd.reason == LabelRepairReason.QUARANTINED_NON_DIAGNOSTIC

    res_conf = resolve_recording_label(
        [
            "heartsounds_data/normal/normals-20241116/20-12/11.14.22 pm.caf",
            "heartsounds_data/abnormal/allpacsfromfoldersenttojoel5_24/PACs.caf",
        ],
        legacy_parquet_labels=["NORMAL", "PREMATURE_ATRIAL_CONTRACTION"],
    )
    assert res_conf.binary_label == BinaryLabel.QUARANTINED
    assert res_conf.subtype == Subtype.CONFLICT
    assert res_conf.reason == LabelRepairReason.QUARANTINED_CROSS_FOLDER_CONFLICT


def test_patient_group_extraction_and_zero_split_leakage():
    g1 = extract_patient_group_single(
        "heartsounds_data/normal/normals-20241116/14-22 480/22-32/07-02-24_10_30pm PT_23w0d ABNORMAL.m4a"
    )
    g2 = extract_patient_group_single(
        "heartsounds_data/abnormal/06-29-24_2_34pm PT_22w4d ABNORMAL.m4a"
    )
    assert g1 == "subj:14-22 480/22-32"
    assert g2 == "subj:14-22 480/22-32"
    assert (
        resolve_patient_group(
            [
                "heartsounds_data/abnormal/7.2.24 pm 24 wks ABNORMAL.mp3",
                "heartsounds_data/normal/normals-20241116/006-19 sent to Joel and Carson/7.2.24 pm 24 wks ABNORMAL.mp3",
            ]
        )
        == "subj:006-19 sent to Joel and Carson"
    )

    rows = []
    for subj_idx in range(35):
        is_abn = int(subj_idx < 7)
        for clip_idx in range(4):
            rows.append(
                {
                    "recording_id": f"r_{subj_idx}_{clip_idx}",
                    "curated_label": is_abn,
                    "curated_subtype": "PREMATURE_ATRIAL_CONTRACTION" if is_abn else "NORMAL",
                    "patient_group": f"subj:p{subj_idx:02d}",
                }
            )
    rows.append(
        {
            "recording_id": "r_quar",
            "curated_label": int(BinaryLabel.QUARANTINED),
            "curated_subtype": "NON_DIAGNOSTIC",
            "patient_group": "subj:quar",
        }
    )
    df = pd.DataFrame(rows)
    splits = assign_patient_grouped_splits(df, n_folds=5, test_Fold_modulus=7, seed=42)
    df["split"] = splits

    assert df.loc[df["recording_id"] == "r_quar", "split"].iloc[0] == "quarantine"
    active = df[df["split"] != "quarantine"]
    assert set(active["split"].unique()) == {
        "test",
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    }
    assert (active.groupby("patient_group")["split"].nunique() == 1).all()
