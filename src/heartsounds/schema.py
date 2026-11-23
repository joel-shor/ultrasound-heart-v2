"""Canonical enums and DataFrame column contracts for HeartSounds v2.

Every Parquet and CSV table exchanged across curation, feature extraction,
training, and evaluation is validated against these schemas so column names
and label semantics are unambiguous and auditable.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

import pandas as pd


class BinaryLabel(IntEnum):
    """Binary ground-truth target used by all classifiers."""

    QUARANTINED = -1
    NORMAL = 0
    ABNORMAL = 1


class Subtype(StrEnum):
    """Fine-grained clinical rhythm category inferred from STOP-BLOQ paths."""

    NORMAL = "NORMAL"
    ABNORMAL_GENERIC = "ABNORMAL_GENERIC"
    PREMATURE_ATRIAL_CONTRACTION = "PREMATURE_ATRIAL_CONTRACTION"
    ATRIOVENTRICULAR_BLOCK = "ATRIOVENTRICULAR_BLOCK"
    TOO_FAST = "TOO_FAST"
    TOO_SLOW = "TOO_SLOW"
    HICCUP = "HICCUP"
    NON_DIAGNOSTIC = "NON_DIAGNOSTIC"
    CONFLICT = "CONFLICT"


ABNORMAL_SUBTYPES: tuple[Subtype, ...] = (
    Subtype.ABNORMAL_GENERIC,
    Subtype.PREMATURE_ATRIAL_CONTRACTION,
    Subtype.ATRIOVENTRICULAR_BLOCK,
    Subtype.TOO_FAST,
    Subtype.TOO_SLOW,
    Subtype.HICCUP,
)

SUBTYPE_TO_INDEX: dict[str, int] = {
    s.value: idx for idx, s in enumerate((Subtype.NORMAL, *ABNORMAL_SUBTYPES))
}


class LabelRepairReason(StrEnum):
    """Audit trail explaining why a recording's curated label differs (or not) from legacy Parquet."""

    UNCHANGED = "unchanged"
    CASE_BUG_REPAIRED_TO_ABNORMAL = "case_bug_repaired_to_abnormal"
    EXPLICIT_NORMAL_IN_ABNORMAL_DIR = "explicit_normal_in_abnormal_dir"
    QUARANTINED_NON_DIAGNOSTIC = "quarantined_nondiagnostic"
    QUARANTINED_CROSS_FOLDER_CONFLICT = "quarantined_conflict"


class TriageState(StrEnum):
    """Clinical 3-state output for Evaluation 2 (HeartSounds app workflow)."""

    GREEN_NORMAL = "GREEN_NORMAL"
    YELLOW_RERECORD = "YELLOW_RERECORD"
    RED_ABNORMAL = "RED_ABNORMAL"


MANIFEST_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "waveform_md5",
    "primary_raw_gcs_uri",
    "all_duplicate_raw_gcs_uris",
    "n_duplicate_copies",
    "original_parquet_labels",
    "curated_label",
    "curated_subtype",
    "label_change_reason",
    "patient_group",
    "duration_s",
    "sample_rate",
    "split",
)

WAVEFORM_TABLE_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "waveform_md5",
    "primary_raw_gcs_uri",
    "curated_label",
    "curated_subtype",
    "patient_group",
    "duration_s",
    "sample_rate",
    "split",
    "samples",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "split",
    "patient_group",
    "y_true",
    "curated_subtype",
    "y_prob",
)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns {missing}; got {list(df.columns)}")


def validate_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a manifest DataFrame conforms to the canonical provenance contract."""
    _require_columns(df, MANIFEST_COLUMNS, "ManifestTable")
    if df["recording_id"].duplicated().any():
        raise ValueError("ManifestTable contains duplicate recording_id values")
    if df["waveform_md5"].duplicated().any():
        raise ValueError("ManifestTable contains duplicate waveform_md5 values")
    valid_labels = {int(x) for x in BinaryLabel}
    bad = set(df["curated_label"].unique()) - valid_labels
    if bad:
        raise ValueError(f"ManifestTable contains invalid curated_label values: {bad}")
    return df


def validate_waveform_df(df: pd.DataFrame, *, allow_quarantined: bool = False) -> pd.DataFrame:
    """Validate a curated waveform split table (`train_val_folds`, `test`, or `quarantine`)."""
    _require_columns(df, WAVEFORM_TABLE_COLUMNS, "WaveformTable")
    if df["recording_id"].duplicated().any():
        raise ValueError("WaveformTable contains duplicate recording_id values")
    if not allow_quarantined and (df["curated_label"] == int(BinaryLabel.QUARANTINED)).any():
        raise ValueError("Quarantined rows must not appear in train/val/test waveform tables")
    return df


def validate_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate standardized model out-of-fold or test predictions."""
    _require_columns(df, PREDICTION_COLUMNS, "PredictionTable")
    if df["recording_id"].duplicated().any():
        raise ValueError("PredictionTable contains duplicate recording_id values")
    if df["y_prob"].isna().any():
        raise ValueError("PredictionTable contains NaN probabilities")
    if ((df["y_prob"] < 0.0) | (df["y_prob"] > 1.0)).any():
        raise ValueError("PredictionTable probabilities must lie in [0.0, 1.0]")
    return df
