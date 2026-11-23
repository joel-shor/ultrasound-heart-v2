"""End-to-end curation of the 4,000 Hz HeartSounds dataset with full GCP provenance.

Reads legacy 4 kHz waveform shards (`dataframes/4000sr_lenNones/data-*.pkl`) and
legacy 1.6 kHz Parquet metadata, repairs labels, deduplicates exact waveforms,
assigns leak-free patient-grouped splits, and writes self-documenting tables ready
for direct upload to `gs://data-processed-dualregion/v2_curated/` and Beaker.
"""

from __future__ import annotations

import collections
import enum
import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from heartsounds.data.grouping import assign_patient_grouped_splits, resolve_patient_group
from heartsounds.data.labels import resolve_recording_label, to_canonical_gcs_uri
from heartsounds.schema import (
    MANIFEST_COLUMNS,
    WAVEFORM_TABLE_COLUMNS,
    BinaryLabel,
    validate_manifest_df,
    validate_waveform_df,
)
from heartsounds.storage.gcs import CURATED_GCS_ROOT, sha256_file


def _install_legacy_pickle_shim() -> None:
    """Register minimal `input_reader` enums so legacy pandas pickles load standalone."""
    if "input_reader" in sys.modules:
        return
    shim = types.ModuleType("input_reader")

    class StorageType(enum.Enum):
        LOCAL = "local"
        DRIVE = "drive"
        GCP = "gcp"

    class Label(enum.Enum):
        NORMAL = 1
        ABNORMAL_GENERIC = 2
        PREMATURE_ATRIAL_CONTRACTION = 3
        ATRIOVENTRICULAR_BLOCK = 4
        TOO_FAST = 5
        TOO_SLOW = 6
        HICCUP = 7

    shim.StorageType = StorageType  # type: ignore[attr-defined]
    shim.Label = Label  # type: ignore[attr-defined]
    sys.modules["input_reader"] = shim


def waveform_md5(samples: np.ndarray) -> str:
    """Compute a deterministic MD5 hash over float32 waveform samples."""
    arr = np.ascontiguousarray(samples, dtype=np.float32)
    return hashlib.md5(arr.tobytes()).hexdigest()


def _label_name(raw_label: Any) -> str:
    if hasattr(raw_label, "name"):
        return str(raw_label.name)
    text = str(raw_label)
    return text.split(".")[-1] if text.startswith("Label.") else text


@dataclass(frozen=True)
class RawRecord:
    """Minimal representation of a single un-deduplicated input row."""

    row_index: int
    filename: str
    legacy_label: str
    sample_rate: int
    samples: np.ndarray


def load_4khz_pickle_shards(shards_dir: str | Path) -> list[RawRecord]:
    """Load all `data-*.pkl` 4,000 Hz waveform shards from `shards_dir`."""
    _install_legacy_pickle_shim()
    folder = Path(shards_dir)
    pkl_files = sorted(
        folder.glob("data-*.pkl"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not pkl_files:
        raise FileNotFoundError(f"No data-*.pkl shards found in {folder}")

    records: list[RawRecord] = []
    idx = 0
    for pkl_path in pkl_files:
        df = pd.read_pickle(pkl_path)
        for row in df.itertuples(index=False):
            arr = np.asarray(row.samples, dtype=np.float32)
            records.append(
                RawRecord(
                    row_index=idx,
                    filename=str(row.filename),
                    legacy_label=_label_name(row.label),
                    sample_rate=int(row.sample_rate),
                    samples=arr,
                )
            )
            idx += 1
    return records


def curate_records(
    records: list[RawRecord],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Deduplicate, repair labels, assign patient-grouped splits, and build provenance tables.

    Returns:
        (manifest_df, raw_row_mapping_df, waveforms_by_recording_id)
    """
    grouped: dict[str, list[RawRecord]] = collections.defaultdict(list)
    row_hashes: list[tuple[int, str, str, str]] = []

    for rec in records:
        if rec.samples.size == 0:
            continue
        md5 = waveform_md5(rec.samples)
        grouped[md5].append(rec)
        row_hashes.append((rec.row_index, rec.filename, rec.legacy_label, md5))

    manifest_rows: list[dict[str, Any]] = []
    waveforms_by_id: dict[str, np.ndarray] = {}

    for md5 in sorted(grouped.keys()):
        grp = grouped[md5]
        rec_id = f"rec_{md5[:12]}"
        uris = sorted({to_canonical_gcs_uri(r.filename) for r in grp})
        legacy_labels = sorted({r.legacy_label for r in grp})
        resolved = resolve_recording_label(uris, legacy_labels)
        patient_grp = resolve_patient_group(uris)

        rep = grp[0]
        sr = int(rep.sample_rate)
        dur_s = round(float(rep.samples.size) / float(sr), 4)
        waveforms_by_id[rec_id] = rep.samples

        # Prefer a `normal/normals-20241116/...` or `abnormal/...` primary URI deterministically
        primary_uri = uris[0]
        manifest_rows.append(
            {
                "recording_id": rec_id,
                "waveform_md5": md5,
                "primary_raw_gcs_uri": primary_uri,
                "all_duplicate_raw_gcs_uris": json.dumps(uris),
                "n_duplicate_copies": len(grp),
                "original_parquet_labels": json.dumps(legacy_labels),
                "curated_label": int(resolved.binary_label),
                "curated_subtype": resolved.subtype.value,
                "label_change_reason": resolved.reason.value,
                "patient_group": patient_grp,
                "duration_s": dur_s,
                "sample_rate": sr,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["split"] = assign_patient_grouped_splits(manifest_df, n_folds=n_folds, seed=seed)
    manifest_df = manifest_df.loc[:, list(MANIFEST_COLUMNS)].reset_index(drop=True)
    validate_manifest_df(manifest_df)

    md5_to_info = manifest_df.set_index("waveform_md5")[
        ["recording_id", "curated_label", "curated_subtype", "split"]
    ].to_dict(orient="index")

    mapping_rows = []
    for row_idx, fn, orig_lbl, md5 in row_hashes:
        info = md5_to_info[md5]
        mapping_rows.append(
            {
                "legacy_row_index": row_idx,
                "raw_gcs_uri": to_canonical_gcs_uri(fn),
                "legacy_label": orig_lbl,
                "waveform_md5": md5,
                "recording_id": info["recording_id"],
                "curated_label": info["curated_label"],
                "curated_subtype": info["curated_subtype"],
                "split": info["split"],
            }
        )
    raw_mapping_df = pd.DataFrame(mapping_rows)
    return manifest_df, raw_mapping_df, waveforms_by_id


def _write_waveform_parquet(
    subset_manifest: pd.DataFrame,
    waveforms_by_id: dict[str, np.ndarray],
    out_path: Path,
    *,
    allow_quarantined: bool = False,
) -> None:
    df = subset_manifest[
        [
            "recording_id",
            "waveform_md5",
            "primary_raw_gcs_uri",
            "curated_label",
            "curated_subtype",
            "patient_group",
            "duration_s",
            "sample_rate",
            "split",
        ]
    ].copy()
    # Store float32 list arrays with ZSTD compression (~50% smaller than uncompressed pickle, lossless vs source float32)
    df["samples"] = [
        np.asarray(waveforms_by_id[rid], dtype=np.float32) for rid in df["recording_id"]
    ]
    df = df.loc[:, list(WAVEFORM_TABLE_COLUMNS)]
    validate_waveform_df(df, allow_quarantined=allow_quarantined)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")


def write_curated_bundle(
    manifest_df: pd.DataFrame,
    raw_mapping_df: pd.DataFrame,
    waveforms_by_id: dict[str, np.ndarray],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write all inspectable CSV, Parquet, JSON, and Markdown provenance files to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_csv = out / "manifest.csv"
    manifest_pq = out / "manifest.parquet"
    mapping_csv = out / "raw_row_mapping.csv"
    train_val_pq = out / "train_val_folds.parquet"
    test_pq = out / "test.parquet"
    quarantine_pq = out / "quarantine.parquet"

    manifest_df.to_csv(manifest_csv, index=False)
    manifest_df.to_parquet(manifest_pq, index=False)
    raw_mapping_df.to_csv(mapping_csv, index=False)

    dev_df = manifest_df[manifest_df["split"].str.startswith("fold_")].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    quar_df = manifest_df[manifest_df["split"] == "quarantine"].reset_index(drop=True)

    _write_waveform_parquet(dev_df, waveforms_by_id, train_val_pq, allow_quarantined=False)
    _write_waveform_parquet(test_df, waveforms_by_id, test_pq, allow_quarantined=False)
    _write_waveform_parquet(quar_df, waveforms_by_id, quarantine_pq, allow_quarantined=True)

    checksums = {
        p.name: sha256_file(p)
        for p in (manifest_csv, manifest_pq, mapping_csv, train_val_pq, test_pq, quarantine_pq)
    }

    split_summary: dict[str, Any] = {}
    for split_name, grp in manifest_df.groupby("split"):
        split_summary[str(split_name)] = {
            "total": int(len(grp)),
            "normal": int((grp["curated_label"] == int(BinaryLabel.NORMAL)).sum()),
            "abnormal": int((grp["curated_label"] == int(BinaryLabel.ABNORMAL)).sum()),
            "quarantined": int((grp["curated_label"] == int(BinaryLabel.QUARANTINED)).sum()),
            "unique_patient_groups": int(grp["patient_group"].nunique()),
            "subtypes": {k: int(v) for k, v in grp["curated_subtype"].value_counts().items()},
        }

    report = {
        "gcs_destination": CURATED_GCS_ROOT,
        "raw_input_rows": int(len(raw_mapping_df)),
        "unique_waveforms": int(len(manifest_df)),
        "clean_usable_recordings": int(len(dev_df) + len(test_df)),
        "quarantined_recordings": int(len(quar_df)),
        "label_change_reasons": {
            k: int(v) for k, v in manifest_df["label_change_reason"].value_counts().items()
        },
        "splits": split_summary,
        "sha256": checksums,
    }
    (out / "curation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    readme_lines = [
        "# HeartSounds v2 Curated Dataset (`v2_curated`)",
        "",
        "Self-documenting, deduplicated, label-repaired 4,000 Hz fetal Doppler dataset",
        "built from `gs://data-processed-dualregion/dataframes/4000sr_lenNones/` and",
        "`gs://data-raw-dualregion/`.",
        "",
        "## Summary Counts",
        f"- **Original un-deduplicated rows:** {report['raw_input_rows']:,}",
        f"- **Unique audio waveforms (by MD5):** {report['unique_waveforms']:,}",
        f"- **Clean usable recordings (`fold_0..fold_4` + `test`):** {report['clean_usable_recordings']:,}",
        f"- **Quarantined recordings (`NON_DIAGNOSTIC` + cross-folder conflicts):** {report['quarantined_recordings']:,}",
        "",
        "## Label Audit Breakdown",
        *[f"- `{k}`: **{v:,}**" for k, v in report["label_change_reasons"].items()],
        "",
        "## Files in This Folder",
        "- `manifest.csv` / `manifest.parquet`: Master 1-row-per-recording provenance table.",
        "- `raw_row_mapping.csv`: Maps every original row index to `recording_id` and `split`.",
        "- `train_val_folds.parquet`: 5-fold patient-grouped development set (`fold_0`..`fold_4`, 4,000 Hz float32).",
        "- `test.parquet`: Held-out patient-grouped test set (`test`, 4,000 Hz float32).",
        "- `quarantine.parquet`: Quarantined recordings excluded from binary train/test.",
        "- `dsp_features.parquet`: Extracted multi-band DSP/HRV tabular features for `recording_id`.",
        "- `predictions/`: Out-of-fold (`oof_predictions.csv`) and held-out test (`test_predictions.csv`) predictions.",
        "",
        "## SHA-256 Checksums",
        *[f"- `{fname}`: `{digest}`" for fname, digest in checksums.items()],
        "",
    ]
    (out / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return report
