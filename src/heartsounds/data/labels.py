"""Pure, auditable label and subtype resolution rules for STOP-BLOQ paths.

Fixes the legacy case-sensitive path bug where files under `normal/normals-20241116/`
explicitly named `... ABNORMAL.m4a` (or `ABNOMRAL`, `TOO FAST`) were mislabeled
`NORMAL`, handles explicit `Normal...` recordings placed inside `abnormal/`,
and quarantines `NON_DIAGNOSTIC` recordings and genuine cross-folder conflicts.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from heartsounds.schema import BinaryLabel, LabelRepairReason, Subtype

ABNORMAL_BASENAME_TOKENS: tuple[str, ...] = (
    "abnormal",
    "abnomral",
    "abn ",
    "pac",
    "avb",
    "block",
    "too fast",
    "toofast",
    "too  fast",
    "too slow",
    "tooslow",
    "too  slow",
    "hiccup",
    "svt",
    "brady",
    "tachy",
)


@dataclass(frozen=True)
class ResolvedLabel:
    """Result of resolving all file paths sharing an identical audio waveform."""

    binary_label: BinaryLabel
    subtype: Subtype
    reason: LabelRepairReason


def to_canonical_gcs_uri(legacy_path: str, bucket: str = "data-raw-dualregion") -> str:
    """Convert `heartsounds_data/normal/...` or local path to `gs://data-raw-dualregion/...`."""
    cleaned = legacy_path.strip()
    if cleaned.startswith("gs://"):
        return cleaned
    for prefix in ("heartsounds_data/", "/heartsounds_data/", "./heartsounds_data/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.lstrip("/")
    return f"gs://{bucket}/{cleaned}"


def _relative_bucket_path(path: str) -> str:
    uri = to_canonical_gcs_uri(path)
    _, _, rest = uri[len("gs://") :].partition("/")
    return rest


def _is_non_diagnostic_basename(basename_lower: str) -> bool:
    return "non-diagnostic" in basename_lower or "nondiagnostic" in basename_lower


def _is_explicit_normal_in_abnormal_dir(rel_lower: str, basename_lower: str) -> bool:
    if not rel_lower.startswith("abnormal/"):
        return False
    has_normal_word = "normal" in basename_lower
    has_abnormal_word = "abnormal" in basename_lower or "abnomral" in basename_lower
    return has_normal_word and not has_abnormal_word


def _has_abnormal_indicator(rel_lower: str, basename_lower: str) -> bool:
    if rel_lower.startswith("abnormal/"):
        return not _is_explicit_normal_in_abnormal_dir(rel_lower, basename_lower)
    if "w. abnormals" in rel_lower:
        return True
    return any(tok in basename_lower for tok in ABNORMAL_BASENAME_TOKENS)


def _is_unmarked_normal_folder_copy(rel_lower: str, basename_lower: str) -> bool:
    if not rel_lower.startswith("normal/"):
        return False
    if "w. abnormals" in rel_lower:
        return False
    return not any(tok in basename_lower for tok in ABNORMAL_BASENAME_TOKENS)


def infer_abnormal_subtype(paths: Sequence[str]) -> Subtype:
    """Infer the most specific abnormal clinical subtype across duplicate file paths."""
    joined_lower = " | ".join(_relative_bucket_path(p).lower() for p in paths)
    # Caution: folder `abnormaltooslowfirsttwoandtoofastthirdone` contains both words;
    # inspect basenames first before falling back to full path keywords.
    basenames_lower = " | ".join(os.path.basename(p).lower() for p in paths)

    for text in (basenames_lower, joined_lower):
        if "avb" in text or "heart block" in text:
            return Subtype.ATRIOVENTRICULAR_BLOCK
        if "pac" in text:
            return Subtype.PREMATURE_ATRIAL_CONTRACTION
        if "hiccup" in text:
            return Subtype.HICCUP
        if ("too slow" in text or "too  slow" in text or "brady" in text) and "toofast" not in text:
            return Subtype.TOO_SLOW
        if (
            "too fast" in text or "too  fast" in text or "tachy" in text or "svt" in text
        ) and "tooslow" not in text:
            return Subtype.TOO_FAST

    return Subtype.ABNORMAL_GENERIC


def resolve_recording_label(
    paths: Sequence[str],
    legacy_parquet_labels: Sequence[str] = (),
) -> ResolvedLabel:
    """Deterministically resolve binary label, subtype, and audit reason for a unique waveform."""
    if not paths:
        raise ValueError("resolve_recording_label requires at least one path")

    rels_lower = [_relative_bucket_path(p).lower() for p in paths]
    bases_lower = [os.path.basename(p).lower() for p in paths]

    # 1. Explicit non-diagnostic clips -> Quarantine (for Evaluation 2 inspection).
    if any(_is_non_diagnostic_basename(b) for b in bases_lower):
        return ResolvedLabel(
            binary_label=BinaryLabel.QUARANTINED,
            subtype=Subtype.NON_DIAGNOSTIC,
            reason=LabelRepairReason.QUARANTINED_NON_DIAGNOSTIC,
        )

    # 2. Recordings inside `abnormal/` explicitly titled `Normal...` / `Repeat was normal`.
    if all(
        _is_explicit_normal_in_abnormal_dir(r, b)
        for r, b in zip(rels_lower, bases_lower, strict=True)
    ):
        return ResolvedLabel(
            binary_label=BinaryLabel.NORMAL,
            subtype=Subtype.NORMAL,
            reason=LabelRepairReason.EXPLICIT_NORMAL_IN_ABNORMAL_DIR,
        )

    any_abnormal = any(
        _has_abnormal_indicator(r, b) for r, b in zip(rels_lower, bases_lower, strict=True)
    )
    any_unmarked_normal = any(
        _is_unmarked_normal_folder_copy(r, b) for r, b in zip(rels_lower, bases_lower, strict=True)
    )

    # 3. Genuine cross-folder conflict: identical audio sits both in `abnormal/` and in a
    #    `normal/` folder WITHOUT an abnormal filename tag. Quarantine rather than guess.
    if any_abnormal and any_unmarked_normal:
        return ResolvedLabel(
            binary_label=BinaryLabel.QUARANTINED,
            subtype=Subtype.CONFLICT,
            reason=LabelRepairReason.QUARANTINED_CROSS_FOLDER_CONFLICT,
        )

    # 4. Clean Abnormal recording.
    if any_abnormal:
        subtype = infer_abnormal_subtype(paths)
        legacy_set = {str(lbl).upper() for lbl in legacy_parquet_labels}
        repaired = legacy_set == {"NORMAL"} or ("NORMAL" in legacy_set and len(legacy_set) > 1)
        return ResolvedLabel(
            binary_label=BinaryLabel.ABNORMAL,
            subtype=subtype,
            reason=(
                LabelRepairReason.CASE_BUG_REPAIRED_TO_ABNORMAL
                if repaired
                else LabelRepairReason.UNCHANGED
            ),
        )

    # 5. Clean Normal recording.
    return ResolvedLabel(
        binary_label=BinaryLabel.NORMAL,
        subtype=Subtype.NORMAL,
        reason=LabelRepairReason.UNCHANGED,
    )
