"""Pure patient/subject folder extraction and leak-free StratifiedGroupKFold splitting."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from heartsounds.data.labels import to_canonical_gcs_uri
from heartsounds.schema import BinaryLabel

NESTED_BATCH_CONTAINERS: frozenset[str] = frozenset(
    {
        "14-22 480",
        "006-19 sent to joel and carson",
        "23-6 abn removed",
        "007-13 sent to joel and carson",
        "23-9",
    }
)

PT_SERIES_RE = re.compile(r"\bpt_\d{2}w\d", re.IGNORECASE)


def extract_patient_group_single(path: str) -> str:
    """Infer a canonical subject/cohort group identifier from a single GCS path."""
    uri = to_canonical_gcs_uri(path)
    _, _, rel = uri[len("gs://") :].partition("/")
    parts = [p.strip() for p in rel.split("/") if p.strip()]
    if not parts:
        return "unknown"

    top = parts[0].lower()
    basename = os.path.basename(rel)

    # Patient-organized folders under `normal/normals-20241116/<batch_or_patient>/...`
    if top == "normal" and len(parts) >= 3 and parts[1].lower() == "normals-20241116":
        lvl2 = parts[2]
        if lvl2.lower() in NESTED_BATCH_CONTAINERS and len(parts) >= 5:
            return f"subj:{lvl2}/{parts[3]}"
        if len(parts) >= 4:
            return f"subj:{lvl2}"

    # Known PT_22wXd / PT_23wXd longitudinal series corresponds to subject 14-22 480/22-32
    if PT_SERIES_RE.search(basename):
        return "subj:14-22 480/22-32"

    # Curated abnormal subfolders (`abnormal/allavb...`, `abnormal/20241120/...`)
    if top == "abnormal":
        if len(parts) >= 3:
            # Group by curated subfolder + coarse filename stem prefix so longitudinal
            # recordings from the same subfolder stay grouped without collapsing all PACs into 1 group.
            stem = os.path.splitext(basename)[0][:10].lower().strip()
            return f"abn:{'/'.join(parts[1:-1])}/{stem}"
        stem = os.path.splitext(basename)[0][:10].lower().strip()
        return f"abn_root:{stem}"

    stem = os.path.splitext(basename)[0][:10].lower().strip()
    return f"norm_root:{stem}"


def resolve_patient_group(paths: Sequence[str]) -> str:
    """Choose the most informative subject group across all duplicate paths of a recording.

    Prefers explicit `subj:...` patient-folder IDs when a recording exists in both a
    patient folder and a flat `abnormal/` copy.
    """
    candidates = [extract_patient_group_single(p) for p in paths]
    for c in candidates:
        if c.startswith("subj:"):
            return c
    return sorted(candidates)[0]


def assign_patient_grouped_splits(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    test_Fold_modulus: int = 7,
    seed: int = 42,
) -> pd.Series:
    """Assign `quarantine`, held-out `test` (~1/7 = 14.3%), and `fold_0`..`fold_4` (~85.7%).

    Guarantees zero `patient_group` overlap across `test` and every CV fold while
    stratifying across clinical subtypes (`NORMAL`, `PAC`, `AVB`, `ABNORMAL_GENERIC`, etc.).
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    splits = np.full(len(df), "quarantine", dtype=object)
    valid_mask = df["curated_label"].to_numpy() != int(BinaryLabel.QUARANTINED)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return pd.Series(splits, index=df.index, name="split")

    sub_df = df.iloc[valid_indices]
    groups = sub_df["patient_group"].to_numpy()
    # Stratify on subtype when frequency >= test_Fold_modulus, else binary label
    subtype_counts = sub_df["curated_subtype"].value_counts()
    strat_target = np.array(
        [
            st if subtype_counts.get(st, 0) >= test_Fold_modulus else str(lbl)
            for st, lbl in zip(
                sub_df["curated_subtype"].to_numpy(),
                sub_df["curated_label"].to_numpy(),
                strict=True,
            )
        ]
    )

    # Step 1: Carve out ~14.3% (1 / 7) patient-grouped held-out test set.
    outer_sgkf = StratifiedGroupKFold(n_splits=test_Fold_modulus, shuffle=True, random_state=seed)
    dev_rel_idx, test_rel_idx = next(outer_sgkf.split(np.zeros(len(sub_df)), strat_target, groups))

    splits[valid_indices[test_rel_idx]] = "test"

    # Step 2: Partition remaining ~85.7% dev recordings into `n_folds` patient-grouped CV folds.
    dev_abs_idx = valid_indices[dev_rel_idx]
    dev_sub = df.iloc[dev_abs_idx]
    dev_groups = dev_sub["patient_group"].to_numpy()
    dev_st_counts = dev_sub["curated_subtype"].value_counts()
    dev_strat = np.array(
        [
            st if dev_st_counts.get(st, 0) >= n_folds else str(lbl)
            for st, lbl in zip(
                dev_sub["curated_subtype"].to_numpy(),
                dev_sub["curated_label"].to_numpy(),
                strict=True,
            )
        ]
    )

    inner_sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
    for fold_id, (_, val_rel) in enumerate(
        inner_sgkf.split(np.zeros(len(dev_abs_idx)), dev_strat, dev_groups)
    ):
        splits[dev_abs_idx[val_rel]] = f"fold_{fold_id}"

    # Verify strict patient-group disjointness across all non-quarantine splits.
    assigned = pd.Series(splits, index=df.index, name="split")
    active = df.loc[valid_mask].assign(_split=assigned.loc[valid_mask])
    group_split_counts = active.groupby("patient_group")["_split"].nunique()
    leaking = group_split_counts[group_split_counts > 1]
    if not leaking.empty:
        raise AssertionError(f"Patient-group leakage detected across splits: {leaking.to_dict()}")

    return assigned
