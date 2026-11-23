"""Evaluation 2: Clinical 3-state triage (`GREEN_NORMAL` / `YELLOW_RERECORD` / `RED_ABNORMAL`).

Implements the clinical workflow where borderline-margin or poor-periodicity recordings
receive `YELLOW_RERECORD` ("Please repeat 20s recording"), while confident recordings
receive `GREEN_NORMAL` or `RED_ABNORMAL`. All Yellow-band boundaries are calibrated
exclusively on Out-of-Fold (OOF) validation data and evaluated on held-out `test`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import confusion_matrix, roc_auc_score

from heartsounds.schema import TriageState


def compute_uncertainty_score(
    probs: np.ndarray,
    center_threshold: float,
    quality_r_peak: np.ndarray | None = None,
    *,
    quality_weight: float = 0.25,
) -> np.ndarray:
    """Compute a per-recording ambiguity/re-record score in `[0, 1]` (higher = more uncertain).

    Combines log-odds proximity to the decision boundary `center_threshold` with optional
    signal periodicity degradation `(1 - quality_r_peak)` so noisy recordings near the
    decision margin are prioritized for `YELLOW_RERECORD`.
    """
    p_clip = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    c_clip = float(np.clip(center_threshold, 1e-5, 1.0 - 1e-5))
    logit_margin = np.abs(logit(p_clip) - logit(c_clip))
    margin_uncertainty = np.exp(-logit_margin / 1.25)

    if quality_r_peak is None:
        return margin_uncertainty

    q_clip = np.clip(quality_r_peak.astype(np.float64), 0.0, 1.0)
    noise_penalty = 1.0 - q_clip
    # Do not suppress high-confidence abnormal alarms (> c_clip + 1.5 logits) just for low R_peak,
    # since chaotic arrhythmias can naturally lower autocorrelation peak height!
    strong_abnormal = (logit(p_clip) - logit(c_clip)) > 1.5
    combined = (1.0 - quality_weight) * margin_uncertainty + quality_weight * (
        margin_uncertainty * 0.5 + noise_penalty * 0.5
    )
    return np.where(strong_abnormal, margin_uncertainty, combined)


def assign_triage_states(
    probs: np.ndarray,
    center_threshold: float,
    uncertainty_scores: np.ndarray,
    yellow_cutoff: float,
) -> np.ndarray:
    """Assign `GREEN_NORMAL`, `YELLOW_RERECORD`, or `RED_ABNORMAL` per recording."""
    states = np.where(
        probs >= center_threshold,
        TriageState.RED_ABNORMAL.value,
        TriageState.GREEN_NORMAL.value,
    )
    yellow_mask = uncertainty_scores > yellow_cutoff
    return np.where(yellow_mask, TriageState.YELLOW_RERECORD.value, states)


def _metrics_on_triage_calls(
    y_true: np.ndarray,
    probs: np.ndarray,
    states: np.ndarray,
) -> dict[str, Any]:
    definitive_mask = states != TriageState.YELLOW_RERECORD.value
    n_total = len(y_true)
    n_def = int(np.sum(definitive_mask))
    n_yellow = n_total - n_def

    if n_def == 0:
        return {
            "n_total": n_total,
            "n_definitive": 0,
            "n_yellow": n_yellow,
            "coverage": 0.0,
            "yellow_rate": 1.0,
        }

    yd = y_true[definitive_mask]
    pd_def = probs[definitive_mask]
    pred_def = (states[definitive_mask] == TriageState.RED_ABNORMAL.value).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(yd, pred_def, labels=[0, 1]).ravel()

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))
    ppv = float(tp / max(1, tp + fp))
    f1 = float(2 * tp / max(1, 2 * tp + fp + fn))
    roc = float(roc_auc_score(yd, pd_def)) if len(np.unique(yd)) > 1 else 0.0

    # Also track how many true abnormals landed in RED vs YELLOW vs GREEN across 100% of cohort
    abn_mask = y_true == 1
    n_abn_total = max(1, int(np.sum(abn_mask)))
    abn_red = int(np.sum(states[abn_mask] == TriageState.RED_ABNORMAL.value))
    abn_yellow = int(np.sum(states[abn_mask] == TriageState.YELLOW_RERECORD.value))
    abn_green_missed = int(np.sum(states[abn_mask] == TriageState.GREEN_NORMAL.value))

    return {
        "n_total": int(n_total),
        "n_definitive": n_def,
        "n_yellow": int(n_yellow),
        "coverage": round(float(n_def / max(1, n_total)), 5),
        "yellow_rate": round(float(n_yellow / max(1, n_total)), 5),
        "definitive_sensitivity": round(sens, 5),
        "definitive_specificity": round(spec, 5),
        "definitive_balanced_accuracy": round(0.5 * (sens + spec), 5),
        "definitive_youden_j": round(sens + spec - 1.0, 5),
        "definitive_precision": round(ppv, 5),
        "definitive_f1": round(f1, 5),
        "definitive_roc_auc": round(roc, 5),
        "definitive_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "all_abnormal_disposition": {
            "total_abnormal": int(np.sum(abn_mask)),
            "red_detected": abn_red,
            "yellow_rerecord": abn_yellow,
            "green_false_reassurance": abn_green_missed,
            "red_or_yellow_actionable_rate": round(float((abn_red + abn_yellow) / n_abn_total), 5),
        },
    }


def evaluate_triage_suite(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    center_threshold: float,
    oof_quality: np.ndarray | None = None,
    test_quality: np.ndarray | None = None,
    quarantine_df: pd.DataFrame | None = None,
    quarantine_quality: np.ndarray | None = None,
    yellow_budgets: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run Evaluation 2 across OOF-calibrated Yellow re-record budgets (`0%..20%`).

    Returns:
        (triage_report_dict, oof_states_at_10pct, test_states_at_10pct)
    """
    y_oof = oof_df["y_true"].to_numpy(dtype=np.int32)
    p_oof = oof_df["y_prob"].to_numpy(dtype=np.float64)
    y_tst = test_df["y_true"].to_numpy(dtype=np.int32)
    p_tst = test_df["y_prob"].to_numpy(dtype=np.float64)

    u_oof = compute_uncertainty_score(p_oof, center_threshold, oof_quality)
    u_tst = compute_uncertainty_score(p_tst, center_threshold, test_quality)

    budget_curves: dict[str, Any] = {}
    default_oof_states = assign_triage_states(p_oof, center_threshold, u_oof, np.inf)
    default_tst_states = assign_triage_states(p_tst, center_threshold, u_tst, np.inf)

    for b in yellow_budgets:
        if b <= 0.0:
            cutoff = float("inf")
        else:
            cutoff = float(np.quantile(u_oof, 1.0 - b))

        st_oof = assign_triage_states(p_oof, center_threshold, u_oof, cutoff)
        st_tst = assign_triage_states(p_tst, center_threshold, u_tst, cutoff)
        if abs(b - 0.10) < 1e-6:
            default_oof_states = st_oof
            default_tst_states = st_tst

        key = f"yellow_budget_{int(round(b * 100)):02d}pct"
        budget_curves[key] = {
            "target_yellow_budget": b,
            "oof_uncertainty_cutoff": None if not np.isfinite(cutoff) else round(cutoff, 6),
            "oof": _metrics_on_triage_calls(y_oof, p_oof, st_oof),
            "test": _metrics_on_triage_calls(y_tst, p_tst, st_tst),
        }

    quarantine_inspection: list[dict[str, Any]] = []
    if quarantine_df is not None and len(quarantine_df) > 0:
        cutoff_10 = float(np.quantile(u_oof, 0.90))
        pq_arr = quarantine_df["y_prob"].to_numpy(dtype=np.float64)
        uq_arr = compute_uncertainty_score(pq_arr, center_threshold, quarantine_quality)
        sq_arr = assign_triage_states(pq_arr, center_threshold, uq_arr, cutoff_10)
        for i, row in enumerate(quarantine_df.itertuples(index=False)):
            quarantine_inspection.append(
                {
                    "recording_id": str(row.recording_id),
                    "curated_subtype": str(row.curated_subtype),
                    "y_prob": round(float(pq_arr[i]), 5),
                    "uncertainty_score": round(float(uq_arr[i]), 5),
                    "triage_at_10pct_budget": str(sq_arr[i]),
                }
            )

    report = {
        "center_threshold": round(float(center_threshold), 6),
        "primary_reference_budget": "yellow_budget_10pct",
        "budget_curves": budget_curves,
        "quarantine_inspection": quarantine_inspection,
    }
    return report, default_oof_states, default_tst_states
