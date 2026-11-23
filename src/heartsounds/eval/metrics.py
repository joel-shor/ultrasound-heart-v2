"""Evaluation 1: Pure 100%-coverage binary metrics and patient-cluster bootstrap CIs.

Selects operating thresholds strictly on Out-of-Fold (OOF) validation predictions
under criteria that jointly maximize sensitivity and specificity (Max Youden's J /
Balanced Accuracy, Equal Sensitivity=Specificity, and Max F1), then evaluates once
on the held-out patient-grouped test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def select_oof_thresholds(y_oof: np.ndarray, p_oof: np.ndarray) -> dict[str, float]:
    """Select optimal probability cutoffs on OOF validation data (never peeking at test)."""
    fpr, tpr, roc_thresh = roc_curve(y_oof, p_oof)
    specificity = 1.0 - fpr
    youden_j = tpr + specificity - 1.0

    # Ignore degenerate +/-inf endpoint from sklearn roc_curve
    finite = np.isfinite(roc_thresh)
    tpr_f, spec_f, j_f, th_f = (
        tpr[finite],
        specificity[finite],
        youden_j[finite],
        roc_thresh[finite],
    )

    idx_j = int(np.argmax(j_f))
    # Equal Sensitivity = Specificity operating point (EER complement)
    eer_gap = np.abs(tpr_f - spec_f) - 0.01 * j_f
    idx_eer = int(np.argmin(eer_gap))

    prec, rec, pr_thresh = precision_recall_curve(y_oof, p_oof)
    if pr_thresh.size > 0:
        f1_vals = 2.0 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        th_f1 = float(pr_thresh[int(np.argmax(f1_vals))])
    else:
        th_f1 = 0.5

    return {
        "max_youden_j": float(th_f[idx_j]),
        "equal_sens_spec": float(th_f[idx_eer]),
        "max_f1": th_f1,
    }


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    subtypes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute confusion matrix, sensitivity, specificity, Youden's J, F1, and subtype recall."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))
    ppv = float(tp / max(1, tp + fp))
    npv = float(tn / max(1, tn + fn))
    f1 = float(2 * tp / max(1, 2 * tp + fp + fn))
    acc = float((tp + tn) / max(1, tp + tn + fp + fn))
    bal_acc = 0.5 * (sens + spec)
    youden_j = sens + spec - 1.0

    out: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
        "sensitivity": round(sens, 5),
        "specificity": round(spec, 5),
        "balanced_accuracy": round(bal_acc, 5),
        "youden_j": round(youden_j, 5),
        "precision": round(ppv, 5),
        "npv": round(npv, 5),
        "f1": round(f1, 5),
        "accuracy": round(acc, 5),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    if subtypes is not None:
        subtype_recall: dict[str, dict[str, float | int]] = {}
        for st in sorted(np.unique(subtypes)):
            if st == "NORMAL":
                continue
            mask = subtypes == st
            n_st = int(np.sum(mask))
            if n_st > 0:
                hits = int(np.sum(y_pred[mask] == 1))
                subtype_recall[str(st)] = {
                    "n": n_st,
                    "detected": hits,
                    "recall": round(float(hits / n_st), 5),
                }
        out["subtype_recall"] = subtype_recall
    return out


def cluster_bootstrap_cis(
    pred_df: pd.DataFrame,
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 95% confidence intervals via patient-group cluster bootstrap."""
    rng = np.random.default_rng(seed)
    groups_map: dict[str, np.ndarray] = {}
    for g_name, grp in pred_df.groupby("patient_group", sort=False):
        groups_map[str(g_name)] = grp.index.to_numpy()
    group_keys = list(groups_map.keys())
    n_groups = len(group_keys)

    y_all = pred_df["y_true"].to_numpy(dtype=np.int32)
    p_all = pred_df["y_prob"].to_numpy(dtype=np.float64)

    rocs, prs, sens_l, spec_l, j_l, f1_l = [], [], [], [], [], []
    for _ in range(n_boot):
        sampled_keys = [group_keys[i] for i in rng.integers(0, n_groups, size=n_groups)]
        idx = np.concatenate([groups_map[k] for k in sampled_keys])
        yb = y_all[idx]
        pb = p_all[idx]
        if len(np.unique(yb)) < 2:
            continue
        rocs.append(float(roc_auc_score(yb, pb)))
        prs.append(float(average_precision_score(yb, pb)))
        yb_hat = (pb >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(yb, yb_hat, labels=[0, 1]).ravel()
        s_val = float(tp / max(1, tp + fn))
        c_val = float(tn / max(1, tn + fp))
        sens_l.append(s_val)
        spec_l.append(c_val)
        j_l.append(s_val + c_val - 1.0)
        f1_l.append(float(2 * tp / max(1, 2 * tp + fp + fn)))

    def _ci(arr: list[float]) -> dict[str, float]:
        if not arr:
            return {"p2_5": 0.0, "p97_5": 0.0}
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"p2_5": round(float(lo), 5), "p97_5": round(float(hi), 5)}

    return {
        "roc_auc": _ci(rocs),
        "pr_auc": _ci(prs),
        "sensitivity": _ci(sens_l),
        "specificity": _ci(spec_l),
        "youden_j": _ci(j_l),
        "f1": _ci(f1_l),
    }


def evaluate_binary_suite(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Run complete Evaluation 1 (100%-coverage binary) using OOF-selected thresholds."""
    y_oof = oof_df["y_true"].to_numpy(dtype=np.int32)
    p_oof = oof_df["y_prob"].to_numpy(dtype=np.float64)
    y_tst = test_df["y_true"].to_numpy(dtype=np.int32)
    p_tst = test_df["y_prob"].to_numpy(dtype=np.float64)
    sub_tst = test_df["curated_subtype"].astype(str).to_numpy()

    thresholds = select_oof_thresholds(y_oof, p_oof)

    operating_points_oof = {
        name: compute_threshold_metrics(y_oof, p_oof, th) for name, th in thresholds.items()
    }
    operating_points_test = {
        name: compute_threshold_metrics(y_tst, p_tst, th, subtypes=sub_tst)
        for name, th in thresholds.items()
    }

    # Bootstrap 95% CIs on test set at primary balanced-accuracy / Youden's J operating point
    boot_cis = cluster_bootstrap_cis(test_df, thresholds["max_youden_j"], n_boot=n_boot)

    return {
        "coverage": 1.0,
        "oof_global": {
            "n": int(len(oof_df)),
            "roc_auc": round(float(roc_auc_score(y_oof, p_oof)), 5),
            "pr_auc": round(float(average_precision_score(y_oof, p_oof)), 5),
            "brier": round(float(brier_score_loss(y_oof, p_oof)), 5),
        },
        "test_global": {
            "n": int(len(test_df)),
            "roc_auc": round(float(roc_auc_score(y_tst, p_tst)), 5),
            "pr_auc": round(float(average_precision_score(y_tst, p_tst)), 5),
            "brier": round(float(brier_score_loss(y_tst, p_tst)), 5),
        },
        "test_95pct_cluster_bootstrap_ci_at_max_youden_j": boot_cis,
        "oof_operating_points": operating_points_oof,
        "test_operating_points": operating_points_test,
    }
