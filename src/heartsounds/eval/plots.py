"""Publication-grade matplotlib visualizations for Evaluation 1 and Evaluation 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_and_pr_frontiers(
    test_tables: dict[str, pd.DataFrame],
    out_path: str | Path,
) -> None:
    """Plot held-out test ROC (Sensitivity vs 1-Specificity) and Precision-Recall curves."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    palette = sns.color_palette("tab10", n_colors=max(4, len(test_tables)))

    for idx, (name, df) in enumerate(test_tables.items()):
        y = df["y_true"].to_numpy(dtype=np.int32)
        p = df["y_prob"].to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        pr_auc = average_precision_score(y, p)

        lw = 2.6 if name.startswith("fusion_") else 1.7
        ls = "-" if name.startswith("fusion_") else "--"
        ax_roc.plot(fpr, tpr, lw=lw, ls=ls, color=palette[idx], label=f"{name} (AUC={roc_auc:.3f})")
        ax_pr.plot(rec, prec, lw=lw, ls=ls, color=palette[idx], label=f"{name} (AP={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.0, ls=":")
    ax_roc.set_xlabel("1 - Specificity (False Positive Rate)")
    ax_roc.set_ylabel("Sensitivity (True Positive Rate)")
    ax_roc.set_title("Eval 1: Held-Out Test ROC Frontier (100% Coverage)")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    ax_roc.grid(alpha=0.25)

    ax_pr.set_xlabel("Sensitivity (Recall)")
    ax_pr.set_ylabel("Precision (PPV)")
    ax_pr.set_title("Eval 1: Held-Out Test Precision-Recall Frontier")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    ax_pr.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_triage_coverage_tradeoff(
    triage_report: dict[str, Any],
    model_label: str,
    out_path: str | Path,
) -> None:
    """Plot Evaluation 2 Sensitivity, Specificity, and Youden's J vs. Yellow Re-record Rate."""
    curves = triage_report["budget_curves"]
    yellow_rates = []
    sens_vals = []
    spec_vals = []
    j_vals = []
    actionable_vals = []

    for item in curves.values():
        tst = item["test"]
        yellow_rates.append(100.0 * tst["yellow_rate"])
        sens_vals.append(100.0 * tst["definitive_sensitivity"])
        spec_vals.append(100.0 * tst["definitive_specificity"])
        j_vals.append(100.0 * tst["definitive_youden_j"])
        actionable_vals.append(
            100.0 * tst["all_abnormal_disposition"]["red_or_yellow_actionable_rate"]
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(
        yellow_rates,
        sens_vals,
        marker="o",
        lw=2.2,
        label="Definitive Sensitivity (Red | Green/Red)",
    )
    ax.plot(
        yellow_rates,
        spec_vals,
        marker="s",
        lw=2.2,
        label="Definitive Specificity (Green | Green/Red)",
    )
    ax.plot(
        yellow_rates,
        actionable_vals,
        marker="^",
        lw=2.0,
        ls="--",
        label="All-Abnormal Actionable Rate (Red + Yellow)",
    )
    ax.plot(
        yellow_rates,
        j_vals,
        marker="d",
        lw=2.0,
        ls=":",
        label="Definitive Youden's J (Sens + Spec - 100)",
    )

    ax.set_xlabel("Actual Held-Out Test Yellow ('Re-record') Rate (%)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Eval 2: 3-State Clinical Triage Tradeoff ({model_label})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_and_subtypes(
    test_df: pd.DataFrame,
    threshold: float,
    model_label: str,
    out_path: str | Path,
) -> None:
    """Plot held-out test confusion matrix alongside per-subtype recall bars."""
    y = test_df["y_true"].to_numpy(dtype=np.int32)
    p = test_df["y_prob"].to_numpy(dtype=np.float64)
    yhat = (p >= threshold).astype(np.int32)
    cm = confusion_matrix(y, yhat, labels=[0, 1])

    fig, (ax_cm, ax_sub) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Normal", "Pred Abnormal"],
        yticklabels=["True Normal", "True Abnormal"],
        ax=ax_cm,
    )
    ax_cm.set_title(f"{model_label} Confusion Matrix (th={threshold:.3f})")

    abn_sub = test_df[test_df["y_true"] == 1].copy()
    abn_sub["detected"] = (abn_sub["y_prob"] >= threshold).astype(int)
    grouped = (
        abn_sub.groupby("curated_subtype")["detected"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("count", ascending=True)
    )
    labels = [f"{r.curated_subtype} (n={int(r['count'])})" for _, r in grouped.iterrows()]
    ax_sub.barh(labels, grouped["mean"] * 100.0, color="#E53935", alpha=0.85)
    ax_sub.set_xlim(0, 105)
    ax_sub.set_xlabel("Sensitivity / Detection Rate (%)")
    ax_sub.set_title("Held-Out Test Recall by Abnormal Subtype")
    ax_sub.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
