"""Unit tests for Evaluation 1 (binary), Evaluation 2 (3-state triage), and ensemble fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heartsounds.eval.ensemble import build_all_ensembles_and_ablations
from heartsounds.eval.metrics import evaluate_binary_suite
from heartsounds.eval.triage import evaluate_triage_suite


def _make_synthetic_preds(seed: int, noise: float, split_mode: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    y = (np.arange(n) < 25).astype(int)
    raw = np.where(y == 1, 1.4, -1.6) + rng.normal(0.0, noise, size=n)
    probs = 1.0 / (1.0 + np.exp(-raw))
    splits = [f"fold_{i % 5}" for i in range(n)] if split_mode == "oof" else ["test"] * n
    return pd.DataFrame(
        {
            "recording_id": [f"{split_mode}_{i:04d}" for i in range(n)],
            "split": splits,
            "patient_group": [f"subj_{i // 2:03d}" for i in range(n)],
            "y_true": y,
            "curated_subtype": np.where(y == 1, "PREMATURE_ATRIAL_CONTRACTION", "NORMAL"),
            "y_prob": probs,
        }
    )


def test_binary_triage_and_three_family_ensemble_suites():
    oof_m1 = _make_synthetic_preds(1, noise=1.1, split_mode="oof")
    tst_m1 = _make_synthetic_preds(2, noise=1.1, split_mode="test")
    oof_m2 = _make_synthetic_preds(3, noise=0.9, split_mode="oof")
    tst_m2 = _make_synthetic_preds(4, noise=0.9, split_mode="test")
    oof_m3 = _make_synthetic_preds(5, noise=1.0, split_mode="oof")
    tst_m3 = _make_synthetic_preds(6, noise=1.0, split_mode="test")

    ensembles, _ = build_all_ensembles_and_ablations(
        {"m1": oof_m1, "m2": oof_m2, "m3": oof_m3},
        {"m1": tst_m1, "m2": tst_m2, "m3": tst_m3},
    )
    assert "fusion_3family_logit_blend" in ensembles
    assert "pair_m1__plus__m2" in ensembles

    blend = ensembles["fusion_3family_logit_blend"]
    ev1 = evaluate_binary_suite(blend.oof_df, blend.test_df, n_boot=100)
    assert ev1["test_global"]["roc_auc"] > 0.90
    assert (
        "PREMATURE_ATRIAL_CONTRACTION"
        in ev1["test_operating_points"]["max_youden_j"]["subtype_recall"]
    )

    center_th = ev1["oof_operating_points"]["max_youden_j"]["threshold"]
    ev2, _, tst_states = evaluate_triage_suite(
        blend.oof_df,
        blend.test_df,
        center_threshold=center_th,
        oof_quality=np.full(len(blend.oof_df), 0.55),
        test_quality=np.full(len(blend.test_df), 0.55),
    )
    assert len(tst_states) == len(blend.test_df)
    b0 = ev2["budget_curves"]["yellow_budget_00pct"]["test"]
    b15 = ev2["budget_curves"]["yellow_budget_15pct"]["test"]
    assert b0["coverage"] == 1.0
    assert b15["coverage"] < 1.0
    assert b15["definitive_youden_j"] >= b0["definitive_youden_j"] - 0.02
