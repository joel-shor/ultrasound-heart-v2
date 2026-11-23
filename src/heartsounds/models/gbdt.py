"""Model Family 1: Patient-grouped GBDT & linear classifiers on auditable DSP features.

Trains and evaluates three tabular learners on the exact same 5-fold patient-grouped
splits (`fold_0..fold_4`) and held-out `test` split:
1. L2-regularized Logistic Regression (interpretable linear baseline)
2. XGBoost (`XGBClassifier`)
3. LightGBM (`LGBMClassifier`)
Plus an equal-logit GBDT blend (`gbdt_ensemble`), emitting standardized `PredictionTable`s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from heartsounds.schema import PREDICTION_COLUMNS, validate_prediction_df

METADATA_COLS: frozenset[str] = frozenset(
    {
        "recording_id",
        "waveform_md5",
        "primary_raw_gcs_uri",
        "curated_label",
        "curated_subtype",
        "patient_group",
        "sample_rate",
        "split",
        # Exclude raw clip length & raw beat count so classifiers are 100% duration-invariant
        "duration_s",
        "ibi_count",
    }
)


@dataclass(frozen=True)
class TabularModelResult:
    """Out-of-fold, test, quarantine, and feature-importance outputs for a tabular model."""

    model_name: str
    oof_df: pd.DataFrame
    test_df: pd.DataFrame
    quarantine_df: pd.DataFrame | None
    oof_roc_auc: float
    oof_pr_auc: float
    feature_importances: pd.DataFrame


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return sorted numeric DSP feature column names, excluding metadata columns."""
    return sorted(c for c in features_df.columns if c not in METADATA_COLS)


def _format_prediction_df(meta_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "recording_id": meta_df["recording_id"].astype(str).to_numpy(),
            "split": meta_df["split"].astype(str).to_numpy(),
            "patient_group": meta_df["patient_group"].astype(str).to_numpy(),
            "y_true": meta_df["curated_label"].astype(int).to_numpy(),
            "curated_subtype": meta_df["curated_subtype"].astype(str).to_numpy(),
            "y_prob": np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6),
        }
    ).loc[:, list(PREDICTION_COLUMNS)]
    return validate_prediction_df(out)


def train_cv_tabular_models(
    dev_features_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    *,
    quarantine_features_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict[str, TabularModelResult]:
    """Train 5-fold CV Logistic, XGBoost, LightGBM, and GBDT blend models."""
    feat_cols = get_feature_columns(dev_features_df)
    x_dev = dev_features_df[feat_cols].to_numpy(dtype=np.float32)
    y_dev = dev_features_df["curated_label"].to_numpy(dtype=np.int32)
    splits_dev = dev_features_df["split"].astype(str).to_numpy()

    x_test = test_features_df[feat_cols].to_numpy(dtype=np.float32)
    x_quar = (
        quarantine_features_df[feat_cols].to_numpy(dtype=np.float32)
        if quarantine_features_df is not None and len(quarantine_features_df) > 0
        else None
    )

    fold_names = sorted(np.unique(splits_dev).tolist())
    n_pos = max(1, int(np.sum(y_dev == 1)))
    n_neg = max(1, int(np.sum(y_dev == 0)))
    pos_ratio = float(n_neg / n_pos)

    model_specs = ("logreg_dsp", "xgboost_dsp", "lightgbm_dsp")
    oof_probs: dict[str, np.ndarray] = {
        m: np.zeros(len(dev_features_df), dtype=np.float64) for m in model_specs
    }
    test_fold_logits: dict[str, list[np.ndarray]] = {m: [] for m in model_specs}
    quar_fold_logits: dict[str, list[np.ndarray]] = {m: [] for m in model_specs}
    importances_accum: dict[str, np.ndarray] = {
        m: np.zeros(len(feat_cols), dtype=np.float64) for m in model_specs
    }

    for fold_idx, val_split in enumerate(fold_names):
        val_mask = splits_dev == val_split
        trn_mask = ~val_mask

        x_trn, y_trn = x_dev[trn_mask], y_dev[trn_mask]
        x_val, y_val = x_dev[val_mask], y_dev[val_mask]
        fold_seed = seed + fold_idx * 17

        # 1. Regularized Logistic Regression
        lr_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.25,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=fold_seed,
                    ),
                ),
            ]
        )
        lr_pipe.fit(x_trn, y_trn)
        p_val_lr = lr_pipe.predict_proba(x_val)[:, 1]
        p_tst_lr = lr_pipe.predict_proba(x_test)[:, 1]
        oof_probs["logreg_dsp"][val_mask] = p_val_lr
        test_fold_logits["logreg_dsp"].append(logit(np.clip(p_tst_lr, 1e-6, 1.0 - 1e-6)))
        if x_quar is not None:
            quar_fold_logits["logreg_dsp"].append(
                logit(np.clip(lr_pipe.predict_proba(x_quar)[:, 1], 1e-6, 1.0 - 1e-6))
            )
        importances_accum["logreg_dsp"] += np.abs(lr_pipe.named_steps["clf"].coef_[0])

        # 2. XGBoost (shallow regularized trees + early stopping on val AUCPR/AUC)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=900,
            max_depth=4,
            min_child_weight=5,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.5,
            scale_pos_weight=np.sqrt(pos_ratio),
            eval_metric="aucpr",
            early_stopping_rounds=60,
            n_jobs=8,
            random_state=fold_seed,
        )
        xgb_clf.fit(x_trn, y_trn, eval_set=[(x_val, y_val)], verbose=False)
        p_val_xgb = xgb_clf.predict_proba(x_val)[:, 1]
        p_tst_xgb = xgb_clf.predict_proba(x_test)[:, 1]
        oof_probs["xgboost_dsp"][val_mask] = p_val_xgb
        test_fold_logits["xgboost_dsp"].append(logit(np.clip(p_tst_xgb, 1e-6, 1.0 - 1e-6)))
        if x_quar is not None:
            quar_fold_logits["xgboost_dsp"].append(
                logit(np.clip(xgb_clf.predict_proba(x_quar)[:, 1], 1e-6, 1.0 - 1e-6))
            )
        importances_accum["xgboost_dsp"] += xgb_clf.feature_importances_

        # 3. LightGBM (leaf-wise regularized trees + early stopping)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=900,
            num_leaves=15,
            max_depth=5,
            min_child_samples=18,
            learning_rate=0.03,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.5,
            scale_pos_weight=np.sqrt(pos_ratio),
            n_jobs=8,
            random_state=fold_seed,
            verbose=-1,
        )
        lgb_clf.fit(
            x_trn,
            y_trn,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        p_val_lgb = lgb_clf.predict_proba(x_val)[:, 1]
        p_tst_lgb = lgb_clf.predict_proba(x_test)[:, 1]
        oof_probs["lightgbm_dsp"][val_mask] = p_val_lgb
        test_fold_logits["lightgbm_dsp"].append(logit(np.clip(p_tst_lgb, 1e-6, 1.0 - 1e-6)))
        if x_quar is not None:
            quar_fold_logits["lightgbm_dsp"].append(
                logit(np.clip(lgb_clf.predict_proba(x_quar)[:, 1], 1e-6, 1.0 - 1e-6))
            )
        importances_accum["lightgbm_dsp"] += lgb_clf.feature_importances_

    results: dict[str, TabularModelResult] = {}
    for name in model_specs:
        p_oof = oof_probs[name]
        p_test = expit(np.mean(np.stack(test_fold_logits[name], axis=0), axis=0))
        q_df = None
        if quarantine_features_df is not None and quar_fold_logits[name]:
            p_quar = expit(np.mean(np.stack(quar_fold_logits[name], axis=0), axis=0))
            q_df = _format_prediction_df(quarantine_features_df, p_quar)
        imp_df = (
            pd.DataFrame(
                {
                    "feature": feat_cols,
                    "importance": importances_accum[name] / len(fold_names),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        results[name] = TabularModelResult(
            model_name=name,
            oof_df=_format_prediction_df(dev_features_df, p_oof),
            test_df=_format_prediction_df(test_features_df, p_test),
            quarantine_df=q_df,
            oof_roc_auc=float(roc_auc_score(y_dev, p_oof)),
            oof_pr_auc=float(average_precision_score(y_dev, p_oof)),
            feature_importances=imp_df,
        )

    # Combined Family 1 GBDT blend (50% XGBoost + 50% LightGBM in logit space)
    blend_oof = expit(
        0.5 * logit(np.clip(oof_probs["xgboost_dsp"], 1e-6, 1.0 - 1e-6))
        + 0.5 * logit(np.clip(oof_probs["lightgbm_dsp"], 1e-6, 1.0 - 1e-6))
    )
    blend_test = expit(
        0.5 * logit(results["xgboost_dsp"].test_df["y_prob"].to_numpy())
        + 0.5 * logit(results["lightgbm_dsp"].test_df["y_prob"].to_numpy())
    )
    blend_quar = None
    if (
        quarantine_features_df is not None
        and results["xgboost_dsp"].quarantine_df is not None
        and results["lightgbm_dsp"].quarantine_df is not None
    ):
        pq_blend = expit(
            0.5 * logit(results["xgboost_dsp"].quarantine_df["y_prob"].to_numpy())
            + 0.5 * logit(results["lightgbm_dsp"].quarantine_df["y_prob"].to_numpy())
        )
        blend_quar = _format_prediction_df(quarantine_features_df, pq_blend)

    results["gbdt_dsp_blend"] = TabularModelResult(
        model_name="gbdt_dsp_blend",
        oof_df=_format_prediction_df(dev_features_df, blend_oof),
        test_df=_format_prediction_df(test_features_df, blend_test),
        quarantine_df=blend_quar,
        oof_roc_auc=float(roc_auc_score(y_dev, blend_oof)),
        oof_pr_auc=float(average_precision_score(y_dev, blend_oof)),
        feature_importances=results["xgboost_dsp"].feature_importances,
    )
    return results


def serialize_tabular_summary(results: dict[str, TabularModelResult]) -> dict[str, Any]:
    """Format compact OOF summary metrics for logging and JSON persistence."""
    return {
        name: {
            "oof_roc_auc": round(res.oof_roc_auc, 5),
            "oof_pr_auc": round(res.oof_pr_auc, 5),
            "top_10_features": res.feature_importances.head(10).to_dict(orient="records"),
        }
        for name, res in results.items()
    }
