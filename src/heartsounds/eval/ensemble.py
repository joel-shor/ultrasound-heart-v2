"""Out-of-fold multi-model fusion (Simplex Logit Blend, Nested Stacking, Rank Average)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from heartsounds.schema import PREDICTION_COLUMNS, validate_prediction_df


@dataclass(frozen=True)
class EnsembleOutput:
    """Fused out-of-fold and held-out test predictions plus learned model weights."""

    name: str
    member_names: tuple[str, ...]
    weights: dict[str, float]
    oof_df: pd.DataFrame
    test_df: pd.DataFrame


def align_prediction_tables(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Align multiple `PredictionTable`s by `recording_id` and validate label consistency."""
    if not tables:
        raise ValueError("align_prediction_tables requires at least one table")

    names = list(tables.keys())
    base = (
        validate_prediction_df(tables[names[0]]).sort_values("recording_id").reset_index(drop=True)
    )
    probs_by_model: dict[str, np.ndarray] = {names[0]: base["y_prob"].to_numpy(dtype=np.float64)}

    for name in names[1:]:
        cur = (
            validate_prediction_df(tables[name]).sort_values("recording_id").reset_index(drop=True)
        )
        if not np.array_equal(base["recording_id"].to_numpy(), cur["recording_id"].to_numpy()):
            raise ValueError(f"recording_id mismatch between {names[0]} and {name}")
        if not np.array_equal(base["y_true"].to_numpy(), cur["y_true"].to_numpy()):
            raise ValueError(f"y_true mismatch between {names[0]} and {name}")
        probs_by_model[name] = cur["y_prob"].to_numpy(dtype=np.float64)

    return base, probs_by_model


def _simplex_grid(n_models: int, step: float = 0.05) -> np.ndarray:
    units = int(round(1.0 / step))
    rows: list[list[float]] = []

    def _rec(rem: int, depth: int, prefix: list[int]) -> None:
        if depth == n_models - 1:
            rows.append([*(x / units for x in prefix), rem / units])
            return
        for k in range(rem + 1):
            _rec(rem - k, depth + 1, [*prefix, k])

    _rec(units, 0, [])
    return np.asarray(rows, dtype=np.float64)


def _score_blend(y_true: np.ndarray, probs: np.ndarray) -> float:
    return 0.5 * float(roc_auc_score(y_true, probs)) + 0.5 * float(
        average_precision_score(y_true, probs)
    )


def fit_nested_simplex_logit_blend(
    oof_tables: dict[str, pd.DataFrame],
    test_tables: dict[str, pd.DataFrame],
    *,
    ensemble_name: str = "fusion_logit_blend",
    quarantine_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[EnsembleOutput, pd.DataFrame | None]:
    """Learn convex logit weights per CV fold (`folds != f`) and evaluate on `fold == f` & `test`."""
    base_oof, oof_probs_map = align_prediction_tables(oof_tables)
    base_tst, tst_probs_map = align_prediction_tables(test_tables)

    members = tuple(oof_tables.keys())
    oof_logits = np.column_stack(
        [logit(np.clip(oof_probs_map[m], 1e-6, 1.0 - 1e-6)) for m in members]
    )
    tst_logits = np.column_stack(
        [logit(np.clip(tst_probs_map[m], 1e-6, 1.0 - 1e-6)) for m in members]
    )
    y_oof = base_oof["y_true"].to_numpy(dtype=np.int32)
    splits = base_oof["split"].astype(str).to_numpy()
    fold_names = sorted(np.unique(splits).tolist())

    grid = _simplex_grid(len(members), step=0.05)
    blended_oof = np.zeros(len(base_oof), dtype=np.float64)
    fold_weights: list[np.ndarray] = []

    for f_name in fold_names:
        val_mask = splits == f_name
        trn_mask = ~val_mask
        y_trn = y_oof[trn_mask]
        l_trn = oof_logits[trn_mask]

        best_w = grid[0]
        best_s = -1.0
        for w in grid:
            cand_p = expit(l_trn @ w)
            s = _score_blend(y_trn, cand_p)
            if s > best_s:
                best_s = s
                best_w = w
        fold_weights.append(best_w)
        blended_oof[val_mask] = expit(oof_logits[val_mask] @ best_w)

    mean_weights = np.mean(np.stack(fold_weights, axis=0), axis=0)
    mean_weights /= np.sum(mean_weights)
    blended_tst = expit(tst_logits @ mean_weights)

    oof_out = base_oof[list(PREDICTION_COLUMNS[:-1])].copy()
    oof_out["y_prob"] = np.clip(blended_oof, 1e-6, 1.0 - 1e-6)
    tst_out = base_tst[list(PREDICTION_COLUMNS[:-1])].copy()
    tst_out["y_prob"] = np.clip(blended_tst, 1e-6, 1.0 - 1e-6)

    quar_out = None
    if quarantine_tables and all(m in quarantine_tables for m in members):
        q_base = quarantine_tables[members[0]].sort_values("recording_id").reset_index(drop=True)
        q_logits = np.column_stack(
            [
                logit(
                    np.clip(
                        quarantine_tables[m]
                        .sort_values("recording_id")["y_prob"]
                        .to_numpy(dtype=np.float64),
                        1e-6,
                        1.0 - 1e-6,
                    )
                )
                for m in members
            ]
        )
        quar_out = q_base.copy()
        quar_out["y_prob"] = np.clip(expit(q_logits @ mean_weights), 1e-6, 1.0 - 1e-6)

    return (
        EnsembleOutput(
            name=ensemble_name,
            member_names=members,
            weights={m: round(float(w), 4) for m, w in zip(members, mean_weights, strict=True)},
            oof_df=validate_prediction_df(oof_out),
            test_df=validate_prediction_df(tst_out),
        ),
        quar_out,
    )


def fit_nested_stacker(
    oof_tables: dict[str, pd.DataFrame],
    test_tables: dict[str, pd.DataFrame],
    *,
    ensemble_name: str = "fusion_logistic_stacker",
) -> EnsembleOutput:
    """Fit out-of-fold L2 Logistic Regression meta-learner over member logits & dispersion."""
    base_oof, oof_probs_map = align_prediction_tables(oof_tables)
    base_tst, tst_probs_map = align_prediction_tables(test_tables)
    members = tuple(oof_tables.keys())

    def _meta_features(p_map: dict[str, np.ndarray]) -> np.ndarray:
        l_mat = np.column_stack([logit(np.clip(p_map[m], 1e-6, 1.0 - 1e-6)) for m in members])
        l_max = np.max(l_mat, axis=1, keepdims=True)
        l_std = np.std(l_mat, axis=1, keepdims=True)
        return np.hstack([l_mat, l_max, l_std])

    x_oof = _meta_features(oof_probs_map)
    x_tst = _meta_features(tst_probs_map)
    y_oof = base_oof["y_true"].to_numpy(dtype=np.int32)
    splits = base_oof["split"].astype(str).to_numpy()
    fold_names = sorted(np.unique(splits).tolist())

    stacked_oof = np.zeros(len(base_oof), dtype=np.float64)
    tst_fold_logits: list[np.ndarray] = []
    coef_accum = np.zeros(len(members), dtype=np.float64)

    for f_name in fold_names:
        val_mask = splits == f_name
        trn_mask = ~val_mask
        meta = LogisticRegression(C=0.35, max_iter=1000)
        meta.fit(x_oof[trn_mask], y_oof[trn_mask])
        stacked_oof[val_mask] = meta.predict_proba(x_oof[val_mask])[:, 1]
        p_t = meta.predict_proba(x_tst)[:, 1]
        tst_fold_logits.append(logit(np.clip(p_t, 1e-6, 1.0 - 1e-6)))
        coef_accum += meta.coef_[0, : len(members)]

    mean_coefs = coef_accum / len(fold_names)
    oof_out = base_oof[list(PREDICTION_COLUMNS[:-1])].copy()
    oof_out["y_prob"] = np.clip(stacked_oof, 1e-6, 1.0 - 1e-6)
    tst_out = base_tst[list(PREDICTION_COLUMNS[:-1])].copy()
    tst_out["y_prob"] = np.clip(
        expit(np.mean(np.stack(tst_fold_logits, axis=0), axis=0)), 1e-6, 1.0 - 1e-6
    )

    return EnsembleOutput(
        name=ensemble_name,
        member_names=members,
        weights={m: round(float(c), 4) for m, c in zip(members, mean_coefs, strict=True)},
        oof_df=validate_prediction_df(oof_out),
        test_df=validate_prediction_df(tst_out),
    )


def fit_rank_average(
    oof_tables: dict[str, pd.DataFrame],
    test_tables: dict[str, pd.DataFrame],
    *,
    ensemble_name: str = "fusion_rank_avg",
) -> EnsembleOutput:
    """Non-parametric uniform rank-percentile average calibrated to base prevalence."""
    base_oof, oof_probs_map = align_prediction_tables(oof_tables)
    base_tst, tst_probs_map = align_prediction_tables(test_tables)
    members = tuple(oof_tables.keys())

    r_oof = np.mean([rankdata(oof_probs_map[m]) / (len(base_oof) + 1.0) for m in members], axis=0)
    r_tst = np.mean([rankdata(tst_probs_map[m]) / (len(base_tst) + 1.0) for m in members], axis=0)

    oof_out = base_oof[list(PREDICTION_COLUMNS[:-1])].copy()
    oof_out["y_prob"] = np.clip(r_oof, 1e-6, 1.0 - 1e-6)
    tst_out = base_tst[list(PREDICTION_COLUMNS[:-1])].copy()
    tst_out["y_prob"] = np.clip(r_tst, 1e-6, 1.0 - 1e-6)

    w = round(1.0 / len(members), 4)
    return EnsembleOutput(
        name=ensemble_name,
        member_names=members,
        weights={m: w for m in members},
        oof_df=validate_prediction_df(oof_out),
        test_df=validate_prediction_df(tst_out),
    )


def build_all_ensembles_and_ablations(
    primary_oof: dict[str, pd.DataFrame],
    primary_test: dict[str, pd.DataFrame],
    quarantine_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, EnsembleOutput], pd.DataFrame | None]:
    """Build pairwise ablations plus 3-way fusion ensembles across primary model families."""
    names = list(primary_oof.keys())
    ensembles: dict[str, EnsembleOutput] = {}
    best_quar: pd.DataFrame | None = None

    if len(names) >= 3:
        for a, b in itertools.combinations(names, 2):
            pair_name = f"pair_{a}__plus__{b}"
            ens, _ = fit_nested_simplex_logit_blend(
                {a: primary_oof[a], b: primary_oof[b]},
                {a: primary_test[a], b: primary_test[b]},
                ensemble_name=pair_name,
            )
            ensembles[pair_name] = ens

    if len(names) >= 2:
        blend, best_quar = fit_nested_simplex_logit_blend(
            primary_oof,
            primary_test,
            ensemble_name="fusion_3family_logit_blend",
            quarantine_tables=quarantine_tables,
        )
        stacker = fit_nested_stacker(
            primary_oof, primary_test, ensemble_name="fusion_3family_stacker"
        )
        rank_ens = fit_rank_average(
            primary_oof, primary_test, ensemble_name="fusion_3family_rank_avg"
        )
        ensembles[blend.name] = blend
        ensembles[stacker.name] = stacker
        ensembles[rank_ens.name] = rank_ens

    return ensembles, best_quar
