"""5-fold patient-grouped cross-validation + held-out test inference runner.

Executes inside a single 1-GPU Beaker container (or locally), producing standardized
`oof_predictions.parquet`, `test_predictions.parquet`, `quarantine_predictions.parquet`,
and `metrics.json` in `--output-dir`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from heartsounds.config import ExperimentConfig, load_experiment_config
from heartsounds.data.torch_dataset import HeartSoundsDataset, masked_waveform_collate
from heartsounds.models.registry import build_model
from heartsounds.schema import PREDICTION_COLUMNS, validate_prediction_df
from heartsounds.training.trainer import OVERSAMPLE_FACTOR, predict_loader, train_single_fold


def _build_pred_table(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "recording_id": df["recording_id"].astype(str).to_numpy(),
            "split": df["split"].astype(str).to_numpy(),
            "patient_group": df["patient_group"].astype(str).to_numpy(),
            "y_true": df["curated_label"].astype(int).to_numpy(),
            "curated_subtype": df["curated_subtype"].astype(str).to_numpy(),
            "y_prob": np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6),
        }
    ).loc[:, list(PREDICTION_COLUMNS)]
    return validate_prediction_df(out)


def run_cross_validation(
    cfg: ExperimentConfig,
    *,
    output_dir: str | Path,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Train requested CV folds, ensemble fold checkpoints on `test`, and write artifacts."""
    t_start = time.time()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg.data.data_dir)
    dev_df = pd.read_parquet(data_root / "train_val_folds.parquet").reset_index(drop=True)
    test_df = pd.read_parquet(data_root / "test.parquet").reset_index(drop=True)
    quar_path = data_root / "quarantine.parquet"
    quar_df = pd.read_parquet(quar_path).reset_index(drop=True) if quar_path.is_file() else None

    test_ds = HeartSoundsDataset(
        test_df,
        sample_rate=cfg.data.sample_rate,
        is_train=False,
        augment=False,
        seed=cfg.seed,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=masked_waveform_collate,
        pin_memory=(dev.type == "cuda"),
    )

    quar_loader = None
    if quar_df is not None and len(quar_df) > 0:
        quar_ds = HeartSoundsDataset(
            quar_df,
            sample_rate=cfg.data.sample_rate,
            is_train=False,
            augment=False,
            seed=cfg.seed,
        )
        quar_loader = DataLoader(
            quar_ds,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=masked_waveform_collate,
        )

    oof_probs = np.full(len(dev_df), np.nan, dtype=np.float64)
    test_fold_logits: list[np.ndarray] = []
    quar_fold_logits: list[np.ndarray] = []
    fold_summaries: dict[str, object] = {}
    prior_offset = math.log(OVERSAMPLE_FACTOR)

    for fold_idx in cfg.folds:
        fold_name = f"fold_{fold_idx}"
        val_mask = (dev_df["split"] == fold_name).to_numpy()
        if not bool(val_mask.any()):
            raise ValueError(f"Split {fold_name!r} not found in train_val_folds.parquet")

        trn_sub = dev_df.loc[~val_mask].reset_index(drop=True)
        val_sub = dev_df.loc[val_mask].reset_index(drop=True)
        print(
            f"\n=== Training {cfg.name} | {fold_name} (train={len(trn_sub)}, val={len(val_sub)}) on {dev} ===",
            flush=True,
        )

        fold_out = train_single_fold(
            cfg,
            trn_sub,
            val_sub,
            fold_name=fold_name,
            device=dev,
            seed=cfg.seed + fold_idx * 101,
        )
        oof_probs[val_mask] = fold_out.val_probs
        fold_summaries[fold_name] = {
            "best_epoch": fold_out.best_epoch,
            "val_roc_auc": round(fold_out.best_val_roc_auc, 5),
            "val_pr_auc": round(fold_out.best_val_pr_auc, 5),
            "val_max_youden_j": round(fold_out.best_val_youden_j, 5),
            "history": fold_out.history,
        }

        eval_model = build_model(cfg.model).to(dev)
        eval_model.load_state_dict(fold_out.best_state_dict)
        t_probs, _, _ = predict_loader(
            eval_model,
            test_loader,
            dev,
            prior_logit_offset=prior_offset,
            use_amp=cfg.optim.mixed_precision,
        )
        test_fold_logits.append(logit(np.clip(t_probs, 1e-6, 1.0 - 1e-6)))

        if quar_loader is not None:
            q_probs, _, _ = predict_loader(
                eval_model,
                quar_loader,
                dev,
                prior_logit_offset=prior_offset,
                use_amp=cfg.optim.mixed_precision,
            )
            quar_fold_logits.append(logit(np.clip(q_probs, 1e-6, 1.0 - 1e-6)))

    evaluated_mask = ~np.isnan(oof_probs)
    eval_dev_df = dev_df.loc[evaluated_mask].reset_index(drop=True)
    eval_oof_probs = oof_probs[evaluated_mask]

    oof_table = _build_pred_table(eval_dev_df, eval_oof_probs)
    test_probs = expit(np.mean(np.stack(test_fold_logits, axis=0), axis=0))
    test_table = _build_pred_table(test_df, test_probs)

    oof_table.to_parquet(out_root / "oof_predictions.parquet", index=False)
    test_table.to_parquet(out_root / "test_predictions.parquet", index=False)

    if quar_df is not None and quar_fold_logits:
        quar_probs = expit(np.mean(np.stack(quar_fold_logits, axis=0), axis=0))
        quar_out = quar_df[
            ["recording_id", "split", "patient_group", "curated_label", "curated_subtype"]
        ].rename(columns={"curated_label": "y_true"})
        quar_out["y_prob"] = np.clip(quar_probs, 1e-6, 1.0 - 1e-6)
        quar_out.to_parquet(out_root / "quarantine_predictions.parquet", index=False)

    y_oof = oof_table["y_true"].to_numpy()
    y_tst = test_table["y_true"].to_numpy()
    summary: dict[str, object] = {
        "experiment": cfg.name,
        "model": cfg.model.name,
        "device": str(dev),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "n_oof": int(len(oof_table)),
        "n_test": int(len(test_table)),
        "oof_roc_auc": round(float(roc_auc_score(y_oof, oof_table["y_prob"])), 5),
        "oof_pr_auc": round(float(average_precision_score(y_oof, oof_table["y_prob"])), 5),
        "test_roc_auc": round(float(roc_auc_score(y_tst, test_table["y_prob"])), 5),
        "test_pr_auc": round(float(average_precision_score(y_tst, test_table["y_prob"])), 5),
        "folds": fold_summaries,
        "config": cfg.to_dict(),
    }
    (out_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n[{cfg.name}] Completed: OOF ROC-AUC={summary['oof_roc_auc']} PR-AUC={summary['oof_pr_auc']} | "
        f"Test ROC-AUC={summary['test_roc_auc']} PR-AUC={summary['test_pr_auc']}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run neural cross-validation & test inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config, overrides=args.overrides)
    out_dir = args.output_dir or (Path(cfg.output_dir) / cfg.name)
    run_cross_validation(cfg, output_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
