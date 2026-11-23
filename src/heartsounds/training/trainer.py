"""Single-fold PyTorch trainer with bf16 AMP, mild minority oversampling, and Mixup."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from heartsounds.config import ExperimentConfig
from heartsounds.data.torch_dataset import (
    HeartSoundsDataset,
    WaveformBatch,
    masked_waveform_collate,
)
from heartsounds.models.losses import CompositeDopplerLoss
from heartsounds.models.pretrain import run_fold_masked_ssl_pretrain
from heartsounds.models.registry import build_model

OVERSAMPLE_FACTOR = 3.0


@dataclass(frozen=True)
class FoldTrainOutput:
    """Best validation metrics, state_dict, and out-of-fold validation predictions."""

    fold_name: str
    best_epoch: int
    best_val_roc_auc: float
    best_val_pr_auc: float
    best_val_youden_j: float
    val_probs: np.ndarray
    history: list[dict[str, float]]
    best_state_dict: dict[str, torch.Tensor]


def _max_youden_j(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _apply_waveform_mixup(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or waveforms.shape[0] < 2:
        return waveforms, lengths, labels
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(waveforms.shape[0], device=waveforms.device)
    mixed_wav = lam * waveforms + (1.0 - lam) * waveforms[perm]
    mixed_len = torch.maximum(lengths, lengths[perm])
    mixed_lbl = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_wav, mixed_len, mixed_lbl


def _build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = [{"params": head_params, "lr": cfg.optim.lr}]
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": cfg.optim.lr * cfg.optim.backbone_lr_scale,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=cfg.optim.weight_decay)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader[WaveformBatch],
    device: torch.device,
    *,
    prior_logit_offset: float = 0.0,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run deterministic full-recording inference, returning `(probs, y_true, recording_ids)`."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_ids: list[str] = []

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    )

    for batch in loader:
        wav = batch.waveforms.to(device, non_blocking=True)
        lens = batch.lengths.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(wav, lens)
            logits = out["logits"].float() - prior_logit_offset
            probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(batch.labels.numpy())
        all_ids.extend(batch.recording_ids)

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0),
        all_ids,
    )


def train_single_fold(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    fold_name: str,
    device: torch.device,
    seed: int,
) -> FoldTrainOutput:
    """Train one CV fold and return the best checkpoint + calibrated validation predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = HeartSoundsDataset(
        train_df,
        sample_rate=cfg.data.sample_rate,
        window_seconds=cfg.data.train_window_seconds,
        is_train=True,
        augment=cfg.data.augment,
        crops_per_recording=cfg.data.crops_per_recording_epoch,
        seed=seed,
    )
    val_ds = HeartSoundsDataset(
        val_df,
        sample_rate=cfg.data.sample_rate,
        is_train=False,
        augment=False,
        seed=seed,
    )

    # Mild 3x minority oversampling ensures every mini-batch has several positives for pairwise AUC loss
    base_labels = np.tile(train_ds.labels, train_ds.crops_per_recording)
    sample_weights = np.where(base_labels > 0.5, OVERSAMPLE_FACTOR, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    prior_offset = math.log(OVERSAMPLE_FACTOR)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=masked_waveform_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) > cfg.data.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=masked_waveform_collate,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.model).to(device)

    if cfg.optim.ssl_epochs > 0:
        ssl_ds = HeartSoundsDataset(
            train_df,
            sample_rate=cfg.data.sample_rate,
            window_seconds=cfg.data.train_window_seconds,
            is_train=True,
            augment=cfg.data.augment,
            crops_per_recording=1,
            seed=seed + 17,
        )
        ssl_loader = DataLoader(
            ssl_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=masked_waveform_collate,
            pin_memory=(device.type == "cuda"),
            drop_last=(len(ssl_ds) > cfg.data.batch_size),
        )
        run_fold_masked_ssl_pretrain(model, ssl_loader, cfg, device, fold_name=fold_name)

    optimizer = _build_optimizer(model, cfg)

    n_pos = max(1, int((train_df["curated_label"] == 1).sum()))
    n_neg = max(1, int((train_df["curated_label"] == 0).sum()))
    effective_pos_ratio = max(1.0, (n_neg / n_pos) / OVERSAMPLE_FACTOR)
    pos_weight = 1.0 + cfg.optim.pos_weight_scale * (math.sqrt(effective_pos_ratio) - 1.0)

    criterion = CompositeDopplerLoss(
        focal_gamma=cfg.optim.focal_gamma,
        pos_weight=pos_weight,
        auc_weight=cfg.optim.auc_loss_weight,
        subtype_weight=cfg.optim.subtype_loss_weight,
    )

    total_steps = max(1, cfg.optim.epochs * len(train_loader))
    warmup_steps = cfg.optim.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_enabled = cfg.optim.mixed_precision and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_enabled and not use_bf16))

    best_score = -1.0
    best_epoch = 0
    best_roc = 0.0
    best_pr = 0.0
    best_j = 0.0
    best_probs = np.zeros(len(val_df), dtype=np.float64)
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            wav = batch.waveforms.to(device, non_blocking=True)
            lens = batch.lengths.to(device, non_blocking=True)
            lbl = batch.labels.to(device, non_blocking=True)
            sub = batch.subtypes.to(device, non_blocking=True)

            use_mix = cfg.optim.mixup_alpha > 0.0 and (np.random.random() < 0.4)
            if use_mix:
                wav, lens, lbl_used = _apply_waveform_mixup(wav, lens, lbl, cfg.optim.mixup_alpha)
                sub_used = None
            else:
                lbl_used = lbl
                sub_used = sub

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(wav, lens)
                loss = criterion(
                    out["logits"].float(),
                    lbl_used.float(),
                    out["subtype_logits"].float() if sub_used is not None else None,
                    sub_used,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                optimizer.step()

            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        val_probs, val_y, _ = predict_loader(
            model,
            val_loader,
            device,
            prior_logit_offset=prior_offset,
            use_amp=cfg.optim.mixed_precision,
        )
        roc_auc = float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else 0.5
        pr_auc = (
            float(average_precision_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else 0.0
        )
        youden_j = _max_youden_j(val_y, val_probs)
        composite = 0.5 * roc_auc + 0.5 * pr_auc
        train_loss = running_loss / max(1, n_batches)

        rec = {
            "epoch": float(epoch),
            "train_loss": round(train_loss, 5),
            "val_roc_auc": round(roc_auc, 5),
            "val_pr_auc": round(pr_auc, 5),
            "val_max_youden_j": round(youden_j, 5),
        }
        history.append(rec)
        print(
            f"[{fold_name}] epoch {epoch:02d}/{cfg.optim.epochs:02d} "
            f"loss={train_loss:.4f} roc_auc={roc_auc:.4f} pr_auc={pr_auc:.4f} youden_j={youden_j:.4f}",
            flush=True,
        )

        if composite > best_score:
            best_score = composite
            best_epoch = epoch
            best_roc = roc_auc
            best_pr = pr_auc
            best_j = youden_j
            best_probs = val_probs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                cfg.optim.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.optim.early_stopping_patience
            ):
                print(
                    f"[{fold_name}] Early stopping at epoch {epoch} (best={best_epoch})",
                    flush=True,
                )
                break

    return FoldTrainOutput(
        fold_name=fold_name,
        best_epoch=best_epoch,
        best_val_roc_auc=best_roc,
        best_val_pr_auc=best_pr,
        best_val_youden_j=best_j,
        val_probs=best_probs,
        history=history,
        best_state_dict=best_state,
    )
