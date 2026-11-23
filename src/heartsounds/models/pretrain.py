"""AudioSet pretrained checkpoint loader and leak-free per-fold Doppler SSL pretraining."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from heartsounds.config import ExperimentConfig
from heartsounds.data.torch_dataset import WaveformBatch
from heartsounds.models.pooling import make_length_mask


def resolve_init_mode(init_mode: str, pretrained: bool) -> str:
    """Normalize `ModelConfig.init_mode` while staying backwards-compatible with `pretrained`."""
    mode = init_mode.strip().lower()
    if mode in ("", "auto"):
        return "imagenet" if pretrained else "scratch"
    valid = {"scratch", "imagenet", "audioset", "doppler_ssl", "audioset_doppler_ssl"}
    if mode not in valid:
        raise ValueError(f"Unsupported init_mode={init_mode!r}; expected one of {sorted(valid)}")
    return mode


def _map_audioset_convnext_key(src_key: str) -> str | None:
    """Translate Facebook/Pellegrini `ConvNeXt-Tiny-AT` keys to `timm` `features_only` keys."""
    if src_key.startswith("downsample_layers.0.0."):
        return "stem_0." + src_key.split("downsample_layers.0.0.", 1)[1]
    if src_key.startswith("downsample_layers.0.1."):
        return "stem_1." + src_key.split("downsample_layers.0.1.", 1)[1]

    if src_key.startswith("downsample_layers."):
        parts = src_key.split(".")
        stage_idx = parts[1]
        sub_idx = parts[2]
        param = parts[3]
        return f"stages_{stage_idx}.downsample.{sub_idx}.{param}"

    if src_key.startswith("stages."):
        parts = src_key.split(".")
        stage_idx, block_idx, layer = parts[1], parts[2], parts[3]
        if layer == "gamma":
            return f"stages_{stage_idx}.blocks.{block_idx}.gamma"
        param = parts[4]
        layer_map = {
            "dwconv": "conv_dw",
            "norm": "norm",
            "pwconv1": "mlp.fc1",
            "pwconv2": "mlp.fc2",
        }
        mapped = layer_map.get(layer)
        if mapped is None:
            return None
        return f"stages_{stage_idx}.blocks.{block_idx}.{mapped}.{param}"

    return None


def load_audioset_convnext_tiny(
    encoder: nn.Module,
    *,
    repo_id: str = "topel/ConvNeXt-Tiny-AT",
    filename: str = "model.safetensors",
) -> int:
    """Load AudioSet-2M ConvNeXt-Tiny spectrogram weights into a `timm` `features_only` encoder.

    Repeats the 1-channel AudioSet spectrogram stem kernel `(96, 1, 4, 4)` across `in_chans`
    scaled by `1 / in_chans` to preserve initial activation variance.
    """
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    src_state = load_file(ckpt_path, device="cpu")
    dst_state = encoder.state_dict()

    mapped_state: dict[str, torch.Tensor] = {}
    for src_k, tensor in src_state.items():
        dst_k = _map_audioset_convnext_key(src_k)
        if dst_k is None or dst_k not in dst_state:
            continue
        target_shape = dst_state[dst_k].shape
        if dst_k == "stem_0.weight" and tensor.ndim == 4 and tensor.shape[1] == 1:
            in_chans = target_shape[1]
            if in_chans != 1:
                tensor = tensor.repeat(1, in_chans, 1, 1) / float(in_chans)
        if tensor.shape != target_shape:
            raise ValueError(
                f"Shape mismatch loading AudioSet key {src_k} -> {dst_k}: "
                f"got {tuple(tensor.shape)}, expected {tuple(target_shape)}"
            )
        mapped_state[dst_k] = tensor

    missing = set(dst_state.keys()) - set(mapped_state.keys())
    if missing:
        raise RuntimeError(
            f"AudioSet checkpoint {repo_id} did not cover all encoder keys; "
            f"missing {len(missing)} keys (sample: {sorted(missing)[:5]})"
        )

    encoder.load_state_dict(mapped_state, strict=True)
    return len(mapped_state)


def _sample_2d_block_mask(
    batch_size: int,
    height: int,
    width: int,
    valid_lengths: torch.Tensor,
    *,
    mask_ratio: float,
    patch_h: int = 8,
    patch_w: int = 8,
    device: torch.device,
) -> torch.Tensor:
    """Return boolean mask `(B, 1, H, W)` where True marks masked valid patches."""
    gh = max(1, math.ceil(height / patch_h))
    gw = max(1, math.ceil(width / patch_w))
    grid_mask = torch.rand((batch_size, 1, gh, gw), device=device) < mask_ratio
    full_mask = F.interpolate(grid_mask.float(), size=(height, width), mode="nearest") > 0.5
    valid_t = make_length_mask(valid_lengths, width).to(device).unsqueeze(1).unsqueeze(2)
    return full_mask & valid_t


def _pretrain_2d_masked_spec(
    model: nn.Module,
    loader: DataLoader[WaveformBatch],
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    fold_name: str,
) -> list[float]:
    """Per-fold Masked Spectrogram Autoencoder pretraining on unlabeled training waveforms."""
    feat_channels = model.encoder.feature_info.channels()
    decoder = nn.Sequential(
        nn.Conv2d(feat_channels[-1], 256, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(128, 3, kernel_size=1),
    ).to(device)

    params = list(model.encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.ssl_lr, weight_decay=cfg.optim.weight_decay)
    total_steps = max(1, cfg.optim.ssl_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    amp_enabled = cfg.optim.mixed_precision and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_enabled and not use_bf16))

    # Disable SpecAugment inside frontend during clean-target SSL reconstruction
    orig_spec_aug = getattr(model.frontend, "spec_augment", False)
    if hasattr(model.frontend, "spec_augment"):
        model.frontend.spec_augment = False

    losses: list[float] = []
    try:
        model.encoder.train()
        decoder.train()
        for epoch in range(1, cfg.optim.ssl_epochs + 1):
            running = 0.0
            n_batches = 0
            for batch in loader:
                wav = batch.waveforms.to(device, non_blocking=True)
                lens = batch.lengths.to(device, non_blocking=True)

                with torch.no_grad():
                    target_spec, spec_lens = model.extract_clean_spectrogram(wav, lens)
                    t_in = target_spec.shape[-1]
                    if t_in < 64:
                        target_spec = F.pad(target_spec, (0, 64 - t_in))
                    mask = _sample_2d_block_mask(
                        target_spec.shape[0],
                        target_spec.shape[2],
                        target_spec.shape[3],
                        spec_lens,
                        mask_ratio=cfg.optim.ssl_mask_ratio,
                        device=device,
                    )
                    masked_spec = target_spec.masked_fill(mask, 0.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    feats = model.encoder(masked_spec)
                    pred = F.interpolate(
                        decoder(feats[-1]),
                        size=(target_spec.shape[2], target_spec.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask_3c = mask.expand_as(target_spec).float()
                    denom = mask_3c.sum().clamp(min=1.0)
                    diff = (pred.float() - target_spec.float()) * mask_3c
                    loss = (diff.pow(2).sum() + 0.5 * diff.abs().sum()) / denom

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                    optimizer.step()
                scheduler.step()
                running += float(loss.item())
                n_batches += 1

            ep_loss = running / max(1, n_batches)
            losses.append(round(ep_loss, 5))
            print(
                f"[{fold_name}] SSL-2D epoch {epoch:02d}/{cfg.optim.ssl_epochs:02d} recon_loss={ep_loss:.4f}",
                flush=True,
            )
    finally:
        if hasattr(model.frontend, "spec_augment"):
            model.frontend.spec_augment = orig_spec_aug
    return losses


def _pretrain_1d_masked_envelope(
    model: nn.Module,
    loader: DataLoader[WaveformBatch],
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    fold_name: str,
) -> list[float]:
    """Per-fold Masked Multi-Band Envelope pretraining for 1D `WaveformMILNet`."""
    ch = int(cfg.model.channels)
    n_env_ch = model.env_stem.num_env_channels
    decoder = nn.Sequential(
        nn.Conv1d(ch, ch, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv1d(ch, n_env_ch, kernel_size=1),
    ).to(device)

    backbone_modules = [
        model.wav_stem,
        model.env_stem,
        model.fuse,
        model.blocks,
        model.rhythm_proj,
        model.gru,
        model.post_norm,
    ]
    params: list[nn.Parameter] = list(decoder.parameters())
    for mod in backbone_modules:
        params.extend(mod.parameters())

    optimizer = torch.optim.AdamW(params, lr=cfg.optim.ssl_lr, weight_decay=cfg.optim.weight_decay)
    total_steps = max(1, cfg.optim.ssl_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    amp_enabled = cfg.optim.mixed_precision and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_enabled and not use_bf16))

    losses: list[float] = []
    model.train()
    decoder.train()
    for epoch in range(1, cfg.optim.ssl_epochs + 1):
        running = 0.0
        n_batches = 0
        for batch in loader:
            wav = batch.waveforms.to(device, non_blocking=True)
            lens = batch.lengths.to(device, non_blocking=True)

            with torch.no_grad():
                target_env, frame_lens = model.extract_target_envelope(wav, lens)
                t_frames = target_env.shape[-1]
                # Mask contiguous ~160 ms spans (40 frames @ 250 Hz)
                patch_w = 40
                gw = max(1, math.ceil(t_frames / patch_w))
                grid = torch.rand((wav.shape[0], 1, gw), device=device) < cfg.optim.ssl_mask_ratio
                span_mask = F.interpolate(grid.float(), size=t_frames, mode="nearest") > 0.5
                valid_f = make_length_mask(frame_lens, t_frames).to(device).unsqueeze(1)
                frame_mask_bool = span_mask & valid_f
                sample_mask_bool = (
                    F.interpolate(
                        frame_mask_bool.float(), size=wav.shape[1], mode="nearest"
                    ).squeeze(1)
                    > 0.5
                )
                masked_wav = wav.masked_fill(sample_mask_bool, 0.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                rhythm_seq, _ = model.encode_rhythm_sequence(masked_wav, lens)
                pred_env = F.interpolate(
                    decoder(rhythm_seq),
                    size=target_env.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
                mask_f = frame_mask_bool.expand_as(target_env).float()
                denom = mask_f.sum().clamp(min=1.0)
                diff = (pred_env.float() - target_env.float()) * mask_f
                loss = (diff.pow(2).sum() + 0.5 * diff.abs().sum()) / denom

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                optimizer.step()
            scheduler.step()
            running += float(loss.item())
            n_batches += 1

        ep_loss = running / max(1, n_batches)
        losses.append(round(ep_loss, 5))
        print(
            f"[{fold_name}] SSL-1D epoch {epoch:02d}/{cfg.optim.ssl_epochs:02d} recon_loss={ep_loss:.4f}",
            flush=True,
        )
    return losses


def run_fold_masked_ssl_pretrain(
    model: nn.Module,
    loader: DataLoader[WaveformBatch],
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    fold_name: str,
) -> list[float]:
    """Run leak-free per-fold unlabeled masked reconstruction pretraining on `trn_sub`."""
    if cfg.optim.ssl_epochs <= 0:
        return []
    if hasattr(model, "extract_clean_spectrogram") and hasattr(model, "encoder"):
        return _pretrain_2d_masked_spec(model, loader, cfg, device, fold_name=fold_name)
    if hasattr(model, "extract_target_envelope") and hasattr(model, "encode_rhythm_sequence"):
        return _pretrain_1d_masked_envelope(model, loader, cfg, device, fold_name=fold_name)
    raise TypeError(f"Model {type(model).__name__} does not support masked SSL pretraining")
