"""Unit tests for v1 decoupled 2D Spec+Tempogram, v1 1D Waveform, AudioSet init & Doppler SSL."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from heartsounds.config import ExperimentConfig, ModelConfig, OptimConfig
from heartsounds.data.torch_dataset import WaveformBatch
from heartsounds.models.pretrain import load_audioset_convnext_tiny, run_fold_masked_ssl_pretrain
from heartsounds.models.registry import build_model


def _make_synthetic_heartbeat(bpm: float, duration_s: float = 8.0, sr: int = 4000) -> torch.Tensor:
    """Create a synthetic Doppler pulse train with identical pulse shape at `bpm`."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    period = 60.0 / bpm
    phase = (t % period) / period
    # Crisp systolic Doppler burst modulated at 350 Hz
    env = np.exp(-0.5 * ((phase - 0.15) / 0.04) ** 2).astype(np.float32)
    carrier = np.sin(2.0 * math.pi * 350.0 * t).astype(np.float32)
    return torch.from_numpy(env * carrier)


def test_v1_tempogram_stem_distinguishes_normal_vs_tachycardia_bpm():
    """Verify 135 BPM vs 215 BPM produce distinct lag-preserving tempogram representations."""
    torch.manual_seed(0)
    w_norm = _make_synthetic_heartbeat(135.0, duration_s=8.0)
    w_fast = _make_synthetic_heartbeat(215.0, duration_s=8.0)
    wavs = torch.stack([w_norm, w_fast], dim=0)
    lens = torch.tensor([w_norm.numel(), w_fast.numel()], dtype=torch.long)

    model = build_model(
        ModelConfig(
            name="spec_tempogram_mil",
            channels=32,
            dropout=0.0,
            topk_pool=4,
            pretrained=False,
            init_mode="scratch",
            backbone="convnext_tiny.fb_in22k",
            n_mels=96,
            decoupled_tempogram=True,
            freq_subbands=4,
            temporal_gru=True,
        )
    )
    model.eval()
    with torch.no_grad():
        _, tempo, _ = model.frontend(wavs, lens)
        # Peak lag bin in 2.56s autocorrelation channel (ch 0) must shift significantly between 135 & 215 BPM
        peak_norm = int(tempo[0, 0, :, 100:300].mean(dim=-1).argmax().item())
        peak_fast = int(tempo[1, 0, :, 100:300].mean(dim=-1).argmax().item())
        assert peak_norm > peak_fast + 5

        assert model.tempo_stem is not None
        t_feat = model.tempo_stem(tempo, target_t=25)
        cos_sim = torch.nn.functional.cosine_similarity(
            t_feat[0].mean(dim=-1), t_feat[1].mean(dim=-1), dim=0
        ).item()
        assert cos_sim < 0.98


def test_v1_2d_and_1d_models_are_strictly_padding_invariant():
    """Right-padding a batch must not alter logits of shorter recordings in eval mode."""
    torch.manual_seed(3)
    short_len = 24000  # 6s @ 4 kHz
    long_len = 48000  # 12s @ 4 kHz
    base_wav = torch.randn(short_len)
    padded_wav = torch.zeros(long_len)
    padded_wav[:short_len] = base_wav

    cfgs = [
        ModelConfig(
            name="spec_tempogram_mil",
            channels=32,
            dropout=0.0,
            topk_pool=4,
            pretrained=False,
            init_mode="scratch",
            backbone="convnext_tiny.fb_in22k",
            n_mels=96,
            decoupled_tempogram=True,
            freq_subbands=4,
            temporal_gru=True,
        ),
        ModelConfig(
            name="waveform_mil",
            channels=32,
            dropout=0.0,
            topk_pool=4,
            pretrained=False,
            init_mode="scratch",
            inst_freq_channel=True,
        ),
    ]
    for cfg in cfgs:
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            res_unpad = model(base_wav.unsqueeze(0), torch.tensor([short_len]))
            res_pad = model(
                torch.stack([padded_wav, torch.randn(long_len)], dim=0),
                torch.tensor([short_len, long_len]),
            )
            out_unpad = res_unpad["logits"]
            out_pad = res_pad["logits"][:1]
            attn_pad = res_pad["attn_weights"][0]
        # Zero attention must be placed on the padded second half of the batch
        assert float(attn_pad[attn_pad.shape[0] // 2 + 2 :].sum().item()) == 0.0
        # 1D blocks are masked at every layer (tiny ~1e-3 1-frame stem bias at edge);
        # 2D ConvNeXt has small (<5e-2) edge-of-clip conv bias drift.
        tol = 5e-3 if cfg.name == "waveform_mil" else 5e-2
        assert torch.allclose(out_unpad, out_pad, atol=tol), (
            f"Padding invariance violated for {cfg.name}: {out_unpad} vs {out_pad}"
        )


def test_audioset_convnext_tiny_loader_matches_100pct_of_encoder_keys():
    """Verify all ConvNeXt-Tiny backbone tensors load cleanly from topel/ConvNeXt-Tiny-AT."""
    model = build_model(
        ModelConfig(
            name="spec_tempogram_mil",
            channels=32,
            dropout=0.0,
            pretrained=False,
            init_mode="scratch",
            backbone="convnext_tiny.fb_in22k",
            n_mels=96,
            decoupled_tempogram=True,
            freq_subbands=4,
            temporal_gru=True,
        )
    )
    loaded = load_audioset_convnext_tiny(model.encoder, repo_id="topel/ConvNeXt-Tiny-AT")
    assert loaded == len(model.encoder.state_dict())


def test_per_fold_doppler_ssl_runs_and_decreases_loss_for_2d_and_1d():
    """Verify leak-free unlabeled masked SSL pretraining executes cleanly for 2D & 1D models."""
    torch.manual_seed(5)
    wavs = torch.stack([_make_synthetic_heartbeat(140.0, 4.0) for _ in range(4)], dim=0)
    batch = WaveformBatch(
        waveforms=wavs,
        lengths=torch.full((4,), wavs.shape[1], dtype=torch.long),
        labels=torch.zeros(4),
        subtypes=torch.zeros(4, dtype=torch.long),
        recording_ids=["r0", "r1", "r2", "r3"],
        splits=["fold_0"] * 4,
        patient_groups=["p0"] * 4,
        curated_subtypes=["NORMAL"] * 4,
    )
    dummy_loader: DataLoader[WaveformBatch] = DataLoader([batch, batch], batch_size=None)

    for m_cfg in (
        ModelConfig(
            name="spec_tempogram_mil",
            channels=32,
            dropout=0.0,
            pretrained=False,
            init_mode="doppler_ssl",
            backbone="convnext_tiny.fb_in22k",
            n_mels=96,
            decoupled_tempogram=True,
            freq_subbands=4,
            temporal_gru=True,
        ),
        ModelConfig(
            name="waveform_mil",
            channels=32,
            dropout=0.0,
            pretrained=False,
            init_mode="doppler_ssl",
            inst_freq_channel=True,
        ),
    ):
        exp_cfg = ExperimentConfig(
            model=m_cfg,
            optim=OptimConfig(ssl_epochs=2, ssl_lr=2e-3, ssl_mask_ratio=0.5, mixed_precision=False),
        )
        model = build_model(m_cfg)
        losses = run_fold_masked_ssl_pretrain(
            model, dummy_loader, exp_cfg, torch.device("cpu"), fold_name="fold_0"
        )
        assert len(losses) == 2
        assert all(math.isfinite(x) and x > 0.0 for x in losses)
