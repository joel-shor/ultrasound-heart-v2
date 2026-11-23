"""Model Family 3: 2D Multi-Resolution Log-Mel + Tempogram ConvNeXt MIL Network.

Constructs a 3-channel 2D time-frequency & local-periodicity tensor on GPU:
- Ch 0: Fine-time Log-Mel spectrogram (16 ms window, sharp Doppler valve clicks)
- Ch 1: Fine-frequency Log-Mel spectrogram (64 ms window, blood-flow harmonics)
- Ch 2: Sliding-window Envelope Autocorrelationgram / Tempogram (64 lag bins across
        0.24s - 1.50s), making rhythm breaks, PACs, and AV-block period shifts
        explicit in 2D image space.
Extracts multi-stage 2D ConvNeXt features, pools across frequency while preserving
temporal frames, and aggregates via `MaskedGatedAttentionMILPool`.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from heartsounds.config import ModelConfig
from heartsounds.models.pooling import MaskedGatedAttentionMILPool, make_length_mask
from heartsounds.schema import SUBTYPE_TO_INDEX


class OnGPUSpecTempogramFrontend(nn.Module):
    """Computes aligned `(B, 3, 64, T_frames)` Log-Mel + Tempogram representations at 50 Hz."""

    SAMPLE_RATE = 4000
    HOP_LENGTH = 80  # 20 ms -> 50 Hz frame rate
    N_MELS = 64
    TEMPO_WIN = 128  # 2.56s sliding window at 50 Hz
    LAG_START = 12  # 0.24s (250 BPM)
    LAG_COUNT = 64  # 12..75 frames -> 0.24s..1.50s (40 - 250 BPM)

    def __init__(self, spec_augment: bool = True) -> None:
        super().__init__()
        self.spec_augment = spec_augment
        self.mel_short = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=128,
            win_length=64,
            hop_length=self.HOP_LENGTH,
            f_min=25.0,
            f_max=1950.0,
            n_mels=self.N_MELS,
            power=2.0,
            center=True,
        )
        self.mel_long = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=512,
            win_length=256,
            hop_length=self.HOP_LENGTH,
            f_min=25.0,
            f_max=1950.0,
            n_mels=self.N_MELS,
            power=2.0,
            center=True,
        )

    def _compute_tempogram(self, mel_short_log: torch.Tensor) -> torch.Tensor:
        """Vectorized FFT sliding-window autocorrelation over broadband log-mel energy."""
        # Sum mel bands in primary heartbeat flow range -> (B, T)
        env = mel_short_log[:, 4:48, :].mean(dim=1)  # (B, T)
        bsz, t_len = env.shape
        pad_left = self.TEMPO_WIN // 2
        pad_right = self.TEMPO_WIN - pad_left - 1
        padded = F.pad(env.unsqueeze(1), (pad_left, pad_right), mode="replicate").squeeze(1)
        windows = padded.unfold(dimension=1, size=self.TEMPO_WIN, step=1)  # (B, T, W)
        centered = windows - windows.mean(dim=-1, keepdim=True)
        energy = (centered * centered).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        spec = torch.fft.rfft(centered, n=256, dim=-1)
        ac = torch.fft.irfft(spec * torch.conj(spec), n=256, dim=-1)[..., : self.TEMPO_WIN]
        ac_norm = (ac / energy)[..., self.LAG_START : self.LAG_START + self.LAG_COUNT]  # (B, T, 64)
        return ac_norm.transpose(1, 2).contiguous()  # (B, 64, T)

    @staticmethod
    def _masked_instance_norm(x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T), frame_mask: (B, T)
        mask_4d = frame_mask.unsqueeze(1).unsqueeze(2).to(x.dtype)  # (B, 1, 1, T)
        count = (mask_4d.sum(dim=(-2, -1), keepdim=True) * x.shape[2]).clamp(min=1.0)
        x_masked = x * mask_4d
        mean = x_masked.sum(dim=(-2, -1), keepdim=True) / count
        var = (((x - mean) * mask_4d) ** 2).sum(dim=(-2, -1), keepdim=True) / count
        return ((x - mean) / torch.sqrt(var + 1e-5)) * mask_4d

    def _apply_spec_augment(self, x: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        bsz, _, f_bins, t_bins = x.shape
        out = x.clone()
        # Mask 1-2 frequency/lag strips (width <= 8) and 1-2 time strips (width <= 12 frames = 0.24s)
        for b in range(bsz):
            if torch.rand((), device=x.device).item() < 0.65:
                fw = int(torch.randint(2, 9, (1,), device=x.device).item())
                f0 = int(torch.randint(0, max(1, f_bins - fw), (1,), device=x.device).item())
                out[b, :2, f0 : f0 + fw, :] = 0.0
            vlen = int(frame_lengths[b].item())
            if vlen > 20 and torch.rand((), device=x.device).item() < 0.65:
                tw = int(torch.randint(3, min(14, vlen // 4 + 4), (1,), device=x.device).item())
                t0 = int(torch.randint(0, max(1, vlen - tw), (1,), device=x.device).item())
                out[b, :, :, t0 : t0 + tw] = 0.0
        return out

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute spectrograms in float32 (cuFFT prefers fp32 over bf16 for non-power-of-2 sizes)
        with torch.amp.autocast(device_type=waveforms.device.type, enabled=False):
            wav32 = waveforms.float()
            sample_mask = make_length_mask(lengths, wav32.shape[1]).to(wav32.dtype)
            wav32 = wav32 * sample_mask

            m_short = torch.log(self.mel_short(wav32).clamp(min=1e-7))
            m_long = torch.log(self.mel_long(wav32).clamp(min=1e-7))
            t_frames = min(m_short.shape[-1], m_long.shape[-1])
            m_short = m_short[..., :t_frames]
            m_long = m_long[..., :t_frames]
            tempo = self._compute_tempogram(m_short)

            frame_lengths = torch.clamp(
                lengths // self.HOP_LENGTH + 1,
                min=1,
                max=t_frames,
            )
            frame_mask = make_length_mask(frame_lengths, t_frames)
            stacked = torch.stack([m_short, m_long, tempo], dim=1)  # (B, 3, 64, T)
            normed = self._masked_instance_norm(stacked, frame_mask)
            if self.training and self.spec_augment:
                normed = self._apply_spec_augment(normed, frame_lengths)

        return normed, frame_lengths


class SpecTempogramMILNet(nn.Module):
    """2D ConvNeXt / ResNet backbone over Log-Mel + Tempogram with Temporal Attention MIL."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.frontend = OnGPUSpecTempogramFrontend(spec_augment=True)
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(2, 3),
        )
        feat_channels = self.encoder.feature_info.channels()
        self.stage_stride = int(self.encoder.feature_info.reduction()[0])
        # Mean + Max over frequency axis on both selected stages (e.g. stride 16/32 or 8/16)
        fused_in = 2 * (feat_channels[0] + feat_channels[1])
        proj_dim = int(cfg.channels) * 2

        self.temporal_neck = nn.Sequential(
            nn.Conv1d(fused_in, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = MaskedGatedAttentionMILPool(
            in_dim=proj_dim,
            attn_dim=cfg.channels,
            topk=max(2, cfg.topk_pool // 2),
            dropout=cfg.dropout * 0.5,
        )
        pooled_dim = proj_dim * self.pool.out_dim_multiplier
        self.head_norm = nn.LayerNorm(pooled_dim)
        self.head_drop = nn.Dropout(cfg.dropout)
        self.binary_head = nn.Linear(pooled_dim, 1)
        self.subtype_head = nn.Linear(pooled_dim, len(SUBTYPE_TO_INDEX))

    @staticmethod
    def _freq_pool(fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, C, F', T') -> (B, 2*C, T')
        return torch.cat([fmap.mean(dim=2), fmap.amax(dim=2)], dim=1)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        img, spec_lengths = self.frontend(waveforms, lengths)
        # Ensure minimum time width >= 64 frames so stride-32 stages never collapse to 0
        t_in = img.shape[-1]
        if t_in < 64:
            img = F.pad(img, (0, 64 - t_in))

        stage3, stage4 = self.encoder(img)
        s3_t = self._freq_pool(stage3)  # (B, 2*C3, T3) at stride 16
        s4_t = F.interpolate(
            self._freq_pool(stage4),
            size=s3_t.shape[-1],
            mode="linear",
            align_corners=False,
        )
        t3_len = s3_t.shape[-1]
        out_lengths = torch.clamp(
            (spec_lengths + self.stage_stride - 1) // self.stage_stride,
            min=1,
            max=t3_len,
        )
        mask_b1t = make_length_mask(out_lengths, t3_len).to(s3_t.dtype).unsqueeze(1)

        seq = self.temporal_neck(torch.cat([s3_t, s4_t], dim=1) * mask_b1t) * mask_b1t
        pooled, attn_weights = self.pool(seq, out_lengths)
        h = self.head_drop(self.head_norm(pooled))
        return {
            "logits": self.binary_head(h).squeeze(-1),
            "subtype_logits": self.subtype_head(h),
            "attn_weights": attn_weights,
        }
