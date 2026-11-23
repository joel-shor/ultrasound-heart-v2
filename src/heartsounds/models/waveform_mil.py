"""Model Family 2: 1D Multi-Scale Waveform + Envelope Dilated ConvNeXt-BiGRU MIL Network.

Processes 4,000 Hz fetal Doppler waveforms through two complementary 250 Hz branches:
1. Learned strided Conv1D waveform stem capturing beat/valve morphology.
2. Multi-band (low/mid/high) + log-energy (+ optional instantaneous carrier frequency in `v1`)
   envelope branch capturing beat energy and Doppler velocity shift.
Fused features pass through dilated 1D ConvNeXt blocks, an 8x Max+Mean rhythm pool
(`250 Hz -> 31.25 Hz`, ~14 frames/heartbeat), a packed 2-layer BiGRU for beat-to-beat
sequential rhythm state, and `MaskedGatedAttentionMILPool`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from heartsounds.config import ModelConfig
from heartsounds.models.pooling import MaskedGatedAttentionMILPool, make_length_mask
from heartsounds.schema import SUBTYPE_TO_INDEX


class ChannelLayerNorm1D(nn.Module):
    """Per-timestep channel LayerNorm on `(B, C, T)` — invariant to sequence padding length."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MultiBandEnvelopeFrontend1D(nn.Module):
    """Fast, differentiable on-GPU multi-scale energy (+ instantaneous frequency) extractor."""

    def __init__(self, out_channels: int, *, inst_freq_channel: bool = False) -> None:
        super().__init__()
        self.inst_freq_channel = bool(inst_freq_channel)
        self.num_env_channels = 5 if self.inst_freq_channel else 4
        self.proj = nn.Sequential(
            nn.Conv1d(self.num_env_channels, out_channels, kernel_size=5, padding=2),
            ChannelLayerNorm1D(out_channels),
            nn.GELU(),
        )

    def extract_raw_envelope(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute `(B, num_env_channels, T_250hz)` strictly non-negative envelope features."""
        x = wav.unsqueeze(1)  # (B, 1, T_samples)
        lp_150 = F.avg_pool1d(x, kernel_size=27, stride=1, padding=13)
        lp_600 = F.avg_pool1d(x, kernel_size=7, stride=1, padding=3)
        band_low = lp_150
        band_mid = lp_600 - lp_150
        band_high = x - lp_600

        sq = x.clamp(-12.0, 12.0).pow(2)
        log_energy = torch.log1p(4.0 * sq)

        channels = [band_low.abs(), band_mid.abs(), band_high.abs(), log_energy]
        stacked = torch.cat(channels, dim=1)  # (B, 4, T_samples)
        pooled = F.avg_pool1d(stacked, kernel_size=32, stride=16, padding=8).clamp(min=0.0)
        env_250 = torch.log1p(pooled)

        if self.inst_freq_channel:
            dx = torch.zeros_like(x)
            if x.shape[-1] > 1:
                dx[..., 1:] = (x[..., 1:] - x[..., :-1]).abs()
            dx_pool = F.avg_pool1d(dx, kernel_size=32, stride=16, padding=8).clamp(min=0.0)
            abs_pool = F.avg_pool1d(x.abs(), kernel_size=32, stride=16, padding=8).clamp(min=0.0)
            # Instantaneous carrier frequency proxy: high for valve clicks / fast Doppler velocity
            inst_freq = torch.log1p(dx_pool / (abs_pool + 1e-3)).clamp(max=4.0)
            env_250 = torch.cat([env_250, inst_freq], dim=1)
        return env_250

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.proj(self.extract_raw_envelope(wav))


class DilatedConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt block with dilated depthwise conv and strict mask zeroing."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, channels * 3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pw2 = nn.Linear(channels * 3, channels)
        self.gamma = nn.Parameter(0.1 * torch.ones(channels))

    def forward(self, x: torch.Tensor, mask_b1t: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.dwconv(x * mask_b1t)
        h_bt = h.transpose(1, 2)  # (B, T, C)
        h_bt = self.pw2(self.drop(self.act(self.pw1(self.norm(h_bt)))))
        h = (h_bt * self.gamma).transpose(1, 2)
        return (residual + h) * mask_b1t


class WaveformMILNet(nn.Module):
    """End-to-end 1D waveform + multi-band envelope MIL classifier.

    Extracts fine morphology at 250 Hz (4 ms) via dilated ConvNeXt blocks, then pools 8x
    (Max + Mean) to 31.25 Hz (32 ms, ~14 frames/beat) before the 2-layer BiGRU and MIL head
    so sequential recurrence spans 8-15 consecutive cardiac cycles efficiently.
    """

    STEM_STRIDE = 16
    RHYTHM_STRIDE = 8

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ch = int(cfg.channels)
        half = ch // 2

        self.wav_stem = nn.Sequential(
            nn.Conv1d(1, half, kernel_size=15, stride=4, padding=7),
            ChannelLayerNorm1D(half),
            nn.GELU(),
            nn.Conv1d(half, half, kernel_size=9, stride=4, padding=4),
            ChannelLayerNorm1D(half),
            nn.GELU(),
        )
        self.env_stem = MultiBandEnvelopeFrontend1D(
            ch - half, inst_freq_channel=cfg.inst_freq_channel
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1),
            ChannelLayerNorm1D(ch),
            nn.GELU(),
        )

        dilations = (1, 2, 4, 8, 16, 32) if cfg.inst_freq_channel else (1, 2, 4, 8, 16, 4)
        self.blocks = nn.ModuleList(
            [
                DilatedConvNeXtBlock1D(ch, kernel_size=9, dilation=d, dropout=cfg.dropout)
                for d in dilations
            ]
        )
        self.rhythm_proj = nn.Sequential(
            nn.Conv1d(ch * 2, ch, kernel_size=3, padding=1),
            ChannelLayerNorm1D(ch),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=ch,
            hidden_size=ch // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0.0,
        )
        self.post_norm = ChannelLayerNorm1D(ch)
        self.pool = MaskedGatedAttentionMILPool(
            in_dim=ch,
            attn_dim=ch,
            topk=cfg.topk_pool,
            dropout=cfg.dropout * 0.5,
        )
        pooled_dim = ch * self.pool.out_dim_multiplier
        self.head_norm = nn.LayerNorm(pooled_dim)
        self.head_drop = nn.Dropout(cfg.dropout)
        self.binary_head = nn.Linear(pooled_dim, 1)
        self.subtype_head = nn.Linear(pooled_dim, len(SUBTYPE_TO_INDEX))

    def extract_target_envelope(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(env_250hz, frame_lengths)` for masked 1D SSL pretraining."""
        sample_mask = make_length_mask(lengths, waveforms.shape[1]).to(waveforms.dtype)
        clean_wav = waveforms * sample_mask
        env_250 = self.env_stem.extract_raw_envelope(clean_wav)
        t_frames = env_250.shape[-1]
        frame_lengths = torch.clamp(
            (lengths + self.STEM_STRIDE - 1) // self.STEM_STRIDE,
            min=1,
            max=t_frames,
        )
        frame_mask = make_length_mask(frame_lengths, t_frames).to(env_250.dtype).unsqueeze(1)
        return env_250 * frame_mask, frame_lengths

    def encode_rhythm_sequence(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw waveforms to `(B, C, T_rhythm)` at 31.25 Hz and `rhythm_lengths`."""
        sample_mask = make_length_mask(lengths, waveforms.shape[1]).to(waveforms.dtype)
        clean_wav = waveforms * sample_mask

        w_feat = self.wav_stem(clean_wav.unsqueeze(1))
        e_feat = self.env_stem(clean_wav)
        t_frames = min(w_feat.shape[-1], e_feat.shape[-1])
        x = self.fuse(torch.cat([w_feat[..., :t_frames], e_feat[..., :t_frames]], dim=1))

        frame_lengths = torch.clamp(
            (lengths + self.STEM_STRIDE - 1) // self.STEM_STRIDE,
            min=1,
            max=t_frames,
        )
        frame_mask = make_length_mask(frame_lengths, t_frames).to(x.dtype).unsqueeze(1)
        x = x * frame_mask

        for blk in self.blocks:
            x = blk(x, frame_mask)

        if x.shape[-1] < self.RHYTHM_STRIDE:
            pad_r = self.RHYTHM_STRIDE - x.shape[-1]
            x = F.pad(x, (0, pad_r))
            frame_mask = F.pad(frame_mask, (0, pad_r))
        x_avg = F.avg_pool1d(
            x, kernel_size=self.RHYTHM_STRIDE, stride=self.RHYTHM_STRIDE, ceil_mode=True
        )
        neg_fill = torch.where(frame_mask > 0.5, x, torch.full_like(x, -1e4))
        x_max = F.max_pool1d(
            neg_fill, kernel_size=self.RHYTHM_STRIDE, stride=self.RHYTHM_STRIDE, ceil_mode=True
        )
        t_rhythm = min(x_avg.shape[-1], x_max.shape[-1])
        rhythm_lengths = torch.clamp(
            (frame_lengths + self.RHYTHM_STRIDE - 1) // self.RHYTHM_STRIDE,
            min=1,
            max=t_rhythm,
        )
        rhythm_mask = make_length_mask(rhythm_lengths, t_rhythm).to(x.dtype).unsqueeze(1)
        x_avg_clean = x_avg[..., :t_rhythm] * rhythm_mask
        x_max_clean = x_max[..., :t_rhythm] * rhythm_mask
        r = self.rhythm_proj(torch.cat([x_avg_clean, x_max_clean], dim=1)) * rhythm_mask

        packed = pack_padded_sequence(
            r.transpose(1, 2),
            rhythm_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        gru_out_packed, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True, total_length=t_rhythm)
        r = self.post_norm(r + gru_out.transpose(1, 2)) * rhythm_mask
        return r, rhythm_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        r, rhythm_lengths = self.encode_rhythm_sequence(waveforms, lengths)
        pooled, attn_weights = self.pool(r, rhythm_lengths)
        h = self.head_drop(self.head_norm(pooled))
        return {
            "logits": self.binary_head(h).squeeze(-1),
            "subtype_logits": self.subtype_head(h),
            "attn_weights": attn_weights,
        }
