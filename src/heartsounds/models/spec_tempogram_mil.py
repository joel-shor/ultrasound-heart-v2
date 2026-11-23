"""Model Family 3: 2D Multi-Resolution Log-Mel + Tempogram MIL Networks (`v0` & `v1`).

Supports both:
- `v0` mode (`decoupled_tempogram=False`): 3-channel `(Mel_short, Mel_long, Tempogram)` image
  with global vertical mean/max readout and Conv1D temporal neck.
- `v1` mode (`decoupled_tempogram=True`):
  1. Dedicated 3-channel Doppler acoustic spectrogram `(B, 3, F=96, T @ 50Hz)`:
     fine-time Log-Mel (`16 ms`, `n_fft=256`), fine-frequency Log-Mel (`64 ms`, `n_fft=512`),
     and temporal spectral-flux delta `d/dt Log-Mel`.
  2. Dedicated dual-window Tempogram branch `(B, 3, L=64, T @ 50Hz)` spanning `41-300 BPM`:
     `2.56s` rhythm autocorrelation, `1.28s` local beat-pair autocorrelation, and an explicit
     normalized BPM lag coordinate channel — processed through a lag-preserving 2D stem that
     never averages across BPM bins.
  3. Sub-band vertical readout (`freq_subbands=4`) preserving Doppler frequency-band location.
  4. Optional packed residual 2-layer BiGRU (`temporal_gru=True`) before `MaskedGatedAttentionMILPool`.
  5. Controlled backbone initialization via `init_mode`: `scratch`, `imagenet`, `audioset`
     (`topel/ConvNeXt-Tiny-AT`), `doppler_ssl`, or `audioset_doppler_ssl`.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from heartsounds.config import ModelConfig
from heartsounds.models.pooling import MaskedGatedAttentionMILPool, make_length_mask
from heartsounds.models.pretrain import load_audioset_convnext_tiny, resolve_init_mode
from heartsounds.schema import SUBTYPE_TO_INDEX


def _masked_instance_norm(x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Per-channel normalization over `(F, valid_T)` ignoring right-padded frames."""
    mask_4d = frame_mask.unsqueeze(1).unsqueeze(2).to(x.dtype)  # (B, 1, 1, T)
    count = (mask_4d.sum(dim=(-2, -1), keepdim=True) * x.shape[2]).clamp(min=1.0)
    x_masked = x * mask_4d
    mean = x_masked.sum(dim=(-2, -1), keepdim=True) / count
    var = (((x - mean) * mask_4d) ** 2).sum(dim=(-2, -1), keepdim=True) / count
    return ((x - mean) / torch.sqrt(var + 1e-5)) * mask_4d


def _apply_spec_augment(
    x: torch.Tensor,
    frame_lengths: torch.Tensor,
    *,
    freq_channels: int = 2,
) -> torch.Tensor:
    bsz, _, f_bins, _ = x.shape
    out = x.clone()
    for b in range(bsz):
        if torch.rand((), device=x.device).item() < 0.65:
            fw = int(torch.randint(2, max(4, f_bins // 8 + 2), (1,), device=x.device).item())
            f0 = int(torch.randint(0, max(1, f_bins - fw), (1,), device=x.device).item())
            out[b, :freq_channels, f0 : f0 + fw, :] = 0.0
        vlen = int(frame_lengths[b].item())
        if vlen > 20 and torch.rand((), device=x.device).item() < 0.65:
            tw = int(torch.randint(3, min(14, vlen // 4 + 4), (1,), device=x.device).item())
            t0 = int(torch.randint(0, max(1, vlen - tw), (1,), device=x.device).item())
            out[b, :, :, t0 : t0 + tw] = 0.0
    return out


def _windowed_autocorr_tempogram(
    env: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    win_size: int,
    lag_start: int,
    lag_count: int,
    fft_size: int = 256,
) -> torch.Tensor:
    """Strictly padding-invariant FFT sliding-window autocorrelation `(B, lag_count, T)`."""
    mask_f = frame_mask.to(env.dtype)
    valid_count = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    env_mean = (env * mask_f).sum(dim=-1, keepdim=True) / valid_count
    env_clean = (env - env_mean) * mask_f

    pad_left = win_size // 2
    pad_right = win_size - pad_left - 1
    padded = F.pad(
        env_clean.unsqueeze(1), (pad_left, pad_right), mode="constant", value=0.0
    ).squeeze(1)
    windows = padded.unfold(dimension=1, size=win_size, step=1)  # (B, T, W)
    centered = windows - windows.mean(dim=-1, keepdim=True)
    energy = (centered * centered).sum(dim=-1, keepdim=True).clamp(min=1e-6)

    spec = torch.fft.rfft(centered, n=fft_size, dim=-1)
    ac = torch.fft.irfft(spec * torch.conj(spec), n=fft_size, dim=-1)[..., :win_size]
    ac_norm = (ac / energy)[..., lag_start : lag_start + lag_count]  # (B, T, <=L)
    if ac_norm.shape[-1] < lag_count:
        ac_norm = F.pad(ac_norm, (0, lag_count - ac_norm.shape[-1]), mode="constant", value=0.0)
    return ac_norm.transpose(1, 2).contiguous() * mask_f.unsqueeze(1)


class OnGPUSpecTempogramFrontend(nn.Module):
    """`v0` frontend: aligned `(B, 3, 64, T_frames)` Log-Mel + Tempogram at 50 Hz."""

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

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=waveforms.device.type, enabled=False):
            wav32 = waveforms.float()
            sample_mask = make_length_mask(lengths, wav32.shape[1]).to(wav32.dtype)
            wav32 = wav32 * sample_mask

            m_short = torch.log(self.mel_short(wav32).clamp(min=1e-7))
            m_long = torch.log(self.mel_long(wav32).clamp(min=1e-7))
            t_frames = min(m_short.shape[-1], m_long.shape[-1])
            m_short = m_short[..., :t_frames]
            m_long = m_long[..., :t_frames]

            frame_lengths = torch.clamp(lengths // self.HOP_LENGTH + 1, min=1, max=t_frames)
            frame_mask = make_length_mask(frame_lengths, t_frames)

            env = m_short[:, 4:48, :].mean(dim=1)
            tempo = _windowed_autocorr_tempogram(
                env,
                frame_mask,
                win_size=self.TEMPO_WIN,
                lag_start=self.LAG_START,
                lag_count=self.LAG_COUNT,
            )
            stacked = torch.stack([m_short, m_long, tempo], dim=1)  # (B, 3, 64, T)
            normed = _masked_instance_norm(stacked, frame_mask)
            if self.training and self.spec_augment:
                normed = _apply_spec_augment(normed, frame_lengths, freq_channels=2)

        return normed, frame_lengths


class OnGPUDopplerDualFrontend(nn.Module):
    """`v1` decoupled frontend producing separate Acoustic Spectrogram & Dual-Window Tempogram.

    - `spec`: `(B, 3, n_mels, T)` — 16 ms Log-Mel (`n_fft=256`), 64 ms Log-Mel (`n_fft=512`),
      and temporal spectral-flux delta `d/dt Log-Mel`.
    - `tempo`: `(B, 3, 64, T)` — 2.56s autocorrelation (`41-300 BPM`), 1.28s local beat-pair
      autocorrelation, and explicit normalized BPM lag coordinate ramp `[-1, +1]`.
    """

    SAMPLE_RATE = 4000
    HOP_LENGTH = 80  # 20 ms -> 50 Hz
    TEMPO_WIN_LONG = 128  # 2.56s rhythm baseline
    TEMPO_WIN_SHORT = 64  # 1.28s local beat-pair window
    LAG_START = 10  # 0.20s -> 300 BPM (covers SVT/tachycardia cleanly)
    LAG_COUNT = 64  # 10..73 frames -> 0.20s..1.46s (41 - 300 BPM)

    def __init__(self, n_mels: int = 96, spec_augment: bool = True) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.spec_augment = spec_augment
        self.mel_short = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=256,
            win_length=64,
            hop_length=self.HOP_LENGTH,
            f_min=20.0,
            f_max=1950.0,
            n_mels=self.n_mels,
            power=2.0,
            center=True,
        )
        self.mel_long = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=512,
            win_length=256,
            hop_length=self.HOP_LENGTH,
            f_min=20.0,
            f_max=1950.0,
            n_mels=self.n_mels,
            power=2.0,
            center=True,
        )
        self.register_buffer(
            "lag_coord",
            torch.linspace(-1.0, 1.0, self.LAG_COUNT).view(1, 1, self.LAG_COUNT, 1),
            persistent=False,
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=waveforms.device.type, enabled=False):
            wav32 = waveforms.float()
            sample_mask = make_length_mask(lengths, wav32.shape[1]).to(wav32.dtype)
            wav32 = wav32 * sample_mask

            m_short = torch.log(self.mel_short(wav32).clamp(min=1e-7))
            m_long = torch.log(self.mel_long(wav32).clamp(min=1e-7))
            t_frames = min(m_short.shape[-1], m_long.shape[-1])
            m_short = m_short[..., :t_frames]
            m_long = m_long[..., :t_frames]

            frame_lengths = torch.clamp(lengths // self.HOP_LENGTH + 1, min=1, max=t_frames)
            frame_mask = make_length_mask(frame_lengths, t_frames)

            # Temporal first-difference (spectral flux) highlighting sharp valve/ectopic onsets
            m_delta = torch.zeros_like(m_short)
            if t_frames > 1:
                m_delta[..., 1:] = m_short[..., 1:] - m_short[..., :-1]

            spec_raw = torch.stack([m_short, m_long, m_delta], dim=1)  # (B, 3, F, T)
            spec = _masked_instance_norm(spec_raw, frame_mask)

            # Multi-band heartbeat envelope for tempogram (primary flow + valve click bands)
            lo_end = max(2, int(round(self.n_mels * 0.78)))
            env = m_short[:, 2:lo_end, :].mean(dim=1)
            tempo_long = _windowed_autocorr_tempogram(
                env,
                frame_mask,
                win_size=self.TEMPO_WIN_LONG,
                lag_start=self.LAG_START,
                lag_count=self.LAG_COUNT,
                fft_size=256,
            )
            tempo_short = _windowed_autocorr_tempogram(
                env,
                frame_mask,
                win_size=self.TEMPO_WIN_SHORT,
                lag_start=self.LAG_START,
                lag_count=self.LAG_COUNT,
                fft_size=128,
            )
            # Zero out lags >= TEMPO_WIN_SHORT in short-window AC cleanly (valid up to lag 63 -> 1.26s)
            coord = self.lag_coord.to(dtype=tempo_long.dtype).expand(
                waveforms.shape[0], 1, self.LAG_COUNT, t_frames
            )
            mask_4d = frame_mask.unsqueeze(1).unsqueeze(2).to(tempo_long.dtype)
            tempo = torch.cat(
                [tempo_long.unsqueeze(1), tempo_short.unsqueeze(1), coord * mask_4d],
                dim=1,
            )  # (B, 3, 64, T)

            if self.training and self.spec_augment:
                spec = _apply_spec_augment(spec, frame_lengths, freq_channels=3)

        return spec, tempo, frame_lengths


class TimewiseLayerNorm2D(nn.Module):
    """Per-timestep normalization across `(C, L)` on `(B, C, L, T)` — strictly invariant to T padding."""

    def __init__(self, channels: int, lag_bins: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm([channels, lag_bins])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L, T) -> (B, T, C, L) -> LayerNorm over (C, L) -> (B, C, L, T)
        return self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class TempogramLagStem2D(nn.Module):
    """Lag-preserving 2D stem that reduces `L=64 -> 8` ordered BPM bands without vertical averaging."""

    def __init__(self, temporal_stride: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        if temporal_stride not in (8, 16):
            raise ValueError(f"Expected temporal_stride in (8, 16), got {temporal_stride}")
        t_strides = (2, 2, 2) if temporal_stride == 8 else (4, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=(2, t_strides[0]), padding=1),
            TimewiseLayerNorm2D(32, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=(2, t_strides[1]), padding=1),
            TimewiseLayerNorm2D(48, 16),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=(2, t_strides[2]), padding=1),
            TimewiseLayerNorm2D(48, 8),
            nn.GELU(),
        )
        # L: 64 -> 32 -> 16 -> 8 ordered cardiac-rate bands, flattened (48 * 8 = 384)
        self.proj = nn.Sequential(
            nn.Conv1d(48 * 8, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tempo: torch.Tensor, target_t: int) -> torch.Tensor:
        h = self.conv(tempo)  # (B, 48, 8, T')
        bsz, ch, lag_bins, t_out = h.shape
        flat = h.reshape(bsz, ch * lag_bins, t_out)
        if t_out < target_t:
            flat = F.pad(flat, (0, target_t - t_out))
        elif t_out > target_t:
            flat = flat[..., :target_t]
        return self.proj(flat)


class SpecTempogramMILNet(nn.Module):
    """2D ConvNeXt / ResNet backbone over Log-Mel + Tempogram with Temporal Attention MIL."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.decoupled_tempogram = bool(cfg.decoupled_tempogram)
        self.freq_subbands = max(1, int(cfg.freq_subbands))
        self.use_temporal_gru = bool(cfg.temporal_gru)
        self.init_mode = resolve_init_mode(cfg.init_mode, cfg.pretrained)

        if self.decoupled_tempogram:
            self.frontend: nn.Module = OnGPUDopplerDualFrontend(
                n_mels=cfg.n_mels, spec_augment=True
            )
        else:
            self.frontend = OnGPUSpecTempogramFrontend(spec_augment=True)

        timm_pretrained = self.init_mode == "imagenet"
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=timm_pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(2, 3),
        )
        if self.init_mode in ("audioset", "audioset_doppler_ssl"):
            loaded = load_audioset_convnext_tiny(self.encoder, repo_id=cfg.audioset_repo)
            print(
                f"[init] Loaded {loaded} AudioSet tensors from {cfg.audioset_repo} into {cfg.backbone}",
                flush=True,
            )

        feat_channels = self.encoder.feature_info.channels()
        self.stage_stride = int(self.encoder.feature_info.reduction()[0])

        self.s3_bands = self.freq_subbands
        self.s4_bands = max(1, self.freq_subbands // 2) if self.freq_subbands > 1 else 1
        fused_in = 2 * self.s3_bands * feat_channels[0] + 2 * self.s4_bands * feat_channels[1]

        proj_dim = int(cfg.channels) * 2
        if self.decoupled_tempogram:
            tempo_dim = int(cfg.channels)
            self.tempo_stem: TempogramLagStem2D | None = TempogramLagStem2D(
                temporal_stride=self.stage_stride,
                out_dim=tempo_dim,
                dropout=cfg.dropout * 0.5,
            )
            fused_in += tempo_dim
        else:
            self.tempo_stem = None

        self.temporal_neck = nn.Sequential(
            nn.Conv1d(fused_in, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        if self.use_temporal_gru:
            self.temporal_gru: nn.GRU | None = nn.GRU(
                input_size=proj_dim,
                hidden_size=proj_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=cfg.dropout if cfg.dropout > 0 else 0.0,
            )
            self.temporal_post_norm: nn.LayerNorm | None = nn.LayerNorm(proj_dim)
        else:
            self.temporal_gru = None
            self.temporal_post_norm = None

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
    def _freq_pool(fmap: torch.Tensor, n_bands: int = 1) -> torch.Tensor:
        # fmap: (B, C, F', T') -> (B, 2*n_bands*C, T')
        if n_bands <= 1:
            return torch.cat([fmap.mean(dim=2), fmap.amax(dim=2)], dim=1)
        bsz, ch, _, t_len = fmap.shape
        avg_b = F.adaptive_avg_pool2d(fmap, (n_bands, t_len)).reshape(bsz, ch * n_bands, t_len)
        max_b = F.adaptive_max_pool2d(fmap, (n_bands, t_len)).reshape(bsz, ch * n_bands, t_len)
        return torch.cat([avg_b, max_b], dim=1)

    def extract_clean_spectrogram(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(spec_3ch, spec_lengths)` for masked spectrogram SSL pretraining."""
        if self.decoupled_tempogram:
            spec, _, spec_lengths = self.frontend(waveforms, lengths)
            return spec, spec_lengths
        spec, spec_lengths = self.frontend(waveforms, lengths)
        return spec, spec_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.decoupled_tempogram:
            img, tempo, spec_lengths = self.frontend(waveforms, lengths)
        else:
            img, spec_lengths = self.frontend(waveforms, lengths)
            tempo = None

        # Ensure minimum time width >= 64 frames so stride-32 stages never collapse to 0
        t_in = img.shape[-1]
        if t_in < 64:
            pad_amt = 64 - t_in
            img = F.pad(img, (0, pad_amt))
            if tempo is not None:
                tempo = F.pad(tempo, (0, pad_amt))

        stage3, stage4 = self.encoder(img)
        s3_t = self._freq_pool(stage3, n_bands=self.s3_bands)  # (B, 2*K3*C3, T3)
        t3_len = s3_t.shape[-1]

        # Strictly padding-invariant 2x upsample from stage4 (stride 2*S) to stage3 (stride S)
        s4_raw = self._freq_pool(stage4, n_bands=self.s4_bands)
        s4_up = F.interpolate(s4_raw, scale_factor=2, mode="nearest")
        if s4_up.shape[-1] < t3_len:
            s4_up = F.pad(s4_up, (0, t3_len - s4_up.shape[-1]))
        else:
            s4_up = s4_up[..., :t3_len]

        out_lengths = torch.clamp(
            (spec_lengths + self.stage_stride - 1) // self.stage_stride,
            min=1,
            max=t3_len,
        )
        mask_b1t = make_length_mask(out_lengths, t3_len).to(s3_t.dtype).unsqueeze(1)

        parts = [s3_t, s4_up]
        if self.tempo_stem is not None and tempo is not None:
            parts.append(self.tempo_stem(tempo, target_t=t3_len))

        seq = self.temporal_neck(torch.cat(parts, dim=1) * mask_b1t) * mask_b1t
        if self.temporal_gru is not None and self.temporal_post_norm is not None:
            packed = pack_padded_sequence(
                seq.transpose(1, 2),
                out_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            gru_packed, _ = self.temporal_gru(packed)
            gru_out, _ = pad_packed_sequence(gru_packed, batch_first=True, total_length=t3_len)
            seq = self.temporal_post_norm(seq.transpose(1, 2) + gru_out).transpose(1, 2) * mask_b1t

        pooled, attn_weights = self.pool(seq, out_lengths)
        h = self.head_drop(self.head_norm(pooled))
        return {
            "logits": self.binary_head(h).squeeze(-1),
            "subtype_logits": self.subtype_head(h),
            "attn_weights": attn_weights,
        }
