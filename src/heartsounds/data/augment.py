"""Physiologically safe waveform augmentations for fetal Doppler recordings.

Preserves rhythm periodicity and ectopic transients while regularizing against
phone microphone gain differences, background room/probe noise, and phase offsets.
"""

from __future__ import annotations

import numpy as np


def normalize_waveform(samples: np.ndarray, clip_sigma: float = 12.0) -> np.ndarray:
    """Zero-mean and robust-scale a Doppler waveform using its 90th-percentile magnitude."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    centered = arr - float(np.mean(arr))
    scale = float(np.percentile(np.abs(centered), 90))
    if scale < 1e-6:
        scale = float(np.std(centered))
    if scale < 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    normed = centered / scale
    return np.clip(normed, -clip_sigma, clip_sigma).astype(np.float32, copy=False)


def augment_waveform(
    samples: np.ndarray,
    rng: np.random.Generator,
    *,
    sample_rate: int = 4000,
) -> np.ndarray:
    """Apply stochastic gain, polarity flip, colored noise, and brief probe-dropout masks."""
    x = samples.copy()
    n = x.size
    if n < sample_rate:
        return x

    # 1. Random polarity inversion (acoustic phase is arbitrary)
    if rng.random() < 0.5:
        x = -x

    # 2. Random microphone gain scaling
    gain = float(rng.uniform(0.7, 1.35))
    x = x * gain

    # 3. Additive Gaussian + low-frequency probe-rub noise
    if rng.random() < 0.65:
        sigma = float(rng.uniform(0.01, 0.08))
        white = rng.standard_normal(n, dtype=np.float32) * sigma
        x = x + white

    # 4. Mild rate perturbation (±2.5%) — small enough to never cross clinical
    #    bradycardia (<110 BPM) / normal / tachycardia (>170 BPM) boundaries.
    if rng.random() < 0.35:
        rate_factor = float(rng.uniform(0.975, 1.025))
        new_len = max(sample_rate, int(round(n / rate_factor)))
        xp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x_interp = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
        x = np.interp(x_interp, xp, x).astype(np.float32)
        n = x.size

    # 5. Short time-dropout mask (simulates brief probe contact loss, <= 0.25s)
    if rng.random() < 0.4:
        n_masks = int(rng.integers(1, 3))
        max_mask_len = int(0.20 * sample_rate)
        for _ in range(n_masks):
            mlen = int(rng.integers(int(0.04 * sample_rate), max_mask_len + 1))
            if mlen < n:
                start = int(rng.integers(0, n - mlen))
                x[start : start + mlen] *= float(rng.uniform(0.0, 0.15))

    return x
