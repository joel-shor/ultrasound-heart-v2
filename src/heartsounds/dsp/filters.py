"""Multi-band Butterworth filtering and Hilbert/Shannon energy envelopes at 4 kHz.

Separates fetal Doppler recordings into physiologically meaningful frequency bands:
- Low band (25-150 Hz): slow myocardial wall motion
- Mid band (150-600 Hz): primary chamber blood flow & S1/S2 energy
- High band (600-1800 Hz): sharp valve clicks & atrial/ventricular transients
- Broad rhythm band (35-1200 Hz): rejects <35 Hz hand/probe motion artifacts
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal


@dataclass(frozen=True)
class MultiBandEnvelopes:
    """Smoothed amplitude and Shannon envelopes decimated to `env_sr` (default 400 Hz)."""

    env_sr: int
    broad_hilbert: np.ndarray
    broad_shannon: np.ndarray
    low_energy: np.ndarray
    mid_energy: np.ndarray
    high_energy: np.ndarray
    band_power_ratios: tuple[float, float, float]


def bandpass_filter(
    samples: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    sr: int = 4000,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * sr
    hi = min(high_hz, nyq * 0.96)
    lo = max(low_hz, 1.0)
    if samples.size < max(32, order * 8) or lo >= hi:
        return samples.astype(np.float32, copy=True)
    sos = scipy.signal.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return scipy.signal.sosfiltfilt(sos, samples).astype(np.float32)


def _lowpass_smooth_and_decimate(
    env: np.ndarray,
    *,
    sr: int = 4000,
    env_sr: int = 400,
    smooth_hz: float = 7.5,
) -> np.ndarray:
    decim = max(1, sr // env_sr)
    if env.size < 32:
        return env[::decim].astype(np.float32)
    sos = scipy.signal.butter(3, smooth_hz, btype="lowpass", fs=sr, output="sos")
    smoothed = scipy.signal.sosfiltfilt(sos, env)
    smoothed = np.maximum(smoothed, 0.0)
    return smoothed[::decim].astype(np.float32)


def log_compressed_energy(signal: np.ndarray) -> np.ndarray:
    """Monotonic, non-negative log-compressed energy `log(1 + 4*(|x|/p95)^2)`.

    Amplifies quiet/premature heartbeats while compressing high-amplitude probe spikes,
    avoiding the non-monotonic dip at `|x| > 0.61` of classical `-x^2 log(x^2)`.
    """
    scale = float(np.percentile(np.abs(signal), 95))
    if scale < 1e-6:
        return np.zeros_like(signal, dtype=np.float32)
    normed = np.clip(np.abs(signal) / scale, 0.0, 6.0)
    return np.log1p(4.0 * normed * normed).astype(np.float32)


def extract_multiband_envelopes(
    samples: np.ndarray,
    *,
    sr: int = 4000,
    env_sr: int = 400,
) -> MultiBandEnvelopes:
    """Compute multi-band filtered envelopes and relative band energy fractions."""
    x = np.asarray(samples, dtype=np.float32)
    broad = bandpass_filter(x, 35.0, 1200.0, sr=sr)
    low = bandpass_filter(x, 25.0, 150.0, sr=sr)
    mid = bandpass_filter(x, 150.0, 600.0, sr=sr)
    high = bandpass_filter(x, 600.0, 1800.0, sr=sr)

    p_low = float(np.mean(low * low)) + 1e-12
    p_mid = float(np.mean(mid * mid)) + 1e-12
    p_high = float(np.mean(high * high)) + 1e-12
    p_tot = p_low + p_mid + p_high

    analytic = scipy.signal.hilbert(broad)
    broad_hilb = _lowpass_smooth_and_decimate(
        np.abs(analytic).astype(np.float32), sr=sr, env_sr=env_sr
    )
    broad_shan = _lowpass_smooth_and_decimate(log_compressed_energy(broad), sr=sr, env_sr=env_sr)
    low_env = _lowpass_smooth_and_decimate(np.abs(low), sr=sr, env_sr=env_sr, smooth_hz=12.0)
    mid_env = _lowpass_smooth_and_decimate(np.abs(mid), sr=sr, env_sr=env_sr, smooth_hz=12.0)
    high_env = _lowpass_smooth_and_decimate(np.abs(high), sr=sr, env_sr=env_sr, smooth_hz=12.0)

    return MultiBandEnvelopes(
        env_sr=env_sr,
        broad_hilbert=broad_hilb,
        broad_shannon=broad_shan,
        low_energy=low_env,
        mid_energy=mid_env,
        high_energy=high_env,
        band_power_ratios=(p_low / p_tot, p_mid / p_tot, p_high / p_tot),
    )
