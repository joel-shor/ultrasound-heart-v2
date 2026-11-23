"""Confidence-gated sliding-window autocorrelation and beat interval extraction.

Upgrades the legacy Jezewski periodicity estimator with:
1. Per-window normalized autocorrelation confidence (`r_peak`) so noisy probe-motion
   segments do not masquerade as arrhythmias.
2. Extended period search `[0.27s, 1.15s]` (~52-222 BPM) covering severe AV-block
   bradycardia through supraventricular tachycardia.
3. Subharmonic peak ratio `R(2*T) / R(T)` sensitive to bigeminy and 2:1 AV block.
4. Direct Shannon-envelope peak segmentation for beat-to-beat ectopy ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal


@dataclass(frozen=True)
class PeriodicityTrack:
    """Sliding-window periodicity and confidence estimates across a recording."""

    times_s: np.ndarray  # Window center timestamps (s)
    periods_s: np.ndarray  # Estimated beat period per window (s)
    r_peaks: np.ndarray  # Normalized autocorr peak prominence in [0, 1]
    subharmonic_ratios: np.ndarray  # R(2*T) / max(R(T), eps)
    fetal_maternal_ratios: np.ndarray
    beat_times_s: np.ndarray  # Detected Shannon-envelope beat timestamps (s)
    ibi_s: np.ndarray  # Beat-to-beat intervals diff(beat_times_s)


def _fft_normalized_autocorr(window: np.ndarray) -> np.ndarray:
    centered = window.astype(np.float64) - float(np.mean(window))
    energy = float(np.dot(centered, centered))
    if energy < 1e-12:
        return np.zeros_like(centered, dtype=np.float64)
    n = centered.size
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(centered, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    return ac / energy


def estimate_periodicity_track(
    hilbert_env: np.ndarray,
    shannon_env: np.ndarray,
    *,
    env_sr: int = 400,
    window_s: float = 2.6,
    step_s: float = 0.25,
    min_period_s: float = 0.27,
    max_period_s: float = 1.15,
) -> PeriodicityTrack:
    """Estimate local beat period, autocorrelation confidence, and beat-to-beat IBIs."""
    combined_env = 0.5 * (
        hilbert_env / (np.percentile(hilbert_env, 95) + 1e-6)
        + shannon_env / (np.percentile(shannon_env, 95) + 1e-6)
    )
    n = combined_env.size
    win_len = int(round(window_s * env_sr))
    step_len = max(1, int(round(step_s * env_sr)))
    lag_min = max(1, int(round(min_period_s * env_sr)))
    lag_max = int(round(max_period_s * env_sr))

    times: list[float] = []
    periods: list[float] = []
    r_peaks: list[float] = []
    subharm: list[float] = []
    fm_ratios: list[float] = []

    fetal_lo, fetal_hi = int(0.33 * env_sr), int(0.56 * env_sr)
    slow_lo, slow_hi = int(0.62 * env_sr), int(1.10 * env_sr)

    prev_period = 0.43  # ~140 BPM prior
    starts = list(range(0, max(1, n - win_len + 1), step_len))
    if n < win_len and n > lag_max + 4:
        win_len = n
        starts = [0]

    for start in starts:
        seg = combined_env[start : start + win_len]
        if seg.size <= lag_min + 4:
            break
        cur_lag_max = min(lag_max, seg.size - 2)
        if cur_lag_max <= lag_min:
            break

        ac = _fft_normalized_autocorr(seg)
        lags = np.arange(lag_min, cur_lag_max + 1)
        lag_secs = lags / float(env_sr)

        # Mild continuity weighting to avoid octave-jumping on clean signals while
        # still allowing genuine bradycardia/tachycardia transitions
        continuity = 1.0 - 0.18 * np.minimum(np.abs(lag_secs - prev_period) / prev_period, 1.5)
        weighted = ac[lag_min : cur_lag_max + 1] * continuity

        local_peaks, _ = scipy.signal.find_peaks(
            ac[lag_min : cur_lag_max + 1], distance=int(0.08 * env_sr)
        )
        if local_peaks.size > 0:
            best_rel = int(local_peaks[int(np.argmax(weighted[local_peaks]))])
        else:
            best_rel = int(np.argmax(weighted))

        best_lag = lag_min + best_rel
        r_val = float(np.clip(ac[best_lag], 0.0, 1.0))
        period_val = best_lag / float(env_sr)
        if r_val > 0.20:
            prev_period = period_val

        double_lag = min(2 * best_lag, seg.size - 1)
        sub_ratio = (
            float(max(0.0, ac[double_lag]) / max(r_val, 0.05)) if 2 * best_lag < seg.size else 0.0
        )

        fetal_max = (
            float(np.max(ac[fetal_lo : min(fetal_hi, seg.size)])) if seg.size > fetal_lo else 0.0
        )
        slow_max = (
            float(np.max(ac[slow_lo : min(slow_hi, seg.size)])) if seg.size > slow_lo else 0.0
        )
        fm_ratio = float(max(0.0, fetal_max) / (max(0.0, slow_max) + 0.05))

        times.append((start + 0.5 * seg.size) / float(env_sr))
        periods.append(period_val)
        r_peaks.append(r_val)
        subharm.append(sub_ratio)
        fm_ratios.append(fm_ratio)

    # Direct beat detection on smoothed Shannon envelope for transient ectopy / PAC detection
    beat_times_s, ibi_s = extract_beat_intervals(shannon_env, env_sr=env_sr)

    return PeriodicityTrack(
        times_s=np.asarray(times, dtype=np.float32),
        periods_s=np.asarray(periods, dtype=np.float32),
        r_peaks=np.asarray(r_peaks, dtype=np.float32),
        subharmonic_ratios=np.asarray(subharm, dtype=np.float32),
        fetal_maternal_ratios=np.asarray(fm_ratios, dtype=np.float32),
        beat_times_s=beat_times_s,
        ibi_s=ibi_s,
    )


def extract_beat_intervals(
    shannon_env: np.ndarray,
    *,
    env_sr: int = 400,
    min_beat_distance_s: float = 0.24,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate individual heartbeat peaks on the Shannon envelope and return (times_s, ibi_s)."""
    if shannon_env.size < int(0.6 * env_sr):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    p90 = float(np.percentile(shannon_env, 90))
    if p90 < 1e-7:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    norm_env = shannon_env / p90
    min_dist = max(1, int(round(min_beat_distance_s * env_sr)))
    prom = max(0.12, float(np.percentile(norm_env, 65) - np.percentile(norm_env, 25)) * 0.35)

    peaks, _ = scipy.signal.find_peaks(
        norm_env,
        distance=min_dist,
        prominence=prom,
        height=0.18,
    )
    beat_times = peaks.astype(np.float32) / float(env_sr)
    if beat_times.size < 2:
        return beat_times, np.zeros(0, dtype=np.float32)
    ibis = np.diff(beat_times)
    # Discard physiologically impossible >2.2s multi-beat dropouts from IBI diffs
    valid_ibis = ibis[(ibis >= 0.22) & (ibis <= 2.20)]
    return beat_times, valid_ibis.astype(np.float32)
