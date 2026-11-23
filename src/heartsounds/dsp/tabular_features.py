"""Interpretable, auditable DSP + HRV + ectopy tabular feature extraction.

Groups features into four domain-specific blocks:
1. Confidence-weighted global heart rate & period distribution
2. Beat-to-beat HRV & Poincaré geometry (`SD1`, `SD2`, `RMSSD`, `pNN20/50`)
3. Local ectopy & AV-block transition detectors (PAC short-long ratio, 2:1 period doubling, sliding 6-beat jump)
4. Signal periodicity quality (`quality_r_peak_mean`) & multi-band spectral descriptors
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.stats

from heartsounds.data.augment import normalize_waveform
from heartsounds.dsp.filters import MultiBandEnvelopes, extract_multiband_envelopes
from heartsounds.dsp.periodicity import PeriodicityTrack, estimate_periodicity_track


def _weighted_quantile(
    values: np.ndarray, quantiles: list[float], weights: np.ndarray
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(quantiles), dtype=np.float64)
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = np.maximum(weights[order], 1e-8)
    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, v_sorted)


def compute_rate_distribution_features(track: PeriodicityTrack) -> dict[str, float]:
    """Confidence-weighted heart rate (BPM) and period statistics across the recording."""
    if track.periods_s.size == 0:
        return {
            "bpm_w_mean": 140.0,
            "bpm_w_median": 140.0,
            "bpm_w_std": 0.0,
            "bpm_w_iqr": 0.0,
            "bpm_w_p05": 140.0,
            "bpm_w_p10": 140.0,
            "bpm_w_p90": 140.0,
            "bpm_w_p95": 140.0,
            "bpm_frac_below_110": 0.0,
            "bpm_frac_below_100": 0.0,
            "bpm_frac_above_165": 0.0,
            "bpm_frac_above_180": 0.0,
            "period_w_logvar": -12.0,
            "period_max_jump_s": 0.0,
            "period_hc_std_s": 0.0,
        }

    bpms = 60.0 / np.clip(track.periods_s.astype(np.float64), 0.20, 2.0)
    weights = np.clip(track.r_peaks.astype(np.float64), 0.05, 1.0) ** 2
    w_sum = float(np.sum(weights))
    w_mean_bpm = float(np.sum(bpms * weights) / w_sum)
    w_var_bpm = float(np.sum(weights * (bpms - w_mean_bpm) ** 2) / w_sum)

    q05, q10, q25, q50, q75, q90, q95 = _weighted_quantile(
        bpms, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], weights
    )

    periods = track.periods_s.astype(np.float64)
    w_mean_p = float(np.sum(periods * weights) / w_sum)
    w_var_p = float(np.sum(weights * (periods - w_mean_p) ** 2) / w_sum)
    hc_periods = periods[track.r_peaks >= 0.28]
    hc_std = float(np.std(hc_periods)) if hc_periods.size >= 2 else float(np.sqrt(w_var_p))
    p_jump = float(np.max(np.abs(np.diff(periods)))) if periods.size >= 2 else 0.0

    return {
        "bpm_w_mean": w_mean_bpm,
        "bpm_w_median": float(q50),
        "bpm_w_std": float(np.sqrt(max(0.0, w_var_bpm))),
        "bpm_w_iqr": float(q75 - q25),
        "bpm_w_p05": float(q05),
        "bpm_w_p10": float(q10),
        "bpm_w_p90": float(q90),
        "bpm_w_p95": float(q95),
        "bpm_frac_below_110": float(np.sum(weights[bpms < 110.0]) / w_sum),
        "bpm_frac_below_100": float(np.sum(weights[bpms < 100.0]) / w_sum),
        "bpm_frac_above_165": float(np.sum(weights[bpms > 165.0]) / w_sum),
        "bpm_frac_above_180": float(np.sum(weights[bpms > 180.0]) / w_sum),
        "period_w_logvar": float(np.log(max(w_var_p, 1e-10))),
        "period_max_jump_s": p_jump,
        "period_hc_std_s": hc_std,
    }


def compute_hrv_poincare_features(ibi_s: np.ndarray) -> dict[str, float]:
    """Standard clinical HRV and Poincaré geometry metrics from beat-to-beat intervals."""
    if ibi_s.size < 3:
        return {
            "ibi_count": float(ibi_s.size),
            "ibi_mean_s": 0.43,
            "ibi_median_s": 0.43,
            "ibi_cv": 0.0,
            "ibi_logvar_norm": -12.0,
            "ibi_rmssd_s": 0.0,
            "ibi_rmssd_norm": 0.0,
            "ibi_pnn20": 0.0,
            "ibi_pnn50": 0.0,
            "poincare_sd1": 0.0,
            "poincare_sd2": 0.0,
            "poincare_sd1_sd2_ratio": 0.0,
            "ibi_norm_p04": 1.0,
            "ibi_norm_p10": 1.0,
            "ibi_norm_p20": 1.0,
            "ibi_norm_p80": 1.0,
            "ibi_norm_p90": 1.0,
            "ibi_norm_p96": 1.0,
        }

    ibi = ibi_s.astype(np.float64)
    mean_ibi = float(np.mean(ibi))
    med_ibi = float(np.median(ibi))
    std_ibi = float(np.std(ibi))
    diffs = np.diff(ibi)
    abs_diffs = np.abs(diffs)
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    sdsd = float(np.std(diffs))

    # Poincaré descriptors: SD1 captures short-term beat-to-beat jump; SD2 captures slow drift
    sd1 = float(np.sqrt(0.5) * sdsd)
    sd2_sq = max(0.0, 2.0 * (std_ibi**2) - 0.5 * (sdsd**2))
    sd2 = float(np.sqrt(sd2_sq))

    norm_ibi = ibi / max(med_ibi, 1e-4)
    p04, p10, p20, p80, p90, p96 = np.percentile(norm_ibi, [4, 10, 20, 80, 90, 96])

    return {
        "ibi_count": float(ibi.size),
        "ibi_mean_s": mean_ibi,
        "ibi_median_s": med_ibi,
        "ibi_cv": float(std_ibi / max(mean_ibi, 1e-4)),
        "ibi_logvar_norm": float(np.log(max(float(np.var(norm_ibi)), 1e-10))),
        "ibi_rmssd_s": rmssd,
        "ibi_rmssd_norm": float(rmssd / max(med_ibi, 1e-4)),
        "ibi_pnn20": float(np.mean(abs_diffs > 0.020)),
        "ibi_pnn50": float(np.mean(abs_diffs > 0.050)),
        "poincare_sd1": sd1,
        "poincare_sd2": sd2,
        "poincare_sd1_sd2_ratio": float(sd1 / (sd2 + 1e-5)),
        "ibi_norm_p04": float(p04),
        "ibi_norm_p10": float(p10),
        "ibi_norm_p20": float(p20),
        "ibi_norm_p80": float(p80),
        "ibi_norm_p90": float(p90),
        "ibi_norm_p96": float(p96),
    }


def compute_ectopy_avb_window_features(
    ibi_s: np.ndarray, track: PeriodicityTrack
) -> dict[str, float]:
    """Targeted features for premature contractions (PAC/PVC), compensatory pauses, and AV block."""
    if ibi_s.size < 4:
        return {
            "ratio_min_succ": 1.0,
            "ratio_max_succ": 1.0,
            "pac_short_long_score": 0.0,
            "pac_triplet_count": 0.0,
            "avb_double_period_frac": 0.0,
            "outlier_beat_frac_12pct": 0.0,
            "outlier_beat_frac_20pct": 0.0,
            "win6_max_cv": 0.0,
            "win6_median_cv": 0.0,
            "win6_max_jump_norm": 0.0,
            "subharmonic_ratio_mean": float(np.mean(track.subharmonic_ratios))
            if track.subharmonic_ratios.size
            else 0.0,
            "subharmonic_ratio_max": float(np.max(track.subharmonic_ratios))
            if track.subharmonic_ratios.size
            else 0.0,
        }

    ibi = ibi_s.astype(np.float64)
    med = max(float(np.median(ibi)), 1e-4)
    norm_ibi = ibi / med
    ratios = ibi[1:] / np.maximum(ibi[:-1], 1e-4)

    # Classic PAC pattern: premature beat (r_prev < 0.85) immediately followed by pause (r_next > 1.15)
    short_part = np.maximum(0.0, 1.0 - ratios[:-1])
    long_part = np.maximum(0.0, ratios[1:] - 1.0)
    pac_scores = short_part * long_part
    pac_triplets = int(np.sum((ratios[:-1] < 0.86) & (ratios[1:] > 1.14)))

    # 2:1 AV block or dropped beat: interval ~ 1.75x - 2.25x baseline period
    double_frac = float(np.mean((norm_ibi >= 1.70) & (norm_ibi <= 2.35)))
    out12 = float(np.mean(np.abs(norm_ibi - 1.0) > 0.12))
    out20 = float(np.mean(np.abs(norm_ibi - 1.0) > 0.20))

    # Sliding 6-beat local windows catch transient ectopy even in 40-beat recordings
    win_cvs: list[float] = []
    win_jumps: list[float] = []
    win = min(6, ibi.size)
    for i in range(ibi.size - win + 1):
        seg = ibi[i : i + win]
        s_med = max(float(np.median(seg)), 1e-4)
        win_cvs.append(float(np.std(seg) / s_med))
        win_jumps.append(float(np.max(np.abs(np.diff(seg))) / s_med))

    return {
        "ratio_min_succ": float(np.min(ratios)),
        "ratio_max_succ": float(np.max(ratios)),
        "pac_short_long_score": float(np.max(pac_scores)) if pac_scores.size else 0.0,
        "pac_triplet_count": float(pac_triplets),
        "avb_double_period_frac": double_frac,
        "outlier_beat_frac_12pct": out12,
        "outlier_beat_frac_20pct": out20,
        "win6_max_cv": float(np.max(win_cvs)),
        "win6_median_cv": float(np.median(win_cvs)),
        "win6_max_jump_norm": float(np.max(win_jumps)),
        "subharmonic_ratio_mean": float(np.mean(track.subharmonic_ratios))
        if track.subharmonic_ratios.size
        else 0.0,
        "subharmonic_ratio_max": float(np.max(track.subharmonic_ratios))
        if track.subharmonic_ratios.size
        else 0.0,
    }


def compute_signal_quality_spectral_features(
    norm_samples: np.ndarray,
    envs: MultiBandEnvelopes,
    track: PeriodicityTrack,
    *,
    sr: int = 4000,
) -> dict[str, float]:
    """Signal periodicity quality (`quality_score`), modulation spectrum, and acoustic texture."""
    r_peaks = track.r_peaks.astype(np.float64)
    r_mean = float(np.mean(r_peaks)) if r_peaks.size else 0.0
    r_p90 = float(np.percentile(r_peaks, 90)) if r_peaks.size else 0.0
    r_std = float(np.std(r_peaks)) if r_peaks.size else 0.0
    r_ok_frac = float(np.mean(r_peaks >= 0.28)) if r_peaks.size else 0.0

    # Power spectrum of the 400 Hz envelope: clean heartbeat concentrates modulation in 1.3-3.5 Hz
    env_centered = envs.broad_shannon.astype(np.float64) - float(np.mean(envs.broad_shannon))
    if env_centered.size >= 64:
        freqs_e, psd_e = scipy.signal.welch(
            env_centered, fs=envs.env_sr, nperseg=min(512, env_centered.size)
        )
        fetal_mod = float(np.sum(psd_e[(freqs_e >= 1.6) & (freqs_e <= 3.3)]))
        slow_mod = float(np.sum(psd_e[(freqs_e >= 0.7) & (freqs_e < 1.6)]))
        fast_mod = float(np.sum(psd_e[(freqs_e > 3.3) & (freqs_e <= 10.0)]))
        tot_mod = fetal_mod + slow_mod + fast_mod + 1e-12
        mod_fetal_frac = fetal_mod / tot_mod
        mod_slow_frac = slow_mod / tot_mod
    else:
        mod_fetal_frac = 0.0
        mod_slow_frac = 0.0

    # Audio power spectrum features (20 - 2000 Hz)
    if norm_samples.size >= 256:
        freqs, psd = scipy.signal.welch(
            norm_samples.astype(np.float64), fs=sr, nperseg=min(1024, norm_samples.size)
        )
        psd = np.maximum(psd, 1e-12)
        psd_norm = psd / np.sum(psd)
        centroid = float(np.sum(freqs * psd_norm))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm)))
        flatness = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    else:
        centroid, bandwidth, flatness = 0.0, 0.0, 1.0

    rms = float(np.sqrt(np.mean(norm_samples * norm_samples)) + 1e-8)
    crest = float(np.max(np.abs(norm_samples)) / rms)
    kurt = (
        float(scipy.stats.kurtosis(norm_samples, fisher=True, bias=False))
        if norm_samples.size > 8
        else 0.0
    )
    dropout_frac = float(np.mean(np.abs(norm_samples) < 0.05))

    p_low, p_mid, p_high = envs.band_power_ratios
    return {
        "duration_s": float(norm_samples.size / float(sr)),
        "quality_r_peak_mean": r_mean,
        "quality_r_peak_p90": r_p90,
        "quality_r_peak_std": r_std,
        "quality_r_ok_frac": r_ok_frac,
        "fetal_maternal_ac_ratio": float(np.median(track.fetal_maternal_ratios))
        if track.fetal_maternal_ratios.size
        else 1.0,
        "mod_fetal_band_frac": mod_fetal_frac,
        "mod_slow_band_frac": mod_slow_frac,
        "band_power_low_frac": float(p_low),
        "band_power_mid_frac": float(p_mid),
        "band_power_high_frac": float(p_high),
        "spec_centroid_hz": centroid,
        "spec_bandwidth_hz": bandwidth,
        "spec_flatness": flatness,
        "waveform_crest_factor": crest,
        "waveform_kurtosis": float(np.clip(kurt, -10.0, 100.0)),
        "waveform_dropout_frac": dropout_frac,
    }


def extract_recording_dsp_features(samples: np.ndarray, *, sr: int = 4000) -> dict[str, float]:
    """Extract the complete auditable ~62-feature DSP/HRV vector from a 4 kHz waveform."""
    normed = normalize_waveform(samples)
    envs = extract_multiband_envelopes(normed, sr=sr, env_sr=400)
    track = estimate_periodicity_track(envs.broad_hilbert, envs.broad_shannon, env_sr=400)

    merged: dict[str, float] = {}
    merged.update(compute_rate_distribution_features(track))
    merged.update(compute_hrv_poincare_features(track.ibi_s))
    merged.update(compute_ectopy_avb_window_features(track.ibi_s, track))
    merged.update(compute_signal_quality_spectral_features(normed, envs, track, sr=sr))
    merged["beat_detection_density_hz"] = float(
        merged["ibi_count"] / max(merged["duration_s"], 1.0)
    )

    for k, v in list(merged.items()):
        if not np.isfinite(v):
            merged[k] = 0.0
    return merged
