"""Unit tests for 4 kHz multi-band DSP, periodicity, and HRV/ectopy features."""

from __future__ import annotations

import numpy as np

from heartsounds.dsp.tabular_features import extract_recording_dsp_features


def _synth_doppler_burst_train(
    beat_times_s: list[float],
    duration_s: float = 16.0,
    sr: int = 4000,
    carrier_hz: float = 320.0,
) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float32) / float(sr)
    sig = np.zeros_like(t)
    for bt in beat_times_s:
        env = np.exp(-0.5 * ((t - bt) / 0.025) ** 2)
        sig += env * np.sin(2.0 * np.pi * carrier_hz * t)
    return sig.astype(np.float32)


def test_normal_140bpm_vs_bradycardia_vs_pac_ectopy():
    # 1. Clean 140 BPM normal sinus rhythm (period = 0.4286s)
    normal_beats = list(np.arange(0.3, 15.8, 60.0 / 140.0))
    sig_norm = _synth_doppler_burst_train(normal_beats)
    f_norm = extract_recording_dsp_features(sig_norm, sr=4000)

    assert 134.0 <= f_norm["bpm_w_median"] <= 146.0
    assert f_norm["quality_r_peak_mean"] > 0.40
    assert f_norm["pac_triplet_count"] == 0.0
    assert f_norm["ibi_cv"] < 0.05

    # 2. Bradycardia at 85 BPM
    brady_beats = list(np.arange(0.4, 15.8, 60.0 / 85.0))
    sig_brady = _synth_doppler_burst_train(brady_beats)
    f_brady = extract_recording_dsp_features(sig_brady, sr=4000)
    assert 80.0 <= f_brady["bpm_w_median"] <= 92.0
    assert f_brady["bpm_frac_below_110"] > 0.80

    # 3. Simulated PAC ectopy: premature beats followed by compensatory pauses
    pac_beats = []
    cur = 0.3
    base_p = 60.0 / 140.0
    k = 0
    while cur < 15.5:
        pac_beats.append(cur)
        if k % 6 == 3:
            pac_beats.append(cur + 0.62 * base_p)
            cur += (0.62 + 1.38) * base_p
        else:
            cur += base_p
        k += 1
    sig_pac = _synth_doppler_burst_train(pac_beats)
    f_pac = extract_recording_dsp_features(sig_pac, sr=4000)

    assert f_pac["pac_short_long_score"] > f_norm["pac_short_long_score"] + 0.03
    assert f_pac["pac_triplet_count"] >= 2.0
    assert f_pac["win6_max_cv"] > f_norm["win6_max_cv"] * 2.5

    # 4. White noise has poor autocorrelation periodicity quality
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(4000 * 12).astype(np.float32)
    f_noise = extract_recording_dsp_features(noise, sr=4000)
    assert f_noise["quality_r_peak_mean"] < f_norm["quality_r_peak_mean"] * 0.5
