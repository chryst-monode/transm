"""Transient shaper — boost attacks, tame sustain."""

from __future__ import annotations

import numpy as np

from transm.dsp.common import (
    apply_gain_curve,
    envelope_follower,
    linear_to_db,
    smooth_gain,
    time_to_coeff,
)
from transm.types import AudioBuffer


def shape_transients(
    buffer: AudioBuffer,
    attack_gain_db: float = 4.0,
    sustain_gain_db: float = -2.0,
    attack_ms: float = 1.0,
    release_ms: float = 50.0,
    slow_attack_ms: float = 20.0,
    slow_release_ms: float = 200.0,
    sensitivity: float = 1.0,
) -> AudioBuffer:
    """Enhance or suppress transient attacks and sustain.

    Algorithm
    ---------
    1. Fast envelope (short attack/release) tracks transient peaks.
    2. Slow envelope (long attack/release) tracks sustain.
    3. Difference in dB ≈ "transient-ness" per sample.
    4. Build per-sample gain curve: positive diff → attack_gain_db,
       negative diff → sustain_gain_db, scaled by sensitivity.
    5. Normalize by 95th percentile of positive values.
    6. Smooth with 200 Hz low-pass to avoid artifacts.
    7. Apply to signal, soft-clip samples above 0.95.

    Edge cases
    ----------
    - Signals below −60 dBFS are bypassed (no gain applied to silence).
    - Stereo: gain curve derived from mid channel (L+R)/2,
      applied identically to both channels.
    """
    sr = buffer.sample_rate
    data = buffer.data  # (samples, channels)

    # --- derive a mono analysis signal ----------------------------------
    if data.shape[1] >= 2:
        analysis = (data[:, 0] + data[:, 1]) / 2.0
    else:
        analysis = data[:, 0].copy()
    analysis = analysis.astype(np.float64)

    # --- silence gate: bypass if peak < −60 dBFS -----------------------
    peak_lin = float(np.max(np.abs(analysis)))
    if peak_lin < 1e-3:  # ≈ −60 dBFS
        return buffer

    # --- envelope coefficients -----------------------------------------
    fast_att = time_to_coeff(attack_ms, sr)
    fast_rel = time_to_coeff(release_ms, sr)
    slow_att = time_to_coeff(slow_attack_ms, sr)
    slow_rel = time_to_coeff(slow_release_ms, sr)

    # --- fast & slow envelopes -----------------------------------------
    env_fast = envelope_follower(analysis, fast_att, fast_rel)
    env_slow = envelope_follower(analysis, slow_att, slow_rel)

    # --- transient-ness in dB ------------------------------------------
    env_fast_db = np.asarray(linear_to_db(env_fast, floor_db=-120.0))
    env_slow_db = np.asarray(linear_to_db(env_slow, floor_db=-120.0))
    diff_db = env_fast_db - env_slow_db

    # --- build gain curve (un-normalised) ------------------------------
    gain_db = np.where(diff_db > 0, diff_db * attack_gain_db, diff_db * -sustain_gain_db)

    # --- normalise using 95th percentile of positive values ------------
    pos_vals = diff_db[diff_db > 0]
    if len(pos_vals) > 0:
        p95 = float(np.percentile(pos_vals, 95))
        if p95 > 0:
            gain_db = gain_db / p95
            # Re-scale so the target gains are reached at p95
            gain_db = np.where(
                diff_db > 0,
                gain_db * attack_gain_db * sensitivity,
                gain_db * abs(sustain_gain_db) * sensitivity,
            )

    # --- smooth with 200 Hz lowpass ------------------------------------
    gain_db_smoothed = smooth_gain(gain_db, cutoff_hz=200.0, sr=sr).astype(np.float32)

    # --- silence mask: zero gain where signal is very quiet ------------
    silence_mask = np.abs(analysis) < 1e-3
    gain_db_smoothed[silence_mask] = 0.0

    # --- apply gain curve to all channels ------------------------------
    result = apply_gain_curve(data, gain_db_smoothed)

    # --- soft-clip via tanh for samples above 0.95 ---------------------
    above = np.abs(result) > 0.95
    if np.any(above):
        result = np.where(above, np.tanh(result), result).astype(np.float32)

    return AudioBuffer(data=result, sample_rate=sr)
