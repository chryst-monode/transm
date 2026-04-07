"""Downward expander — reduce level of quiet passages to restore dynamic range."""

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


def expand_downward(
    buffer: AudioBuffer,
    threshold_db: float = -30.0,
    ratio: float = 1.5,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    knee_db: float = 6.0,
    range_db: float = -20.0,
) -> AudioBuffer:
    """Apply downward expansion to an AudioBuffer.

    Algorithm
    ---------
    1. Compute envelope → convert to dB.
    2. Below threshold: ``gain_reduction = (env_db - threshold_db) * (ratio - 1) / ratio``
    3. Soft knee: quadratic interpolation within [threshold ± knee/2].
    4. Clamp to ``range_db`` floor (maximum attenuation).
    5. Smooth gain curve and apply.

    Stereo handling: envelope is derived from mid (L+R)/2, gain applied to both.
    """
    sr = buffer.sample_rate
    data = buffer.data  # (samples, channels)

    # --- mono analysis signal -------------------------------------------
    if data.shape[1] >= 2:
        analysis = (data[:, 0] + data[:, 1]) / 2.0
    else:
        analysis = data[:, 0].copy()
    analysis = analysis.astype(np.float64)

    # --- envelope -------------------------------------------------------
    att = time_to_coeff(attack_ms, sr)
    rel = time_to_coeff(release_ms, sr)
    env = envelope_follower(analysis, att, rel)
    env_db = np.asarray(linear_to_db(env, floor_db=-120.0))

    # --- gain-reduction curve -------------------------------------------
    gain_db = _compute_gain(env_db, threshold_db, ratio, knee_db, range_db)

    # --- smooth ---------------------------------------------------------
    gain_db_smoothed = smooth_gain(gain_db, cutoff_hz=100.0, sr=sr).astype(np.float32)

    # --- apply ----------------------------------------------------------
    result = apply_gain_curve(data, gain_db_smoothed)
    return AudioBuffer(data=result, sample_rate=sr)


def _compute_gain(
    env_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
    range_db: float,
) -> np.ndarray:
    """Vectorised gain-reduction calculation with soft knee."""
    gain = np.zeros_like(env_db)
    half_knee = knee_db / 2.0
    slope = (ratio - 1.0) / ratio  # expansion slope

    knee_lo = threshold_db - half_knee
    knee_hi = threshold_db + half_knee

    # --- below knee region: full expansion ------------------------------
    below = env_db < knee_lo
    gain[below] = (env_db[below] - threshold_db) * slope

    # --- knee region: quadratic interpolation ---------------------------
    in_knee = (env_db >= knee_lo) & (env_db <= knee_hi)
    if knee_db > 0 and np.any(in_knee):
        x = env_db[in_knee] - threshold_db + half_knee  # 0 … knee_db
        # Quadratic: gain ramps from 0 at knee_hi to full slope at knee_lo
        gain[in_knee] = -slope * (x - knee_db) ** 2 / (2.0 * knee_db)

    # --- above knee: no gain reduction (already zero) -------------------

    # --- clamp to range_db floor ----------------------------------------
    gain = np.maximum(gain, range_db)

    return gain
