"""Shared DSP utilities: envelope follower, gain helpers, coefficient conversion."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Envelope follower — numba-accelerated with pure-Python fallback
# ---------------------------------------------------------------------------

def _envelope_follower_python(
    signal_1d: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
) -> np.ndarray:
    """Pure-Python one-pole IIR envelope follower (attack/release split)."""
    n = len(signal_1d)
    env = np.empty(n, dtype=np.float64)
    current = 0.0
    for i in range(n):
        inp = abs(signal_1d[i])
        if inp > current:
            coeff = attack_coeff
        else:
            coeff = release_coeff
        current = coeff * current + (1.0 - coeff) * inp
        env[i] = current
    return env


try:
    import numba  # type: ignore[import-untyped]

    @numba.jit(nopython=True)  # type: ignore[misc]
    def _envelope_follower_numba(
        signal_1d: np.ndarray,
        attack_coeff: float,
        release_coeff: float,
    ) -> np.ndarray:
        n = len(signal_1d)
        env = np.empty(n, dtype=np.float64)
        current = 0.0
        for i in range(n):
            inp = abs(signal_1d[i])
            if inp > current:
                coeff = attack_coeff
            else:
                coeff = release_coeff
            current = coeff * current + (1.0 - coeff) * inp
            env[i] = current
        return env

    _envelope_impl = _envelope_follower_numba
except ImportError:
    _envelope_impl = _envelope_follower_python


def envelope_follower(
    signal_1d: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
) -> np.ndarray:
    """One-pole IIR envelope follower with separate attack/release coefficients.

    Parameters
    ----------
    signal_1d : 1-D array of audio samples
    attack_coeff : IIR coefficient used when input exceeds current envelope
    release_coeff : IIR coefficient used when input is below current envelope

    Returns
    -------
    Envelope array (same length as input, float64).
    """
    return _envelope_impl(
        np.asarray(signal_1d, dtype=np.float64),
        float(attack_coeff),
        float(release_coeff),
    )


# ---------------------------------------------------------------------------
# Coefficient & dB helpers
# ---------------------------------------------------------------------------

def time_to_coeff(time_ms: float, sr: int) -> float:
    """Convert a time constant in milliseconds to a one-pole IIR coefficient.

    coeff = exp(-1 / (time_ms/1000 * sr))
    """
    if time_ms <= 0:
        return 0.0
    return math.exp(-1.0 / (time_ms / 1000.0 * sr))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain: 10 ** (db / 20)."""
    return 10.0 ** (db / 20.0)


def linear_to_db(
    linear: float | NDArray[np.floating],
    floor_db: float = -120.0,
) -> float | NDArray[np.floating]:
    """Convert linear amplitude to dB with a noise floor.

    Returns 20 * log10(max(linear, floor_linear)).
    """
    floor_linear = 10.0 ** (floor_db / 20.0)
    if isinstance(linear, np.ndarray):
        safe = np.maximum(linear, floor_linear)
        return 20.0 * np.log10(safe)  # type: ignore[return-value]
    return 20.0 * math.log10(max(float(linear), floor_linear))


# ---------------------------------------------------------------------------
# Gain-curve smoothing and application
# ---------------------------------------------------------------------------

def smooth_gain(gain_curve: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """Low-pass filter a gain curve with a 2nd-order Butterworth (zero-phase).

    This removes zipper noise / abrupt gain transitions.
    """
    nyq = sr / 2.0
    # Clamp cutoff to a safe fraction of Nyquist
    freq = min(cutoff_hz, nyq * 0.95)
    sos = butter(2, freq, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, gain_curve).astype(np.float64)


def apply_gain_curve(signal: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """Multiply *signal* by a per-sample gain curve specified in dB.

    Both arrays must share the first (sample) dimension. The gain curve is
    broadcast across channels if *signal* is 2-D.
    """
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    if signal.ndim == 2 and gain_linear.ndim == 1:
        gain_linear = gain_linear[:, np.newaxis]
    return (signal * gain_linear).astype(np.float32)
