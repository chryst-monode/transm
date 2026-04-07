"""Final true-peak limiting — normalize to target LUFS with ceiling enforcement."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import resample_poly

from transm.analysis import measure_lufs
from transm.types import AudioBuffer

logger = logging.getLogger(__name__)


def apply_final_limiter(
    buffer: AudioBuffer,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
    max_iterations: int = 5,
) -> AudioBuffer:
    """Final mastering stage: normalize to target LUFS with true-peak ceiling.

    Approach: iteratively adjust gain and apply true-peak limiting until
    LUFS is within tolerance of target and true peak is at or below ceiling.
    """
    sr = buffer.sample_rate
    data = buffer.data.copy().astype(np.float64)  # work in float64 for precision
    tolerance_lu = 1.0

    current_lufs = measure_lufs(AudioBuffer(data=data.astype(np.float32), sample_rate=sr))
    if np.isinf(current_lufs):
        logger.warning("Signal is silent, skipping limiter")
        return buffer

    # Step 1: Apply gain to reach target LUFS
    gain_db = target_lufs - current_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)
    data = data * gain_linear

    # Step 2: True-peak limiting — oversample, clamp, downsample
    ceiling_linear = 10.0 ** (ceiling_dbtp / 20.0)

    for i in range(max_iterations):
        # Apply true-peak limiting via 4x oversampling
        data = _true_peak_limit(data, ceiling_linear, sr)

        # Re-measure and adjust if needed
        result_buf = AudioBuffer(data=data.astype(np.float32), sample_rate=sr)
        result_lufs = measure_lufs(result_buf)

        if np.isinf(result_lufs):
            break

        if abs(result_lufs - target_lufs) <= tolerance_lu:
            logger.debug(
                "Limiter converged after %d iteration(s): LUFS=%.1f (target=%.1f)",
                i + 1,
                result_lufs,
                target_lufs,
            )
            break

        # Small correction gain to get closer to target
        correction_db = target_lufs - result_lufs
        correction_linear = 10.0 ** (correction_db / 20.0)
        data = data * correction_linear
    else:
        logger.warning(
            "Limiter did not converge after %d iterations: LUFS=%.1f (target=%.1f)",
            max_iterations,
            result_lufs,
            target_lufs,
        )

    return AudioBuffer(data=data.astype(np.float32), sample_rate=sr)


def _true_peak_limit(data: np.ndarray, ceiling_linear: float, sr: int) -> np.ndarray:
    """Apply true-peak limiting via 4x oversampling.

    1. Upsample 4x
    2. Clamp peaks at ceiling
    3. Downsample back to original rate
    """
    n_samples, n_channels = data.shape
    result = data.copy()

    for ch in range(n_channels):
        channel = data[:, ch]

        # 4x oversample
        upsampled = resample_poly(channel, 4, 1)

        # Find peaks exceeding ceiling
        peak = np.max(np.abs(upsampled))
        if peak > ceiling_linear:
            # Scale down the entire channel to bring peak to ceiling
            # This preserves waveform shape better than hard clipping
            scale = ceiling_linear / peak
            result[:, ch] = channel * scale

    return result
