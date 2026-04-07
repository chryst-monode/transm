"""De-esser — reduce sibilance energy in a configurable frequency band."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from transm.dsp.common import (
    envelope_follower,
    linear_to_db,
    time_to_coeff,
)
from transm.types import AudioBuffer


def deess(
    buffer: AudioBuffer,
    freq_low: float = 6000.0,
    freq_high: float = 9000.0,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 0.5,
    release_ms: float = 20.0,
) -> AudioBuffer:
    """Split-band de-esser.

    Algorithm
    ---------
    1. Extract sibilance band via zero-phase 4th-order Butterworth bandpass.
    2. Compute envelope of the extracted band.
    3. Where envelope exceeds threshold, compute compressor-style gain reduction.
    4. Apply gain reduction to the band only.
    5. Recombine: ``output = (original - band) + compressed_band``.

    Stereo handling: each channel is processed independently (band extraction
    and gain are per-channel) to preserve stereo imaging.
    """
    sr = buffer.sample_rate
    data = buffer.data.astype(np.float64)  # (samples, channels)
    n_channels = data.shape[1]

    # --- bandpass filter for sibilance band -----------------------------
    sos = butter(4, [freq_low, freq_high], btype="bandpass", fs=sr, output="sos")

    # --- envelope coefficients ------------------------------------------
    att = time_to_coeff(attack_ms, sr)
    rel = time_to_coeff(release_ms, sr)

    threshold_lin = 10.0 ** (threshold_db / 20.0)
    result = data.copy()

    for ch in range(n_channels):
        # 1. Extract the sibilance band (zero-phase)
        band = sosfiltfilt(sos, data[:, ch])

        # 2. Envelope of the band
        env = envelope_follower(band, att, rel)

        # 3. Gain reduction: compressor math on samples above threshold
        env_db = np.asarray(linear_to_db(env, floor_db=-120.0))
        over_db = env_db - threshold_db
        over_db = np.maximum(over_db, 0.0)

        # gain_reduction_db is negative (attenuation)
        gain_reduction_db = over_db * (1.0 / ratio - 1.0)

        # Convert to linear gain
        gain_linear = np.power(10.0, gain_reduction_db / 20.0)

        # 4. Apply gain to the band
        compressed_band = band * gain_linear

        # 5. Recombine: subtract original band, add compressed band
        result[:, ch] = data[:, ch] - band + compressed_band

    return AudioBuffer(data=result.astype(np.float32), sample_rate=sr)
