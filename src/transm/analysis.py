"""Audio analysis metrics — pure functions, no side effects.

Implements ITU-R BS.1770-4 loudness measurement, true peak detection,
spectral analysis, and clipping detection.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm
from numpy.typing import NDArray
from scipy import signal

from transm.types import AudioBuffer, Metrics, MetricsDelta


def compute_metrics(buffer: AudioBuffer) -> Metrics:
    """Compute all analysis metrics for an audio buffer."""
    lufs = measure_lufs(buffer)
    lra = measure_lra(buffer)
    tp = measure_true_peak(buffer)
    plr = measure_plr(buffer)
    cf = measure_crest_factor(buffer)
    sc = measure_spectral_centroid(buffer)
    clip = measure_clipping_percent(buffer)
    tilt = measure_spectral_tilt(buffer)

    return Metrics(
        lufs_integrated=lufs,
        loudness_range=lra,
        true_peak_dbtp=tp,
        peak_to_loudness_ratio=plr,
        crest_factor_db=cf,
        spectral_centroid_hz=sc,
        clipping_percent=clip,
        spectral_tilt=tilt,
    )


def measure_lufs(buffer: AudioBuffer) -> float:
    """ITU-R BS.1770-4 integrated loudness via pyloudnorm."""
    meter = pyloudnorm.Meter(buffer.sample_rate, block_size=0.400)
    return float(meter.integrated_loudness(buffer.data))


def measure_lra(buffer: AudioBuffer) -> float:
    """Loudness Range in LU via short-term loudness distribution.

    Uses 3-second windows with 1-second hop, then computes the difference
    between the 95th and 10th percentile of gated short-term loudness values.
    """
    meter = pyloudnorm.Meter(buffer.sample_rate, block_size=0.400)
    window_samples = int(3.0 * buffer.sample_rate)
    hop_samples = int(1.0 * buffer.sample_rate)
    n_samples = buffer.num_samples

    # Need at least one full window
    if n_samples < window_samples:
        # Fall back: treat entire buffer as one block
        lufs = measure_lufs(buffer)
        return 0.0 if np.isinf(lufs) else 0.0

    short_term: list[float] = []
    start = 0
    while start + window_samples <= n_samples:
        chunk_data = buffer.data[start : start + window_samples]
        chunk = AudioBuffer(data=chunk_data, sample_rate=buffer.sample_rate)
        st_lufs = float(meter.integrated_loudness(chunk.data))
        if not np.isinf(st_lufs):
            short_term.append(st_lufs)
        start += hop_samples

    if len(short_term) < 2:
        return 0.0

    arr = np.array(short_term)

    # Absolute gate at -70 LUFS
    gated = arr[arr > -70.0]
    if len(gated) < 2:
        return 0.0

    p95 = float(np.percentile(gated, 95))
    p10 = float(np.percentile(gated, 10))
    return p95 - p10


def measure_true_peak(buffer: AudioBuffer) -> float:
    """True peak in dBTP (4x oversampled per ITU-R BS.1770).

    Uses scipy.signal.resample_poly for 4x oversampling, then measures
    the absolute maximum across all channels.
    """
    data = buffer.data  # (samples, channels)
    max_peak = 0.0

    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        # 4x oversample
        oversampled = signal.resample_poly(channel_data, 4, 1)
        ch_peak = float(np.max(np.abs(oversampled)))
        max_peak = max(max_peak, ch_peak)

    if max_peak == 0.0:
        return -np.inf

    return float(20.0 * np.log10(max_peak))


def measure_plr(buffer: AudioBuffer) -> float:
    """Peak-to-Loudness Ratio = true_peak_dBTP - LUFS."""
    tp = measure_true_peak(buffer)
    lufs = measure_lufs(buffer)
    if np.isinf(tp) or np.isinf(lufs):
        return 0.0
    return tp - lufs


def measure_crest_factor(buffer: AudioBuffer) -> float:
    """Crest factor = 20*log10(peak / rms) in dB."""
    data = buffer.data
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))

    if rms == 0.0 or peak == 0.0:
        return 0.0

    return float(20.0 * np.log10(peak / rms))


def measure_spectral_centroid(buffer: AudioBuffer) -> float:
    """Weighted mean frequency via librosa.feature.spectral_centroid."""
    import librosa

    # librosa expects (channels, samples) or mono (samples,)
    # Use the mean across channels for a single centroid value
    mono = np.mean(buffer.data, axis=1)
    centroid = librosa.feature.spectral_centroid(
        y=mono, sr=buffer.sample_rate
    )
    return float(np.mean(centroid))


def measure_clipping_percent(buffer: AudioBuffer, threshold: float = 0.99) -> float:
    """Percentage of samples at or above threshold of full scale."""
    data = buffer.data
    total_samples = data.size  # all samples across all channels
    clipped = int(np.sum(np.abs(data) >= threshold))
    return float(clipped / total_samples * 100.0)


def measure_spectral_tilt(buffer: AudioBuffer) -> float:
    """Spectral tilt in dB/octave.

    Computes the PSD via scipy.signal.welch, converts to dB, then fits a
    linear regression on log2(frequency) vs dB. The slope is dB/octave.
    """
    mono = np.mean(buffer.data, axis=1)
    freqs, psd = signal.welch(mono, fs=buffer.sample_rate, nperseg=min(4096, len(mono)))

    # Skip DC component (freq=0) to avoid log2(0)
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]

    if len(freqs) < 2:
        return 0.0

    # Avoid log of zero PSD values
    psd_safe = np.maximum(psd, 1e-30)
    psd_db = 10.0 * np.log10(psd_safe)
    log2_freqs = np.log2(freqs)

    # Linear regression: slope = dB/octave
    # Using numpy polyfit degree 1
    coeffs = np.polyfit(log2_freqs, psd_db, 1)
    slope = float(coeffs[0])

    return slope


def compute_delta(before: Metrics, after: Metrics) -> MetricsDelta:
    """Create a MetricsDelta from before/after Metrics."""
    return MetricsDelta(before=before, after=after)
