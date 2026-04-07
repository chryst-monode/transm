"""Quality assessment for separated stems.

Measures bleed, artifacts, and reconstruction accuracy to validate
that stem separation is good enough for per-stem processing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from transm.types import AudioBuffer, StemQAReport, StemSet

# STFT parameters for spectral analysis
_FFT_SIZE = 2048
_HOP_LENGTH = 512


def assess_stems(stems: StemSet, original: AudioBuffer) -> StemQAReport:
    """Assess quality of stem separation.

    Computes bleed scores, artifact scores, and reconstruction error
    for each stem relative to the original mix.

    Returns:
        StemQAReport with per-stem bleed and artifact scores,
        reconstruction error in dB, and any warnings.
    """
    warnings: list[str] = []
    bleed_scores: dict[str, float] = {}
    artifact_scores: dict[str, float] = {}

    stem_items = stems.items()

    for name, stem in stem_items:
        other_buffers = [buf for other_name, buf in stem_items if other_name != name]
        bleed = estimate_bleed(stem, other_buffers)
        bleed_scores[name] = bleed
        if bleed > 0.5:
            warnings.append(f"High bleed in {name} stem: {bleed:.2f}")

    for name, stem in stem_items:
        artifact = estimate_artifacts(stem)
        artifact_scores[name] = artifact
        if artifact > 0.5:
            warnings.append(f"High artifacts in {name} stem: {artifact:.2f}")

    reconstruction_error_db = check_reconstruction(stems, original)
    if reconstruction_error_db > -10.0:
        warnings.append(
            f"Poor reconstruction: {reconstruction_error_db:.1f} dB "
            "(expected < -20 dB)"
        )

    return StemQAReport(
        bleed_scores=bleed_scores,
        artifact_scores=artifact_scores,
        reconstruction_error_db=reconstruction_error_db,
        warnings=warnings,
    )


def estimate_bleed(stem: AudioBuffer, other_stems: list[AudioBuffer]) -> float:
    """Estimate bleed from other stems into this stem.

    Computes the STFT of the stem and the sum of all other stems,
    then measures normalized cross-correlation of magnitude spectra.

    Returns:
        Float from 0.0 (no bleed) to 1.0 (severe bleed).
    """
    stem_mono = _to_mono(stem.data)
    # Sum all other stems
    other_sum = np.zeros_like(stem_mono)
    for other in other_stems:
        other_mono = _to_mono(other.data)
        # Align lengths
        min_len = min(len(stem_mono), len(other_mono))
        other_sum[:min_len] += other_mono[:min_len]

    # Compute STFTs
    stem_spec = _stft_magnitude(stem_mono)
    other_spec = _stft_magnitude(other_sum[: len(stem_mono)])

    # Normalized cross-correlation of magnitude spectra
    correlation = _spectral_correlation(stem_spec, other_spec)
    return float(np.clip(correlation, 0.0, 1.0))


def estimate_artifacts(stem: AudioBuffer, threshold_db: float = -40.0) -> float:
    """Estimate separation artifacts in a stem.

    Measures spectral flux in quiet passages (below threshold_db).
    High spectral flux in quiet sections indicates 'watery' separation artifacts.

    Returns:
        Float from 0.0 (clean) to 1.0 (severe artifacts).
    """
    mono = _to_mono(stem.data)
    threshold_linear = 10.0 ** (threshold_db / 20.0)

    # Compute STFT
    spec = _stft_magnitude(mono)
    n_frames = spec.shape[1]

    if n_frames < 2:
        return 0.0

    # Compute per-frame energy
    frame_energy = np.mean(spec**2, axis=0)
    max_energy = np.max(frame_energy)
    if max_energy == 0:
        return 0.0

    # Normalize energy
    frame_energy_norm = frame_energy / max_energy

    # Find quiet frames (energy below threshold)
    quiet_mask = frame_energy_norm < (threshold_linear**2)
    quiet_indices = np.where(quiet_mask)[0]

    if len(quiet_indices) < 2:
        return 0.0

    # Compute spectral flux in quiet frames
    flux_values = []
    for i in range(len(quiet_indices) - 1):
        idx = quiet_indices[i]
        next_idx = quiet_indices[i + 1]
        if next_idx - idx == 1 and next_idx < n_frames:
            diff = spec[:, next_idx] - spec[:, idx]
            # Half-wave rectified flux (only increases)
            flux = np.sum(np.maximum(diff, 0.0) ** 2)
            flux_values.append(flux)

    if not flux_values:
        return 0.0

    mean_flux = float(np.mean(flux_values))
    # Normalize: compare quiet flux against overall spectral energy
    overall_mean_energy = float(np.mean(spec**2))
    if overall_mean_energy == 0:
        return 0.0

    # Scale so that artifact scores fall in a useful range
    artifact_ratio = mean_flux / overall_mean_energy
    # Clamp to [0, 1] with a scaling factor tuned empirically
    return float(np.clip(artifact_ratio * 5.0, 0.0, 1.0))


def check_reconstruction(stems: StemSet, original: AudioBuffer) -> float:
    """Check that sum(stems) approximates the original.

    Returns:
        RMS difference in dB. Good separation yields < -20 dB.
    """
    original_mono = _to_mono(original.data)

    # Sum all stems
    reconstructed = np.zeros_like(original_mono)
    for _name, stem in stems.items():
        stem_mono = _to_mono(stem.data)
        min_len = min(len(reconstructed), len(stem_mono))
        reconstructed[:min_len] += stem_mono[:min_len]

    # Align lengths
    min_len = min(len(original_mono), len(reconstructed))
    original_trimmed = original_mono[:min_len]
    reconstructed_trimmed = reconstructed[:min_len]

    # RMS of difference
    diff = original_trimmed - reconstructed_trimmed
    rms_diff = float(np.sqrt(np.mean(diff**2)))
    rms_original = float(np.sqrt(np.mean(original_trimmed**2)))

    if rms_original == 0:
        return 0.0 if rms_diff == 0 else -np.inf

    # Convert to dB ratio
    ratio = rms_diff / rms_original
    if ratio == 0:
        return -120.0  # effectively perfect reconstruction
    return float(20.0 * np.log10(ratio))


# --- Internal helpers ---


def _to_mono(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (samples, channels) to mono by averaging channels."""
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1).astype(np.float32)


def _stft_magnitude(signal: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute magnitude STFT.

    Returns:
        2D array of shape (n_freq_bins, n_frames).
    """
    # Window
    window = np.hanning(_FFT_SIZE).astype(np.float32)

    n_frames = max(1, (len(signal) - _FFT_SIZE) // _HOP_LENGTH + 1)
    n_bins = _FFT_SIZE // 2 + 1
    result = np.zeros((n_bins, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * _HOP_LENGTH
        end = start + _FFT_SIZE
        if end > len(signal):
            break
        frame = signal[start:end] * window
        spectrum = np.fft.rfft(frame)
        result[:, i] = np.abs(spectrum).astype(np.float32)

    return result


def _spectral_correlation(
    spec_a: NDArray[np.float32],
    spec_b: NDArray[np.float32],
) -> float:
    """Compute normalized cross-correlation between two magnitude spectrograms.

    Returns a value in [0, 1] where 1 means perfectly correlated.
    """
    # Align frame counts
    min_frames = min(spec_a.shape[1], spec_b.shape[1])
    a = spec_a[:, :min_frames].ravel()
    b = spec_b[:, :min_frames].ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    correlation = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, correlation)
