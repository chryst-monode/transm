"""Tests for stem remixing — unit tests, no model needed."""

from __future__ import annotations

import numpy as np

from transm.remix import check_polarity, remix_stems
from transm.types import AudioBuffer, StemSet

SR = 44100
DURATION = 1.0


def _make_sine(freq_hz: float, amplitude: float = 0.2) -> AudioBuffer:
    """Create a stereo sine wave AudioBuffer."""
    n = int(SR * DURATION)
    t = np.arange(n, dtype=np.float32) / SR
    mono = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    data = np.column_stack([mono, mono])
    return AudioBuffer(data=data, sample_rate=SR)


def _make_stem_set(
    freqs: tuple[float, ...] = (200.0, 400.0, 800.0, 1600.0),
    amplitude: float = 0.2,
) -> tuple[StemSet, AudioBuffer]:
    """Create a StemSet of 4 sine waves and the expected original (sum)."""
    vocals = _make_sine(freqs[0], amplitude)
    drums = _make_sine(freqs[1], amplitude)
    bass = _make_sine(freqs[2], amplitude)
    other = _make_sine(freqs[3], amplitude)
    stems = StemSet(vocals=vocals, drums=drums, bass=bass, other=other)

    # Build the "original" as the sum of all stems
    original_data = vocals.data + drums.data + bass.data + other.data
    original = AudioBuffer(data=original_data.astype(np.float32), sample_rate=SR)

    return stems, original


class TestRemixPreservesShape:
    def test_remix_preserves_shape(self) -> None:
        """Remixed output should have same shape and sample rate as original."""
        stems, original = _make_stem_set()
        result = remix_stems(stems, original)

        assert result.data.shape == original.data.shape
        assert result.sample_rate == original.sample_rate

    def test_remix_preserves_sample_rate(self) -> None:
        stems, original = _make_stem_set()
        result = remix_stems(stems, original)
        assert result.sample_rate == SR


class TestRemixSumsCorrectly:
    def test_remix_contains_all_frequencies(self) -> None:
        """All 4 sine frequencies should be present in the remix (FFT check)."""
        freqs = (200.0, 400.0, 800.0, 1600.0)
        stems, original = _make_stem_set(freqs)
        result = remix_stems(stems, original)

        # FFT of mono-summed signal
        mono = np.mean(result.data, axis=1)
        fft_mag = np.abs(np.fft.rfft(mono))
        fft_freqs = np.fft.rfftfreq(len(mono), 1.0 / SR)

        # Check each target frequency has a peak (within 5 Hz tolerance)
        for target in freqs:
            idx = np.argmin(np.abs(fft_freqs - target))
            # The bin at target freq should be significantly above neighbors
            local_region = fft_mag[max(0, idx - 20) : idx + 20]
            assert fft_mag[idx] > np.median(local_region) * 2, (
                f"Expected peak at {target} Hz not found in remix FFT"
            )


class TestRemixHeadroomManagement:
    def test_remix_headroom_management(self) -> None:
        """When 4 loud stems sum above 1.0, output peak should be <= 0.95."""
        # Each stem at 0.5 amplitude -> sum peaks at ~2.0
        stems, original = _make_stem_set(amplitude=0.5)
        result = remix_stems(stems, original)

        peak = float(np.max(np.abs(result.data)))
        assert peak <= 0.95 + 1e-6, f"Peak {peak} exceeds 0.95 headroom limit"


class TestCheckPolarity:
    def test_check_polarity_in_phase(self) -> None:
        """Two identical signals should be in-phase."""
        buf = _make_sine(440.0)
        assert check_polarity(buf, buf) is True

    def test_check_polarity_inverted(self) -> None:
        """Inverted signal should be out-of-phase."""
        buf = _make_sine(440.0)
        inverted = AudioBuffer(data=-buf.data, sample_rate=SR)
        assert check_polarity(inverted, buf) is False

    def test_check_polarity_silence(self) -> None:
        """Silent signal defaults to in-phase."""
        buf = _make_sine(440.0)
        silent = AudioBuffer(
            data=np.zeros_like(buf.data), sample_rate=SR
        )
        assert check_polarity(silent, buf) is True
