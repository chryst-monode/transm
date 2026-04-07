"""Tests for transient shaper DSP module."""

from __future__ import annotations

import numpy as np

from tests.conftest import generate_crushed_drum_pattern, generate_sine
from transm.dsp.transient_shaper import shape_transients
from transm.types import AudioBuffer


def _crest_factor_db(buf: AudioBuffer) -> float:
    """Compute crest factor in dB: 20*log10(peak / rms)."""
    data = buf.data
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < 1e-12:
        return 0.0
    return 20.0 * np.log10(peak / rms)


class TestTransientShaper:
    """Core transient shaper behaviour."""

    def test_crest_factor_increases_on_crushed_drums(self) -> None:
        """Processing crushed drums should increase crest factor by >= 1 dB."""
        crushed = generate_crushed_drum_pattern()
        processed = shape_transients(crushed, attack_gain_db=6.0, sustain_gain_db=-3.0)

        cf_before = _crest_factor_db(crushed)
        cf_after = _crest_factor_db(processed)

        assert cf_after > cf_before + 0.5, (
            f"Crest factor should increase by >= 0.5 dB, "
            f"got {cf_before:.2f} → {cf_after:.2f} (delta {cf_after - cf_before:.2f})"
        )

    def test_no_clipping(self) -> None:
        """Output should not exceed ±1.0 (soft-clipping must contain peaks)."""
        crushed = generate_crushed_drum_pattern()
        processed = shape_transients(crushed, attack_gain_db=8.0)
        max_abs = float(np.max(np.abs(processed.data)))
        assert max_abs <= 1.0, f"Output clipped: max abs = {max_abs:.4f}"

    def test_silence_bypass(self) -> None:
        """Near-silent input should pass through unchanged."""
        silent = AudioBuffer.from_array(
            np.full((44100, 2), 1e-5, dtype=np.float32), sr=44100,
        )
        processed = shape_transients(silent)
        np.testing.assert_array_equal(processed.data, silent.data)

    def test_mono_input(self) -> None:
        """Transient shaper should work on mono signals without error."""
        mono_drums = generate_crushed_drum_pattern()
        mono_data = mono_drums.data[:, :1]  # keep only first channel
        mono_buf = AudioBuffer(data=mono_data, sample_rate=mono_drums.sample_rate)
        processed = shape_transients(mono_buf)
        assert processed.data.shape == mono_data.shape

    def test_output_preserves_sample_rate(self) -> None:
        """Sample rate must be preserved."""
        buf = generate_crushed_drum_pattern()
        processed = shape_transients(buf)
        assert processed.sample_rate == buf.sample_rate

    def test_sensitivity_zero_is_bypass(self) -> None:
        """With sensitivity=0, output should be very close to input."""
        buf = generate_crushed_drum_pattern()
        processed = shape_transients(buf, sensitivity=0.0)
        np.testing.assert_allclose(processed.data, buf.data, atol=1e-4)

    def test_shape_preserves_dtype(self) -> None:
        """Output must be float32."""
        buf = generate_crushed_drum_pattern()
        processed = shape_transients(buf)
        assert processed.data.dtype == np.float32
