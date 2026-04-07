"""Tests for de-esser DSP module."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from tests.conftest import generate_sibilant_vocal, generate_sine
from transm.dsp.deesser import deess
from transm.types import AudioBuffer


def _band_energy(data_1d: np.ndarray, low: float, high: float, sr: int) -> float:
    """RMS energy in a frequency band via bandpass filtering."""
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, data_1d)
    return float(np.sqrt(np.mean(filtered ** 2)))


class TestDeesser:
    """Core de-esser behaviour."""

    def test_sibilance_energy_reduced(self) -> None:
        """Energy in the 6-9 kHz band should decrease by at least 30%."""
        buf = generate_sibilant_vocal()
        processed = deess(buf)

        sr = buf.sample_rate
        # Use first channel for measurement
        before_energy = _band_energy(buf.data[:, 0], 6000.0, 9000.0, sr)
        after_energy = _band_energy(processed.data[:, 0], 6000.0, 9000.0, sr)

        reduction_pct = 1.0 - (after_energy / before_energy)
        assert reduction_pct >= 0.30, (
            f"Sibilance band energy should reduce by >= 30%, "
            f"got {reduction_pct * 100:.1f}%"
        )

    def test_non_sibilant_band_preserved(self) -> None:
        """Energy outside the sibilance band (100-5000 Hz) should change < 5%."""
        buf = generate_sibilant_vocal()
        processed = deess(buf)

        sr = buf.sample_rate
        before_energy = _band_energy(buf.data[:, 0], 100.0, 5000.0, sr)
        after_energy = _band_energy(processed.data[:, 0], 100.0, 5000.0, sr)

        change_pct = abs(after_energy - before_energy) / max(before_energy, 1e-12)
        assert change_pct < 0.05, (
            f"Non-sibilant band energy should change < 5%, "
            f"got {change_pct * 100:.1f}%"
        )

    def test_output_shape_preserved(self) -> None:
        """Output shape and sample rate must match input."""
        buf = generate_sibilant_vocal()
        processed = deess(buf)
        assert processed.data.shape == buf.data.shape
        assert processed.sample_rate == buf.sample_rate

    def test_mono_input(self) -> None:
        """De-esser works on mono signals."""
        buf = generate_sibilant_vocal()
        mono = AudioBuffer(data=buf.data[:, :1], sample_rate=buf.sample_rate)
        processed = deess(mono)
        assert processed.data.shape == mono.data.shape

    def test_output_dtype(self) -> None:
        """Output must be float32."""
        buf = generate_sibilant_vocal()
        processed = deess(buf)
        assert processed.data.dtype == np.float32

    def test_below_threshold_bypass(self) -> None:
        """Very quiet sibilant signal below threshold should be minimally affected."""
        # Generate a very quiet sibilant vocal
        buf = generate_sibilant_vocal()
        quiet = AudioBuffer(
            data=(buf.data * 0.001).astype(np.float32),
            sample_rate=buf.sample_rate,
        )
        processed = deess(quiet, threshold_db=-10.0)
        # Should be mostly unchanged since the signal is well below threshold
        np.testing.assert_allclose(processed.data, quiet.data, atol=1e-5)
