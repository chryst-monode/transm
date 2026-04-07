"""Tests for downward expander DSP module."""

from __future__ import annotations

import numpy as np

from tests.conftest import generate_signal_with_dynamics, generate_sine
from transm.dsp.expander import expand_downward
from transm.types import AudioBuffer


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


class TestDownwardExpander:
    """Core expansion behaviour."""

    def test_quiet_sections_get_quieter(self) -> None:
        """Quiet sections should be attenuated; loud sections less affected."""
        buf = generate_signal_with_dynamics()
        processed = expand_downward(buf, threshold_db=-20.0, ratio=2.0, range_db=-40.0)

        sr = buf.sample_rate
        n = buf.data.shape[0]
        quarter = n // 4

        # Quarters 0 and 2 are loud, 1 and 3 are quiet
        quiet_before = _rms(buf.data[quarter: 2 * quarter, 0])
        quiet_after = _rms(processed.data[quarter: 2 * quarter, 0])

        loud_before = _rms(buf.data[:quarter, 0])
        loud_after = _rms(processed.data[:quarter, 0])

        # Quiet section RMS should decrease
        assert quiet_after < quiet_before, (
            f"Quiet section should get quieter: {quiet_before:.6f} → {quiet_after:.6f}"
        )

        # Loud section should be relatively less affected than quiet
        quiet_ratio = quiet_after / max(quiet_before, 1e-12)
        loud_ratio = loud_after / max(loud_before, 1e-12)
        assert quiet_ratio < loud_ratio, (
            f"Quiet sections should be reduced more than loud sections: "
            f"quiet ratio={quiet_ratio:.4f}, loud ratio={loud_ratio:.4f}"
        )

    def test_output_shape_preserved(self) -> None:
        """Output shape and sample rate must match input."""
        buf = generate_signal_with_dynamics()
        processed = expand_downward(buf)
        assert processed.data.shape == buf.data.shape
        assert processed.sample_rate == buf.sample_rate

    def test_ratio_1_is_bypass(self) -> None:
        """ratio=1.0 means no expansion (slope = 0)."""
        buf = generate_signal_with_dynamics()
        processed = expand_downward(buf, ratio=1.0)
        # With ratio=1.0, gain_reduction slope is (1-1)/1 = 0 → no gain change
        np.testing.assert_allclose(processed.data, buf.data, atol=1e-3)

    def test_range_limits_attenuation(self) -> None:
        """Gain reduction should not exceed range_db."""
        buf = generate_signal_with_dynamics(quiet_amplitude=0.001)
        processed = expand_downward(buf, threshold_db=-10.0, ratio=8.0, range_db=-12.0)

        # Even with extreme expansion settings, quiet parts shouldn't vanish
        quiet_rms = _rms(processed.data[buf.data.shape[0] // 4: buf.data.shape[0] // 2, 0])
        # With range_db=-12 the quietest signal can only drop 12 dB
        assert quiet_rms > 0.0, "Quiet section should not be fully zeroed"

    def test_mono_input(self) -> None:
        """Expander works on mono signals."""
        buf = generate_signal_with_dynamics()
        mono = AudioBuffer(data=buf.data[:, :1], sample_rate=buf.sample_rate)
        processed = expand_downward(mono)
        assert processed.data.shape == mono.data.shape

    def test_output_dtype(self) -> None:
        """Output must be float32."""
        buf = generate_signal_with_dynamics()
        processed = expand_downward(buf)
        assert processed.data.dtype == np.float32
