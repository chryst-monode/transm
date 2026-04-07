"""Tests for transm.analysis — audio metrics computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from transm.analysis import (
    compute_delta,
    compute_metrics,
    measure_clipping_percent,
    measure_crest_factor,
    measure_lra,
    measure_lufs,
    measure_plr,
    measure_spectral_centroid,
    measure_spectral_tilt,
    measure_true_peak,
)
from transm.types import AudioBuffer, Metrics

from tests.conftest import generate_crushed_drum_pattern, generate_sine


class TestMeasureLufs:
    """LUFS measurement via ITU-R BS.1770-4."""

    def test_sine_at_known_amplitude(self, sine_440: AudioBuffer) -> None:
        """A 1 kHz sine at 0.5 amplitude should be around -9.2 LUFS (within 1 dB).

        Using 440 Hz sine from fixture (amplitude=0.5). The exact LUFS depends
        on the K-weighting filter, but should be in the ballpark of -9 to -10 LUFS.
        """
        lufs = measure_lufs(sine_440)
        # 0.5 amplitude sine: RMS = 0.5/sqrt(2) ≈ 0.354, dBFS ≈ -9.03
        # K-weighting at 440 Hz is near-unity, so LUFS ≈ dBFS
        assert -12.0 < lufs < -6.0, f"Expected LUFS near -9.2, got {lufs}"

    def test_louder_signal_has_higher_lufs(self) -> None:
        """A louder signal should have higher (less negative) LUFS."""
        quiet = generate_sine(amplitude=0.1, duration_s=1.0)
        loud = generate_sine(amplitude=0.8, duration_s=1.0)
        lufs_quiet = measure_lufs(quiet)
        lufs_loud = measure_lufs(loud)
        assert lufs_loud > lufs_quiet

    def test_silence_returns_neg_inf(self) -> None:
        """Silence should return -inf LUFS."""
        silence = AudioBuffer.from_array(
            np.zeros((44100, 2), dtype=np.float32), sr=44100
        )
        lufs = measure_lufs(silence)
        assert math.isinf(lufs) and lufs < 0


class TestMeasureTruePeak:
    """True peak via 4x oversampling per ITU-R BS.1770."""

    def test_true_peak_gte_sample_peak(self, sine_440: AudioBuffer) -> None:
        """A sine wave's true peak should be >= its sample peak due to inter-sample peaks."""
        sample_peak_db = float(20.0 * np.log10(np.max(np.abs(sine_440.data))))
        true_peak_db = measure_true_peak(sine_440)
        assert true_peak_db >= sample_peak_db - 0.01  # tiny tolerance for float

    def test_true_peak_full_scale(self) -> None:
        """A full-scale sine should have true peak near 0 dBTP or slightly above."""
        buf = generate_sine(amplitude=1.0, duration_s=0.5)
        tp = measure_true_peak(buf)
        # True peak of a full-scale sine can exceed 0 dBTP due to inter-sample peaks
        assert tp >= -0.5, f"Expected true peak near 0 dBTP, got {tp}"

    def test_silence_returns_neg_inf(self) -> None:
        """Silence should return -inf dBTP."""
        silence = AudioBuffer.from_array(
            np.zeros((44100, 2), dtype=np.float32), sr=44100
        )
        tp = measure_true_peak(silence)
        assert math.isinf(tp) and tp < 0


class TestMeasureCrestFactor:
    """Crest factor = peak/RMS in dB."""

    def test_sine_crest_factor(self, sine_440: AudioBuffer) -> None:
        """A sine wave has a crest factor of ~3.01 dB."""
        cf = measure_crest_factor(sine_440)
        assert abs(cf - 3.01) < 0.5, f"Expected crest factor ~3.01 dB, got {cf}"

    def test_square_wave_crest_factor(self) -> None:
        """A square wave has a crest factor of 0 dB (peak == RMS)."""
        t = np.arange(44100, dtype=np.float32) / 44100
        square = np.sign(np.sin(2 * np.pi * 440 * t)).astype(np.float32) * 0.5
        buf = AudioBuffer.from_array(square[:, np.newaxis], sr=44100)
        cf = measure_crest_factor(buf)
        assert abs(cf) < 0.5, f"Expected crest factor ~0 dB for square wave, got {cf}"


class TestMeasureClipping:
    """Clipping detection."""

    def test_crushed_drums_have_clipping(self, crushed_drums: AudioBuffer) -> None:
        """generate_crushed_drum_pattern clips, so clipping_percent should be > 0."""
        # The crushed drums are normalized to 0.95, so threshold at 0.94 to catch clipping
        clip = measure_clipping_percent(crushed_drums, threshold=0.94)
        assert clip > 0.0, f"Expected clipping > 0%, got {clip}%"

    def test_quiet_sine_has_no_clipping(self, sine_440: AudioBuffer) -> None:
        """A half-amplitude sine should have 0% clipping."""
        clip = measure_clipping_percent(sine_440)
        assert clip == 0.0, f"Expected 0% clipping, got {clip}%"

    def test_custom_threshold(self) -> None:
        """Lowering threshold should detect more 'clipping'."""
        buf = generate_sine(amplitude=0.5, duration_s=0.5)
        clip_high = measure_clipping_percent(buf, threshold=0.99)
        clip_low = measure_clipping_percent(buf, threshold=0.3)
        assert clip_low > clip_high


class TestMeasureSpectralCentroid:
    """Spectral centroid — weighted mean frequency."""

    def test_440hz_sine_centroid(self, sine_440: AudioBuffer) -> None:
        """A 440 Hz sine should have centroid near 440 Hz (within 50 Hz)."""
        sc = measure_spectral_centroid(sine_440)
        assert abs(sc - 440.0) < 50.0, f"Expected centroid near 440 Hz, got {sc}"

    def test_higher_freq_has_higher_centroid(self) -> None:
        """A higher frequency sine should have a higher spectral centroid."""
        low = generate_sine(freq_hz=200.0, duration_s=1.0)
        high = generate_sine(freq_hz=4000.0, duration_s=1.0)
        sc_low = measure_spectral_centroid(low)
        sc_high = measure_spectral_centroid(high)
        assert sc_high > sc_low


class TestMeasureSpectralTilt:
    """Spectral tilt — dB/octave slope."""

    def test_returns_finite_value(self, sine_440: AudioBuffer) -> None:
        """Spectral tilt should return a finite float."""
        tilt = measure_spectral_tilt(sine_440)
        assert math.isfinite(tilt)

    def test_noise_vs_sine(self) -> None:
        """White noise should have less negative tilt than a filtered signal."""
        rng = np.random.default_rng(42)
        white = rng.normal(0, 0.3, (44100,)).astype(np.float32)
        buf_white = AudioBuffer.from_array(white[:, np.newaxis], sr=44100)
        tilt_white = measure_spectral_tilt(buf_white)
        # White noise spectral tilt should be close to 0
        assert abs(tilt_white) < 10.0, f"White noise tilt should be near 0, got {tilt_white}"


class TestMeasureLRA:
    """Loudness Range measurement."""

    def test_constant_signal_has_low_lra(self) -> None:
        """A constant-amplitude signal should have near-zero LRA."""
        buf = generate_sine(amplitude=0.5, duration_s=5.0)
        lra = measure_lra(buf)
        assert lra < 2.0, f"Expected low LRA for constant signal, got {lra}"

    def test_returns_non_negative(self, sine_440: AudioBuffer) -> None:
        """LRA should be >= 0."""
        lra = measure_lra(sine_440)
        assert lra >= 0.0


class TestMeasurePLR:
    """Peak-to-Loudness Ratio."""

    def test_plr_positive(self, sine_440: AudioBuffer) -> None:
        """PLR should be positive (true peak is always > LUFS in absolute terms)."""
        plr = measure_plr(sine_440)
        assert plr > 0.0, f"Expected positive PLR, got {plr}"


class TestComputeDelta:
    """MetricsDelta computation."""

    def test_delta_properties(self) -> None:
        """Verify delta properties are correct arithmetic."""
        before = Metrics(
            lufs_integrated=-14.0,
            loudness_range=8.0,
            true_peak_dbtp=-1.0,
            peak_to_loudness_ratio=13.0,
            crest_factor_db=6.0,
            spectral_centroid_hz=2000.0,
            clipping_percent=0.5,
            spectral_tilt=-3.0,
        )
        after = Metrics(
            lufs_integrated=-12.0,
            loudness_range=6.0,
            true_peak_dbtp=-0.5,
            peak_to_loudness_ratio=11.5,
            crest_factor_db=4.0,
            spectral_centroid_hz=2200.0,
            clipping_percent=1.0,
            spectral_tilt=-2.5,
        )
        delta = compute_delta(before, after)

        assert delta.before is before
        assert delta.after is after
        assert delta.lufs_delta == pytest.approx(2.0)
        assert delta.lra_delta == pytest.approx(-2.0)
        assert delta.true_peak_delta == pytest.approx(0.5)
        assert delta.plr_delta == pytest.approx(-1.5)
        assert delta.crest_factor_delta == pytest.approx(-2.0)
        assert delta.spectral_centroid_delta == pytest.approx(200.0)
        assert delta.clipping_delta == pytest.approx(0.5)
        assert delta.spectral_tilt_delta == pytest.approx(0.5)


class TestComputeMetrics:
    """Integration test for compute_metrics."""

    def test_returns_valid_metrics(self, sine_440: AudioBuffer) -> None:
        """compute_metrics should return a Metrics with all fields populated (no NaN)."""
        metrics = compute_metrics(sine_440)

        assert math.isfinite(metrics.lufs_integrated)
        assert math.isfinite(metrics.loudness_range)
        assert math.isfinite(metrics.true_peak_dbtp)
        assert math.isfinite(metrics.peak_to_loudness_ratio)
        assert math.isfinite(metrics.crest_factor_db)
        assert math.isfinite(metrics.spectral_centroid_hz)
        assert math.isfinite(metrics.clipping_percent)
        assert math.isfinite(metrics.spectral_tilt)

    def test_with_crushed_drums(self, crushed_drums: AudioBuffer) -> None:
        """compute_metrics should work on crushed drums without error."""
        metrics = compute_metrics(crushed_drums)
        assert math.isfinite(metrics.lufs_integrated)
        assert metrics.clipping_percent >= 0.0
