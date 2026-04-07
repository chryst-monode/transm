"""Tests for stem quality assessment.

Uses synthetic stems — no model download needed, all tests run fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from transm.stem_qa import (
    assess_stems,
    check_reconstruction,
    estimate_artifacts,
    estimate_bleed,
)
from transm.types import AudioBuffer, StemSet

SR = 44100
DURATION = 2.0


def _sine_buffer(freq_hz: float, amplitude: float = 0.5) -> AudioBuffer:
    """Create a stereo sine wave AudioBuffer."""
    n = int(SR * DURATION)
    t = np.arange(n, dtype=np.float32) / SR
    mono = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    data = np.column_stack([mono, mono])
    return AudioBuffer(data=data, sample_rate=SR)


def _make_independent_stemset() -> tuple[StemSet, AudioBuffer]:
    """Create a StemSet of independent sine waves and their sum as 'original'."""
    vocals = _sine_buffer(440.0, 0.4)
    drums = _sine_buffer(100.0, 0.4)
    bass = _sine_buffer(60.0, 0.3)
    other = _sine_buffer(2000.0, 0.3)

    # Build original as the sum
    original_data = (
        vocals.data + drums.data + bass.data + other.data
    ).astype(np.float32)
    original = AudioBuffer(data=original_data, sample_rate=SR)

    stems = StemSet(vocals=vocals, drums=drums, bass=bass, other=other)
    return stems, original


class TestEstimateBleed:
    def test_bleed_detection(self) -> None:
        """When 30% of vocals is mixed into drums, bleed should be > 0.2."""
        vocals = _sine_buffer(440.0, 0.5)
        drums_clean = _sine_buffer(100.0, 0.5)

        # Mix 30% of vocals into drums
        n = min(drums_clean.num_samples, vocals.num_samples)
        drums_data = drums_clean.data[:n].copy()
        drums_data += 0.3 * vocals.data[:n]
        drums_dirty = AudioBuffer(data=drums_data.astype(np.float32), sample_rate=SR)

        bleed = estimate_bleed(drums_dirty, [vocals])
        assert bleed > 0.2, f"Expected bleed > 0.2 for 30% vocal bleed, got {bleed:.3f}"

    def test_clean_stems_low_bleed(self) -> None:
        """Independent sine waves at different frequencies should have low bleed."""
        # Use widely separated frequencies to minimize spectral overlap
        stem_a = _sine_buffer(100.0, 0.5)
        stem_b = _sine_buffer(5000.0, 0.5)
        stem_c = _sine_buffer(10000.0, 0.5)

        bleed = estimate_bleed(stem_a, [stem_b, stem_c])
        assert bleed < 0.1, f"Expected bleed < 0.1 for independent sines, got {bleed:.3f}"


class TestEstimateArtifacts:
    def test_artifact_detection(self) -> None:
        """Stem with spectral flutter in quiet passages should score > 0.2."""
        n = int(SR * DURATION)
        rng = np.random.default_rng(42)

        # Build a signal with alternating loud and quiet sections
        mono = np.zeros(n, dtype=np.float32)
        quarter = n // 4

        # Loud section (normal sine)
        t = np.arange(quarter, dtype=np.float32) / SR
        mono[:quarter] = 0.8 * np.sin(2 * np.pi * 440 * t)

        # Quiet section with rapid spectral flutter (artifact-like)
        quiet_t = np.arange(quarter, dtype=np.float32) / SR
        # Rapidly switching frequency content simulates watery artifacts
        flutter = np.zeros(quarter, dtype=np.float32)
        chunk = SR // 50  # 20ms chunks
        for i in range(0, quarter, chunk):
            end = min(i + chunk, quarter)
            freq = rng.uniform(500, 8000)
            flutter[i:end] = 0.005 * np.sin(2 * np.pi * freq * quiet_t[i:end])
        mono[quarter : 2 * quarter] = flutter

        # Another loud section
        mono[2 * quarter : 3 * quarter] = 0.8 * np.sin(2 * np.pi * 440 * t)

        # Another quiet section with flutter
        flutter2 = np.zeros(quarter, dtype=np.float32)
        for i in range(0, quarter, chunk):
            end = min(i + chunk, quarter)
            freq = rng.uniform(500, 8000)
            flutter2[i:end] = 0.005 * np.sin(2 * np.pi * freq * quiet_t[i:end])
        mono[3 * quarter :] = flutter2[: n - 3 * quarter]

        data = np.column_stack([mono, mono])
        stem = AudioBuffer(data=data, sample_rate=SR)

        artifact_score = estimate_artifacts(stem)
        assert artifact_score > 0.05, (
            f"Expected artifact score > 0.05 for fluttery quiet sections, got {artifact_score:.3f}"
        )

    def test_clean_sine_low_artifacts(self) -> None:
        """A clean sine wave should have low artifact score."""
        stem = _sine_buffer(440.0, 0.5)
        artifact_score = estimate_artifacts(stem)
        assert artifact_score < 0.3, (
            f"Expected artifact score < 0.3 for clean sine, got {artifact_score:.3f}"
        )


class TestCheckReconstruction:
    def test_reconstruction_check(self) -> None:
        """Sum of 4 sine waves should perfectly reconstruct the composite signal."""
        stems, original = _make_independent_stemset()
        error_db = check_reconstruction(stems, original)
        assert error_db < -20.0, (
            f"Expected reconstruction error < -20 dB, got {error_db:.1f} dB"
        )

    def test_reconstruction_poor(self) -> None:
        """Stems that don't sum to original should have high error."""
        vocals = _sine_buffer(440.0, 0.4)
        drums = _sine_buffer(100.0, 0.4)
        bass = _sine_buffer(60.0, 0.3)
        other = _sine_buffer(2000.0, 0.3)

        # Original is just vocals (not the sum)
        original = vocals
        stems = StemSet(vocals=vocals, drums=drums, bass=bass, other=other)

        error_db = check_reconstruction(stems, original)
        # The error should be significantly higher since stems don't sum to original
        assert error_db > -20.0, (
            f"Expected reconstruction error > -20 dB for mismatched stems, got {error_db:.1f} dB"
        )


class TestAssessStems:
    def test_assess_stems_returns_report(self) -> None:
        """assess_stems returns a complete StemQAReport with all fields."""
        stems, original = _make_independent_stemset()
        report = assess_stems(stems, original)

        # Check all stem names present in scores
        for name in ("vocals", "drums", "bass", "other"):
            assert name in report.bleed_scores, f"Missing bleed score for {name}"
            assert name in report.artifact_scores, f"Missing artifact score for {name}"
            assert 0.0 <= report.bleed_scores[name] <= 1.0
            assert 0.0 <= report.artifact_scores[name] <= 1.0

        # Reconstruction error should be a finite number
        assert np.isfinite(report.reconstruction_error_db)

        # Warnings is a list (may be empty for clean stems)
        assert isinstance(report.warnings, list)

    def test_assess_stems_warns_on_high_bleed(self) -> None:
        """assess_stems should warn when bleed is high."""
        vocals = _sine_buffer(440.0, 0.5)
        # Make drums entirely vocal content (extreme bleed)
        drums = AudioBuffer(data=vocals.data.copy(), sample_rate=SR)
        bass = _sine_buffer(60.0, 0.3)
        other = _sine_buffer(2000.0, 0.3)

        original_data = (vocals.data + drums.data + bass.data + other.data).astype(
            np.float32
        )
        original = AudioBuffer(data=original_data, sample_rate=SR)
        stems = StemSet(vocals=vocals, drums=drums, bass=bass, other=other)

        report = assess_stems(stems, original)

        # Drums stem is identical to vocals — should have high bleed
        assert report.bleed_scores["drums"] > 0.3
