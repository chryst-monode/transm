"""Tests for report formatting."""

from __future__ import annotations

import json

from transm.report import (
    format_comparison_table,
    format_metrics_table,
    metrics_to_json,
)
from transm.types import Metrics


def _sample_metrics(
    lufs: float = -14.0,
    lra: float = 8.0,
    true_peak: float = -1.0,
    plr: float = 13.0,
    crest: float = 12.0,
    centroid: float = 3500.0,
    clipping: float = 0.5,
    tilt: float = -3.0,
) -> Metrics:
    return Metrics(
        lufs_integrated=lufs,
        loudness_range=lra,
        true_peak_dbtp=true_peak,
        peak_to_loudness_ratio=plr,
        crest_factor_db=crest,
        spectral_centroid_hz=centroid,
        clipping_percent=clipping,
        spectral_tilt=tilt,
    )


class TestFormatMetricsTable:
    def test_format_metrics_table(self) -> None:
        """Create a Metrics instance, format it, verify output contains key labels."""
        m = _sample_metrics()
        output = format_metrics_table(m)

        assert "LUFS" in output
        assert "True Peak" in output
        assert "Crest Factor" in output
        assert "Spectral Centroid" in output
        assert "Clipping" in output
        assert "Spectral Tilt" in output
        assert "Loudness Range" in output
        assert "-14.0" in output


class TestFormatComparisonTable:
    def test_format_comparison_table(self) -> None:
        """Two Metrics, verify output contains 'Before', 'After', 'Delta'."""
        before = _sample_metrics(lufs=-10.0, clipping=2.5, crest=6.0)
        after = _sample_metrics(lufs=-14.0, clipping=0.1, crest=12.0)
        output = format_comparison_table(before, after)

        assert "Before" in output
        assert "After" in output
        assert "Delta" in output
        # Should show the actual values
        assert "-10.00" in output
        assert "-14.00" in output


class TestMetricsToJson:
    def test_metrics_to_json(self) -> None:
        """Verify valid JSON, all fields present."""
        m = _sample_metrics()
        json_str = metrics_to_json(m)

        data = json.loads(json_str)
        assert data["lufs_integrated"] == -14.0
        assert data["loudness_range"] == 8.0
        assert data["true_peak_dbtp"] == -1.0
        assert data["peak_to_loudness_ratio"] == 13.0
        assert data["crest_factor_db"] == 12.0
        assert data["spectral_centroid_hz"] == 3500.0
        assert data["clipping_percent"] == 0.5
        assert data["spectral_tilt"] == -3.0
