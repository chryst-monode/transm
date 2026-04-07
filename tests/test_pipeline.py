"""Tests for the Pipeline orchestrator.

All tests are marked @slow and @integration since the full pipeline requires
model downloads. Minimal tests verify instantiation and analysis-only mode.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from transm.pipeline import Pipeline
from transm.types import Metrics, PresetParams

SR = 44100


def _write_test_wav(path: Path, duration_s: float = 2.0) -> None:
    """Write a synthetic stereo WAV file."""
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float32) / SR
    mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    data = np.column_stack([mono, mono])
    sf.write(str(path), data, SR, format="WAV", subtype="FLOAT")


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineAnalysisOnly:
    def test_pipeline_analysis_only(self) -> None:
        """run_analysis_only should return valid Metrics from a WAV file."""
        preset = PresetParams(name="test")
        pipeline = Pipeline(preset=preset)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test_input.wav"
            _write_test_wav(wav_path)

            metrics = pipeline.run_analysis_only(wav_path)

            assert isinstance(metrics, Metrics)
            assert not np.isinf(metrics.lufs_integrated)
            assert metrics.spectral_centroid_hz > 0


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineInit:
    def test_pipeline_init(self) -> None:
        """Pipeline should instantiate with a default preset."""
        preset = PresetParams(name="default")
        pipeline = Pipeline(preset=preset)

        assert pipeline.preset.name == "default"
        assert pipeline.backend == "demucs"
        assert pipeline.output_format == "wav"

    def test_pipeline_init_custom_backend(self) -> None:
        """Pipeline should accept alternate backend."""
        preset = PresetParams(name="custom")
        pipeline = Pipeline(preset=preset, backend="roformer", output_format="flac")

        assert pipeline.backend == "roformer"
        assert pipeline.output_format == "flac"
