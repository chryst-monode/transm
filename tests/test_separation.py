"""Tests for stem separation wrapper.

All tests are marked slow+integration since they require model downloads.
Only test_separator_init and test_device_detection run without models.
"""

from __future__ import annotations

import pytest

from transm.separation import StemSeparator


@pytest.mark.slow
@pytest.mark.integration
class TestStemSeparator:
    """Tests for StemSeparator (most require model download)."""

    def test_separator_init(self) -> None:
        """StemSeparator can be instantiated with default settings."""
        sep = StemSeparator()
        assert sep.backend == "demucs"
        assert sep.model_name == "htdemucs_ft.yaml"
        assert sep.device in ("cuda", "mps", "cpu")

    def test_separator_init_roformer(self) -> None:
        """StemSeparator can be instantiated with roformer backend."""
        sep = StemSeparator(backend="roformer")
        assert sep.backend == "roformer"
        assert sep.model_name == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    def test_separator_init_invalid_backend(self) -> None:
        """StemSeparator raises ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            StemSeparator(backend="invalid")

    def test_separator_init_custom_model(self) -> None:
        """StemSeparator accepts a custom model name."""
        sep = StemSeparator(model_name="custom_model.yaml")
        assert sep.model_name == "custom_model.yaml"

    def test_device_detection(self) -> None:
        """detect_device returns a valid device string."""
        device = StemSeparator.detect_device()
        assert device in ("cuda", "mps", "cpu")

    def test_separate_missing_file(self) -> None:
        """separate() raises FileNotFoundError for nonexistent input."""
        from pathlib import Path

        sep = StemSeparator()
        with pytest.raises(FileNotFoundError):
            sep.separate(Path("/nonexistent/audio.wav"))

    def test_separate_synthetic(self, tmp_path: pytest.TempPathFactory) -> None:
        """Integration: separate a short synthetic WAV and verify StemSet structure.

        This test downloads the model on first run and is slow.
        """
        import numpy as np
        import soundfile as sf

        # Create a short (3s) synthetic mix of 4 sine waves
        sr = 44100
        duration = 3.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr

        # Simulate a simple mix: low bass, mid vocal, high "other", percussive hit
        bass = 0.3 * np.sin(2 * np.pi * 80 * t)
        vocal = 0.3 * np.sin(2 * np.pi * 440 * t)
        other = 0.2 * np.sin(2 * np.pi * 2000 * t)
        drums = 0.2 * np.sin(2 * np.pi * 150 * t) * np.exp(-t * 3)

        mix = (bass + vocal + other + drums).astype(np.float32)
        stereo = np.column_stack([mix, mix])

        input_path = tmp_path / "test_mix.wav"
        sf.write(str(input_path), stereo, sr)

        sep = StemSeparator(device="cpu")
        stem_set = sep.separate(input_path)

        # Verify structure
        assert stem_set.vocals.sample_rate == sr
        assert stem_set.drums.sample_rate == sr
        assert stem_set.bass.sample_rate == sr
        assert stem_set.other.sample_rate == sr

        # All stems should have audio data
        for name, stem in stem_set.items():
            assert stem.num_samples > 0, f"{name} stem has no samples"
            assert stem.num_channels == 2, f"{name} stem is not stereo"
