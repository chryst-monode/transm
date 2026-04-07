"""Tests for audio_io module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transm.audio_io import get_audio_info, read_audio, write_audio
from transm.types import AudioBuffer


class TestWriteAndRead:
    """Round-trip tests: write then read, verify data integrity."""

    def test_wav_round_trip(self, tmp_path: Path, sine_440: AudioBuffer) -> None:
        out = tmp_path / "test.wav"
        write_audio(sine_440, out)
        loaded = read_audio(out)

        assert loaded.sample_rate == sine_440.sample_rate
        assert loaded.num_channels == sine_440.num_channels
        assert loaded.num_samples == sine_440.num_samples
        np.testing.assert_allclose(loaded.data, sine_440.data, atol=1e-6)

    def test_flac_round_trip(self, tmp_path: Path, sine_440: AudioBuffer) -> None:
        out = tmp_path / "test.flac"
        write_audio(sine_440, out)
        loaded = read_audio(out)

        assert loaded.sample_rate == sine_440.sample_rate
        assert loaded.num_channels == sine_440.num_channels
        # FLAC is lossless but uses PCM_24, so allow slightly more tolerance
        np.testing.assert_allclose(loaded.data, sine_440.data, atol=1e-4)

    def test_mono_converted_to_stereo(self, tmp_path: Path) -> None:
        """Mono input files should be read back as stereo."""
        import soundfile as sf

        mono = np.sin(np.arange(1000, dtype=np.float32) * 0.1)
        path = tmp_path / "mono.wav"
        sf.write(str(path), mono, 44100, format="WAV", subtype="FLOAT")

        loaded = read_audio(path)
        assert loaded.num_channels == 2
        np.testing.assert_allclose(loaded.data[:, 0], loaded.data[:, 1], atol=1e-7)


class TestReadErrors:
    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_audio(Path("/nonexistent/file.wav"))

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.wav"
        empty.touch()
        with pytest.raises((ValueError, Exception)):
            read_audio(empty)


class TestWriteErrors:
    def test_unsupported_format(self, sine_440: AudioBuffer, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported output format"):
            write_audio(sine_440, tmp_path / "test.mp3")

    def test_creates_parent_dirs(self, sine_440: AudioBuffer, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "test.wav"
        result = write_audio(sine_440, nested)
        assert result.exists()


class TestGetAudioInfo:
    def test_returns_metadata(self, tmp_path: Path, sine_440: AudioBuffer) -> None:
        path = tmp_path / "info_test.wav"
        write_audio(sine_440, path)
        info = get_audio_info(path)

        assert info["channels"] == 2
        assert info["sample_rate"] == 44100
        assert info["format"] == "WAV"
        assert info["duration"] > 0
