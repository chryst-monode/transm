"""Tests for single-track capture module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transm.capture import (
    capture_track,
    list_loopback_devices,
    parse_spotify_url,
    save_flac_with_metadata,
    trim_silence,
)
from transm.types import AudioBuffer, TrackMetadata


class TestParseSpotifyUrl:
    def test_https_url(self) -> None:
        track_id = parse_spotify_url(
            "https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8"
        )
        assert track_id == "4PTG3Z6ehGkBFwjybzWkR8"

    def test_https_url_with_params(self) -> None:
        track_id = parse_spotify_url(
            "https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8?si=abc123"
        )
        assert track_id == "4PTG3Z6ehGkBFwjybzWkR8"

    def test_spotify_uri(self) -> None:
        track_id = parse_spotify_url("spotify:track:4PTG3Z6ehGkBFwjybzWkR8")
        assert track_id == "4PTG3Z6ehGkBFwjybzWkR8"

    def test_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_spotify_url("https://example.com/not-a-spotify-url")

    def test_album_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_spotify_url("https://open.spotify.com/album/abc123")


class TestTrimSilence:
    def test_trims_leading_silence(self) -> None:
        sr = 44100
        silence = np.zeros((sr, 2), dtype=np.float32)  # 1s silence
        tone = np.full((sr, 2), 0.5, dtype=np.float32)  # 1s signal
        data = np.concatenate([silence, tone])
        buf = AudioBuffer(data=data, sample_rate=sr)

        trimmed = trim_silence(buf, threshold_db=-30.0)
        # Should be shorter than original (silence removed)
        assert trimmed.num_samples < buf.num_samples

    def test_trims_trailing_silence(self) -> None:
        sr = 44100
        tone = np.full((sr, 2), 0.5, dtype=np.float32)
        silence = np.zeros((sr, 2), dtype=np.float32)
        data = np.concatenate([tone, silence])
        buf = AudioBuffer(data=data, sample_rate=sr)

        trimmed = trim_silence(buf, threshold_db=-30.0)
        assert trimmed.num_samples < buf.num_samples

    def test_all_silence_returns_original(self) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.zeros((sr, 2), dtype=np.float32), sample_rate=sr
        )
        trimmed = trim_silence(buf)
        assert trimmed.num_samples == buf.num_samples

    def test_no_silence_preserves_length(self) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.full((sr, 2), 0.5, dtype=np.float32), sample_rate=sr
        )
        trimmed = trim_silence(buf, threshold_db=-30.0)
        # Should be approximately the same (within min_samples buffer)
        assert abs(trimmed.num_samples - buf.num_samples) < 2048


class TestSaveFlacWithMetadata:
    def test_saves_tagged_flac(self, tmp_path: Path) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.random.default_rng(42).normal(0, 0.3, (sr * 2, 2)).astype(
                np.float32
            ),
            sample_rate=sr,
        )
        meta = TrackMetadata(
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            track_number=3,
            release_date="2004-06-15",
        )

        path = save_flac_with_metadata(buf, meta, tmp_path)
        assert path.exists()
        assert path.suffix == ".flac"
        assert "Test Artist" in path.name
        assert "Test Song" in path.name

    def test_metadata_tags_written(self, tmp_path: Path) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.random.default_rng(42).normal(0, 0.3, (sr, 2)).astype(
                np.float32
            ),
            sample_rate=sr,
        )
        meta = TrackMetadata(
            title="Tagged Track",
            artist="Tagged Artist",
            album="Tagged Album",
            track_number=5,
        )

        path = save_flac_with_metadata(buf, meta, tmp_path)

        try:
            from mutagen.flac import FLAC

            audio = FLAC(str(path))
            assert audio["title"][0] == "Tagged Track"
            assert audio["artist"][0] == "Tagged Artist"
            assert audio["album"][0] == "Tagged Album"
            assert audio["tracknumber"][0] == "5"
        except ImportError:
            pytest.skip("mutagen not installed")

    def test_sanitizes_filename(self, tmp_path: Path) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.zeros((sr, 2), dtype=np.float32), sample_rate=sr
        )
        meta = TrackMetadata(
            title='Song: "With/Special" <Chars>',
            artist="Artist?Name",
            album="Album",
        )
        path = save_flac_with_metadata(buf, meta, tmp_path)
        assert path.exists()
        assert "?" not in path.name
        assert '"' not in path.name


class TestFilenameCollisions:
    def test_no_overwrite_on_collision(self, tmp_path: Path) -> None:
        sr = 44100
        buf = AudioBuffer(
            data=np.zeros((sr, 2), dtype=np.float32), sample_rate=sr
        )
        meta = TrackMetadata(title="Same Song", artist="Same Artist", album="Album")

        path1 = save_flac_with_metadata(buf, meta, tmp_path)
        path2 = save_flac_with_metadata(buf, meta, tmp_path)

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert "(1)" in path2.name


class TestListDevices:
    def test_returns_list(self) -> None:
        devices = list_loopback_devices()
        assert isinstance(devices, list)


class TestCaptureOrchestration:
    """Tests for the capture_track orchestration logic using mocks."""

    def test_playback_paused_on_recording_failure(self) -> None:
        """Playback must be paused even if recording raises an exception."""
        from unittest.mock import patch

        from transm.capture import capture_track

        pause_called = False

        def mock_pause(token: str) -> None:
            nonlocal pause_called
            pause_called = True

        with (
            patch("transm.capture.get_track_metadata") as mock_meta,
            patch("transm.capture.start_playback"),
            patch("transm.capture.pause_playback", side_effect=mock_pause),
            patch("transm.capture.record_loopback", side_effect=RuntimeError("device error")),
            patch("transm.spotify_auth.get_access_token", return_value="fake-token"),
        ):
            mock_meta.return_value = TrackMetadata(
                title="Test", artist="Test", album="Test", duration_ms=5000,
                source_uri="spotify:track:abc",
            )
            with pytest.raises(RuntimeError, match="device error"):
                capture_track("https://open.spotify.com/track/abc123")

        assert pause_called, "pause_playback must be called even when recording fails"

    def test_recording_starts_before_playback(self) -> None:
        """Recording must start before playback to capture opening transients."""
        from unittest.mock import patch

        call_order: list[str] = []

        def mock_record(*args: object, **kwargs: object) -> AudioBuffer:
            call_order.append("record_start")
            import time
            time.sleep(0.1)
            call_order.append("record_end")
            return AudioBuffer(
                data=np.zeros((44100, 2), dtype=np.float32), sample_rate=44100
            )

        def mock_play(*args: object, **kwargs: object) -> None:
            call_order.append("playback_start")

        with (
            patch("transm.capture.get_track_metadata") as mock_meta,
            patch("transm.capture.start_playback", side_effect=mock_play),
            patch("transm.capture.pause_playback"),
            patch("transm.capture.record_loopback", side_effect=mock_record),
            patch("transm.capture.save_flac_with_metadata", return_value=Path("/tmp/test.flac")),
            patch("transm.capture.trim_silence", side_effect=lambda b, **kw: b),
            patch("transm.spotify_auth.get_access_token", return_value="fake-token"),
        ):
            mock_meta.return_value = TrackMetadata(
                title="Test", artist="Test", album="Test", duration_ms=1000,
                source_uri="spotify:track:abc",
            )
            capture_track("https://open.spotify.com/track/abc123")

        assert call_order.index("record_start") < call_order.index("playback_start"), (
            f"Recording must start before playback. Order was: {call_order}"
        )
