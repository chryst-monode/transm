"""Single-track audio capture via system loopback recording.

Records audio from a virtual loopback device (e.g., BlackHole on macOS)
while a streaming service plays the track. Saves as tagged FLAC.

Note: This implementation buffers the full track in memory. For tracks
under ~10 minutes at 44.1kHz stereo float32, this is ~200 MB. A streaming-
to-disk encoder (pyFLAC) is planned for album-length continuous capture.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from transm.types import AudioBuffer, TrackMetadata

logger = logging.getLogger(__name__)

_SPOTIFY_TRACK_PATTERN = re.compile(
    r"(?:spotify:track:|https?://open\.spotify\.com/track/)([a-zA-Z0-9]+)"
)


def parse_spotify_url(url: str) -> str:
    """Extract track ID from a Spotify URL or URI.

    Accepts:
        https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8
        https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8?si=...
        spotify:track:4PTG3Z6ehGkBFwjybzWkR8
    """
    match = _SPOTIFY_TRACK_PATTERN.search(url)
    if not match:
        msg = f"Could not parse Spotify track ID from: {url}"
        raise ValueError(msg)
    return match.group(1)


def get_track_metadata(track_id: str, token: str) -> TrackMetadata:
    """Fetch track metadata from the Spotify API."""
    resp = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    artists = ", ".join(a["name"] for a in data.get("artists", []))
    album = data.get("album", {})

    return TrackMetadata(
        title=data.get("name", "Unknown"),
        artist=artists,
        album=album.get("name", "Unknown"),
        album_artist=", ".join(a["name"] for a in album.get("artists", [])),
        track_number=data.get("track_number", 0),
        disc_number=data.get("disc_number", 1),
        duration_ms=data.get("duration_ms", 0),
        isrc=data.get("external_ids", {}).get("isrc", ""),
        source_uri=f"spotify:track:{track_id}",
        release_date=album.get("release_date", ""),
        genre=[],  # Spotify moved genres to the artist endpoint
        album_art_url=(
            album.get("images", [{}])[0].get("url", "") if album.get("images") else ""
        ),
    )


def start_playback(track_uri: str, token: str) -> None:
    """Start playback of a track on the user's active Spotify device."""
    resp = requests.put(
        "https://api.spotify.com/v1/me/player/play",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"uris": [track_uri]},
        timeout=10,
    )
    if resp.status_code == 404:
        msg = (
            "No active Spotify device found. "
            "Open Spotify and start playing something first."
        )
        raise RuntimeError(msg)
    resp.raise_for_status()


def pause_playback(token: str) -> None:
    """Pause playback on the user's active Spotify device (best-effort)."""
    with contextlib.suppress(Exception):
        requests.put(
            "https://api.spotify.com/v1/me/player/pause",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )


_DEFAULT_SAMPLE_RATE = 44100


def _default_sample_rate() -> int:
    """Return the default capture sample rate.

    soundcard does not expose device native sample rates.
    When real detection is needed, this should be replaced with
    platform-specific queries (CoreAudio on macOS, etc.).
    """
    return _DEFAULT_SAMPLE_RATE


def record_loopback(
    device_name: str,
    duration_s: float,
    sample_rate: int | None = None,
    stop_event: threading.Event | None = None,
) -> AudioBuffer:
    """Record audio from a loopback device for the specified duration.

    If sample_rate is None, attempts to detect the device's native rate.
    If stop_event is set, recording stops early and the recorder context
    manager exits cleanly — preventing CoreAudio thread teardown crashes.
    """
    try:
        import soundcard  # type: ignore[import-untyped]
    except ImportError as e:
        msg = (
            "Audio capture requires the 'soundcard' package. "
            "Install with: pip install transm[capture]"
        )
        raise ImportError(msg) from e

    if sample_rate is None:
        sample_rate = _default_sample_rate()

    # Find the loopback device
    mics = soundcard.all_microphones(include_loopback=True)
    device = None
    for mic in mics:
        if device_name.lower() in mic.name.lower():
            device = mic
            break

    if device is None:
        available = [m.name for m in mics]
        msg = (
            f"Loopback device '{device_name}' not found. "
            f"Available devices: {available}"
        )
        raise RuntimeError(msg)

    num_frames = int(duration_s * sample_rate)
    chunk_frames = sample_rate  # 1-second chunks
    logger.info(
        "Recording from '%s' for %.1fs at %d Hz...",
        device.name,
        duration_s,
        sample_rate,
    )

    chunks: list[np.ndarray] = []
    recorded = 0
    with device.recorder(samplerate=sample_rate, channels=2) as recorder:
        while recorded < num_frames:
            if stop_event is not None and stop_event.is_set():
                logger.info("Recording stopped early by stop event.")
                break
            remaining = num_frames - recorded
            n = min(chunk_frames, remaining)
            chunks.append(recorder.record(numframes=n))
            recorded += n

    if not chunks:
        return AudioBuffer(
            data=np.empty((0, 2), dtype=np.float32), sample_rate=sample_rate
        )

    data = np.concatenate(chunks, axis=0)
    return AudioBuffer(data=data.astype(np.float32), sample_rate=sample_rate)


def list_loopback_devices() -> list[str]:
    """List available loopback/input audio devices."""
    try:
        import soundcard  # type: ignore[import-untyped]
    except ImportError:
        return ["(soundcard not installed — pip install transm[capture])"]

    mics = soundcard.all_microphones(include_loopback=True)
    return [m.name for m in mics]


def trim_silence(
    buffer: AudioBuffer,
    threshold_db: float = -50.0,
    min_samples: int = 1024,
) -> AudioBuffer:
    """Trim leading and trailing silence from an AudioBuffer.

    Silence is defined as frames below threshold_db RMS.
    """
    data = buffer.data
    threshold_linear = 10.0 ** (threshold_db / 20.0)

    # Compute per-frame RMS (mono sum)
    mono = np.mean(np.abs(data), axis=1)

    # Find first and last non-silent sample
    above = np.where(mono > threshold_linear)[0]
    if len(above) == 0:
        # Entire buffer is silence
        return buffer

    start = max(0, above[0] - min_samples)
    end = min(len(data), above[-1] + min_samples)

    return AudioBuffer(data=data[start:end].copy(), sample_rate=buffer.sample_rate)


def save_flac_with_metadata(
    buffer: AudioBuffer,
    metadata: TrackMetadata,
    output_dir: Path,
) -> Path:
    """Save AudioBuffer as a tagged FLAC file.

    Filename: {Artist} - {Title}.flac
    If the file already exists, appends a numeric suffix to avoid overwriting.
    """
    # Sanitize filename
    safe_artist = re.sub(r'[<>:"/\\|?*]', "_", metadata.artist)
    safe_title = re.sub(r'[<>:"/\\|?*]', "_", metadata.title)
    base_name = f"{safe_artist} - {safe_title}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting existing files
    output_path = output_dir / f"{base_name}.flac"
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{base_name} ({counter}).flac"
        counter += 1

    # Write FLAC via soundfile
    sf.write(
        str(output_path),
        buffer.data,
        buffer.sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )

    # Tag with metadata via mutagen
    try:
        from mutagen.flac import FLAC

        audio = FLAC(str(output_path))
        audio["title"] = metadata.title
        audio["artist"] = metadata.artist
        audio["album"] = metadata.album
        if metadata.album_artist:
            audio["albumartist"] = metadata.album_artist
        if metadata.track_number:
            audio["tracknumber"] = str(metadata.track_number)
        if metadata.disc_number:
            audio["discnumber"] = str(metadata.disc_number)
        if metadata.release_date:
            audio["date"] = metadata.release_date
        if metadata.isrc:
            audio["isrc"] = metadata.isrc
        audio.save()
    except ImportError:
        logger.warning("mutagen not installed — FLAC saved without metadata tags.")

    return output_path


def capture_track(
    spotify_url: str,
    output_dir: Path | str | None = None,
    device_name: str = "BlackHole 2ch",
    token: str | None = None,
) -> Path:
    """Capture a single track from Spotify via loopback recording.

    Recording starts BEFORE playback to avoid losing opening transients.
    Playback is always paused in a finally block, even on errors.

    Steps:
    1. Parse Spotify URL and get metadata
    2. Start recording (pre-roll captures silence before playback)
    3. Start playback (while recording is already running)
    4. Wait for track duration
    5. Stop playback (in finally)
    6. Trim silence (removes pre-roll and trailing buffer)
    7. Save as tagged FLAC
    """
    if output_dir is None:
        output_dir = Path.home() / "Music" / "transm-captures"
    output_dir = Path(output_dir)

    # 1. Auth
    if token is None:
        from transm.spotify_auth import get_access_token

        token = get_access_token()

    # 2. Parse URL and get metadata
    track_id = parse_spotify_url(spotify_url)
    metadata = get_track_metadata(track_id, token)
    duration_s = metadata.duration_ms / 1000.0

    logger.info(
        "Capturing: %s - %s (%.0fs)",
        metadata.artist,
        metadata.title,
        duration_s,
    )

    # Pre-roll buffer: record starts before playback to capture opening transients.
    # Post-roll buffer: extra time after track ends to catch any tail.
    pre_roll_s = 2.0
    post_roll_s = 2.0
    total_record_s = pre_roll_s + duration_s + post_roll_s

    # 3. Start recording FIRST (captures silence during pre-roll)
    #    Then start playback while recording is running.
    #    Always pause playback in finally, even if recording fails.
    #    The stop_event lets us cancel recording cleanly on error —
    #    the recorder context manager exits properly, preventing the
    #    CoreAudio IO thread teardown crash (pthread_exit from workgroup).
    stop_event = threading.Event()
    recording_result: dict[str, AudioBuffer | Exception] = {}

    def _do_record() -> None:
        try:
            recording_result["buffer"] = record_loopback(
                device_name=device_name,
                duration_s=total_record_s,
                stop_event=stop_event,
            )
        except Exception as e:
            recording_result["error"] = e

    record_thread = threading.Thread(target=_do_record)
    record_thread.start()

    # Small delay to ensure recorder is listening before playback starts
    time.sleep(0.3)

    try:
        # 4. Start playback (recorder is already capturing)
        start_playback(metadata.source_uri, token)

        # 5. Wait for recording to complete
        record_thread.join(timeout=total_record_s + 30)
    finally:
        # 6. Always pause playback
        pause_playback(token)

        # 7. Signal the recorder to stop and wait for clean shutdown.
        #    The chunk-based recorder checks stop_event between 1s chunks,
        #    so the context manager __exit__ runs and CoreAudio cleans up.
        if record_thread.is_alive():
            logger.info("Signaling recording thread to stop...")
            stop_event.set()
            record_thread.join(timeout=15)
            if record_thread.is_alive():
                logger.error(
                    "Recording thread did not exit within timeout. "
                    "Audio device may remain busy."
                )

    # Check for recording errors
    if "error" in recording_result:
        raise recording_result["error"]  # type: ignore[misc]
    if "buffer" not in recording_result:
        msg = "Recording did not complete"
        raise RuntimeError(msg)

    recording = recording_result["buffer"]
    assert isinstance(recording, AudioBuffer)

    # 7. Trim silence (removes pre-roll silence and trailing buffer)
    trimmed = trim_silence(recording)

    # 8. Save
    output_path = save_flac_with_metadata(trimmed, metadata, output_dir)
    logger.info("Saved: %s", output_path)

    return output_path
