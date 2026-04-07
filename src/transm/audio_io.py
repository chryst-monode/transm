"""Unified audio file I/O via soundfile, with librosa fallback for MP3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from transm.types import AudioBuffer


def read_audio(path: Path | str) -> AudioBuffer:
    """Read an audio file and return as float32 AudioBuffer.

    Mono files are converted to stereo (duplicated channel).
    MP3 files fall back to librosa.load if soundfile can't handle them.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Audio file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        # Fallback for MP3 and other formats soundfile can't read
        try:
            import librosa

            mono_data, sr = librosa.load(str(path), sr=None, mono=False)
            if mono_data.ndim == 1:
                data = mono_data[:, np.newaxis]
            else:
                data = mono_data.T  # librosa returns (channels, samples)
            data = data.astype(np.float32)
        except Exception as e:
            msg = f"Could not read audio file: {path}"
            raise ValueError(msg) from e

    if data.shape[0] == 0:
        msg = f"Audio file is empty: {path}"
        raise ValueError(msg)

    # Convert mono to stereo by duplicating the channel
    if data.shape[1] == 1:
        data = np.column_stack([data[:, 0], data[:, 0]])

    return AudioBuffer(data=data, sample_rate=int(sr))


def write_audio(buffer: AudioBuffer, path: Path | str) -> Path:
    """Write an AudioBuffer to a WAV or FLAC file.

    Format is inferred from the file extension.
    """
    path = Path(path)
    ext = path.suffix.lower()

    format_map = {
        ".wav": ("WAV", "FLOAT"),
        ".flac": ("FLAC", "PCM_24"),
    }

    if ext not in format_map:
        msg = f"Unsupported output format '{ext}'. Use .wav or .flac"
        raise ValueError(msg)

    fmt, subtype = format_map[ext]

    # Validate data range
    peak = np.max(np.abs(buffer.data))
    if peak > 1.0:
        # Warn but don't fail — the limiter should have caught this
        import warnings

        warnings.warn(
            f"Audio data exceeds [-1, 1] range (peak={peak:.4f}). "
            "Output may clip.",
            stacklevel=2,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), buffer.data, buffer.sample_rate, format=fmt, subtype=subtype)
    return path


def get_audio_info(path: Path | str) -> dict:
    """Return metadata without loading the full audio data."""
    path = Path(path)
    info = sf.info(str(path))
    return {
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
    }
