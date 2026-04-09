# Transm Capture: Lossless Stream Capture Component

> **Implementation status (v0.1):** Single-track capture only. Given a Spotify
> URL, starts loopback recording, triggers playback, waits for track duration,
> trims silence, saves tagged FLAC. Multi-track album capture, silence-based
> boundary detection, streaming-to-disk encoding, and non-Spotify source support
> are planned but not yet implemented. See `transm capture --help` for current CLI.

## Overview

Transm Capture is a local system audio capture module that records lossless audio from any streaming service (Spotify, Apple Music, Tidal, Qobuz, Deezer) by capturing the decoded audio output at the OS level. It encodes to FLAC in real-time, automatically detects track boundaries, and enriches output files with metadata from the Spotify API.

The audio never touches a lossy codec after decoding — the capture happens downstream of the streaming app's own decoder, so the output is a bit-perfect copy of whatever the service delivered.

**This is not DRM circumvention.** The tool captures audio from the system's audio output bus, the same as plugging a cable from headphone-out to line-in. No encrypted streams are intercepted, no protocol is reverse-engineered, no keys are extracted.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ OS Audio Layer                                              │
│                                                             │
│  Spotify/Tidal/Apple Music                                  │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐     ┌──────────────┐     ┌───────────────┐   │
│  │ App Audio │────▶│ Multi-Output │────▶│ Real Speakers │   │
│  │ Output    │     │ Device       │────▶│ (monitoring)  │   │
│  └──────────┘     └──────┬───────┘     └───────────────┘   │
│                          │                                  │
│                          ▼                                  │
│                   ┌──────────────┐                          │
│                   │  BlackHole   │                          │
│                   │  (virtual    │                          │
│                   │   loopback)  │                          │
│                   └──────┬───────┘                          │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Transm Capture                                               │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  SoundCard   │──▶│  Ring Buffer  │──▶│  Track Slicer  │  │
│  │  Loopback    │   │  (numpy)      │   │                │  │
│  │  Capture     │   │               │   │  Silence Det.  │  │
│  │              │   │  24-bit       │   │  + Spotify API  │  │
│  │  44.1/48kHz  │   │  stereo       │   │  polling       │  │
│  └─────────────┘   └──────────────┘   └───────┬────────┘  │
│                                                │            │
│                                    ┌───────────┼─────────┐  │
│                                    │           │         │  │
│                                    ▼           ▼         ▼  │
│                               ┌────────┐ ┌────────┐        │
│                               │ pyFLAC │ │Metadata│        │
│                               │ Stream │ │Tagger  │        │
│                               │Encoder │ │(mutagen│        │
│                               │        │ │ /FLAC) │        │
│                               └───┬────┘ └───┬────┘        │
│                                   │          │              │
│                                   ▼          ▼              │
│                              ┌──────────────────┐           │
│                              │  Output FLAC     │           │
│                              │  01 - Atreyu -   │           │
│                              │  The Curse.flac  │           │
│                              └──────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Three Concurrent Threads

The capture module runs three threads:

**Thread 1: Audio Capture (real-time, highest priority)**
Reads audio frames from the loopback device via SoundCard in a tight loop. Writes frames into a thread-safe ring buffer. This thread must never block or drop frames.

**Thread 2: Spotify Poller (background, low priority)**
Polls the Spotify Web API every ~2 seconds for the currently playing track. Detects track changes by comparing track URIs. When a change is detected, it pushes a `TrackChange` event (with metadata) onto an event queue.

**Thread 3: Track Slicer + Encoder (consumer)**
Reads from the ring buffer and monitors both the event queue and the audio signal. When a track boundary is detected (via Spotify event OR silence gap), it finalizes the current FLAC file, tags it with metadata, and opens a new FLAC stream encoder for the next track.

### Track Boundary Detection (Dual Strategy)

Relying on only one detection method is fragile. Transm Capture uses both:

**Primary: Spotify API polling**
The Spotify API endpoint `GET /v1/me/player/currently-playing` returns the current track URI and playback position. When the URI changes, a new track has started. The API also returns the track duration, so we know roughly when to expect the transition. Poll every 1-2 seconds.

Pros: Precise track identification, gives us metadata immediately.
Cons: 1-2 second latency on detection. Doesn't work for non-Spotify sources.

**Secondary: Silence detection (RMS-based)**
Calculate RMS energy over sliding windows (~50ms). When RMS drops below a threshold (configurable, default -50 dBFS) for longer than a minimum gap duration (configurable, default 300ms), flag it as a potential track boundary.

Pros: Works with any audio source. Near-zero latency.
Cons: False positives on songs with quiet intros or deliberate silence. No metadata.

**Combined logic:**
- If Spotify API is connected: use API as the authoritative boundary signal, but use silence detection to find the precise sample where the gap occurs (the API is 1-2s behind real-time). When the API says "track changed," look backwards in the buffer for the most recent silence gap and split there.
- If no API connection (Apple Music, Tidal, etc.): fall back to silence detection only. Prompt the user to manually tag files after capture, or attempt metadata lookup via AcoustID/Chromaprint fingerprinting.

---

## Platform-Specific Audio Routing

### macOS (Primary Target)

**Requirement:** BlackHole virtual audio driver (open source, GPL-3.0).

**Setup (one-time):**
1. Install BlackHole: `brew install blackhole-2ch`
2. Open Audio MIDI Setup
3. Create a Multi-Output Device containing both BlackHole 2ch and the real output device (speakers/headphones)
4. Set the Multi-Output Device as system output

**Transm Capture automates detection:** On launch, enumerate audio devices via SoundCard, look for a device named "BlackHole 2ch". If not found, prompt the user to install it and walk through setup.

**Capture:** SoundCard records from BlackHole 2ch at the system sample rate (typically 44.1kHz or 48kHz), 24-bit.

### Linux

**PulseAudio:** Every output sink automatically creates a monitor source. SoundCard can list these via `soundcard.all_microphones(include_loopback=True)`. No additional driver needed.

**PipeWire:** Same monitor source pattern. Also supports `pw-loopback` for explicit routing. SoundCard works with PipeWire's PulseAudio compatibility layer.

**Capture:** Select the monitor source of the output device, record at native sample rate.

### Windows

**WASAPI Loopback:** Windows Audio Session API natively supports loopback capture. However, the standard `sounddevice` and `PyAudio` libraries don't expose it. Use **PyAudioWPatch** (a fork of PyAudio with WASAPI loopback support).

**Setup:** No virtual audio driver needed. WASAPI loopback captures directly from the output device.

**Capture:** PyAudioWPatch opens the default output device in loopback mode, records at native sample rate.

### Abstraction Layer

```python
class CaptureBackend:
    """Platform-agnostic audio capture interface."""

    def list_loopback_devices(self) -> list[AudioDevice]: ...
    def open_stream(self, device: AudioDevice, sample_rate: int,
                    bit_depth: int, channels: int) -> AudioStream: ...
    def read_frames(self, stream: AudioStream,
                    num_frames: int) -> np.ndarray: ...
    def close_stream(self, stream: AudioStream) -> None: ...


class MacOSCapture(CaptureBackend):
    """BlackHole + SoundCard."""
    ...

class LinuxCapture(CaptureBackend):
    """PulseAudio/PipeWire monitor + SoundCard."""
    ...

class WindowsCapture(CaptureBackend):
    """WASAPI loopback + PyAudioWPatch."""
    ...
```

---

## Real-Time FLAC Encoding

The key library here is **pyFLAC**, which wraps libFLAC and supports streaming encode:

```python
import pyflac

class StreamingFlacWriter:
    """Writes audio frames to FLAC in real-time as they arrive."""

    def __init__(self, output_path: str, sample_rate: int = 44100,
                 channels: int = 2, bit_depth: int = 24):
        self.output_path = output_path
        self.file = open(output_path, 'wb')
        self.encoder = pyflac.StreamEncoder(
            write_callback=self._write_callback,
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bit_depth,
            compression_level=5,  # balance of speed vs size
        )

    def _write_callback(self, buffer: bytes,
                        num_bytes: int, num_samples: int,
                        current_frame: int) -> None:
        """Called by libFLAC with compressed data chunks."""
        self.file.write(buffer)

    def write_frames(self, audio: np.ndarray) -> None:
        """Feed raw PCM frames to the encoder."""
        self.encoder.process(audio)

    def finalize(self) -> None:
        """Flush encoder and close file."""
        self.encoder.finish()
        self.file.close()
```

This means we never buffer an entire track in memory. Frames flow from the loopback device through the encoder to disk in near-real-time. Memory usage stays constant regardless of track length.

---

## Metadata Pipeline

### With Spotify API Connected

When the Spotify poller detects a track change, it captures:

```python
@dataclass
class TrackMetadata:
    title: str
    artist: str
    album: str
    album_artist: str
    track_number: int
    disc_number: int
    duration_ms: int
    isrc: str              # International Standard Recording Code
    spotify_uri: str
    release_date: str
    genre: list[str]       # from artist endpoint
    album_art_url: str     # highest resolution available
```

After the FLAC file is finalized, we tag it using **mutagen**:

```python
from mutagen.flac import FLAC

def tag_flac(path: str, meta: TrackMetadata, art_bytes: bytes):
    audio = FLAC(path)
    audio['title'] = meta.title
    audio['artist'] = meta.artist
    audio['album'] = meta.album
    audio['albumartist'] = meta.album_artist
    audio['tracknumber'] = str(meta.track_number)
    audio['discnumber'] = str(meta.disc_number)
    audio['date'] = meta.release_date
    audio['isrc'] = meta.isrc
    audio['genre'] = meta.genre

    # Embed album art
    picture = Picture()
    picture.type = 3  # front cover
    picture.mime = 'image/jpeg'
    picture.data = art_bytes
    audio.add_picture(picture)

    audio.save()
```

### Without Spotify (Fallback)

For Apple Music, Tidal, or any other source, Transm Capture can attempt post-hoc identification using **Chromaprint** (open-source audio fingerprinting) via the **pyacoustid** library, which queries the AcoustID/MusicBrainz database:

```python
import acoustid

def identify_track(flac_path: str) -> dict:
    """Fingerprint a captured file and look up metadata."""
    results = acoustid.match(ACOUSTID_API_KEY, flac_path)
    for score, recording_id, title, artist in results:
        if score > 0.8:
            return {'title': title, 'artist': artist,
                    'musicbrainz_id': recording_id}
    return None
```

---

## CLI Interface

```bash
# List available loopback devices
transm capture --list-devices

# Start capturing from BlackHole with Spotify integration
transm capture --device "BlackHole 2ch" --spotify --output ./captured/

# Capture without Spotify (silence detection + fingerprint lookup)
transm capture --device "BlackHole 2ch" --output ./captured/

# Capture a specific number of tracks then stop
transm capture --device "BlackHole 2ch" --spotify --tracks 12 --output ./captured/

# Capture and immediately feed into the Transm processing pipeline
transm capture --device "BlackHole 2ch" --spotify --output ./captured/ --process --preset metalcore
```

### Output Structure

```
captured/
├── Atreyu - The Curse (2004)/
│   ├── 01 - Blood Children.flac
│   ├── 02 - The Crimson.flac
│   ├── 03 - Bleeding Mascara.flac
│   ├── ...
│   └── cover.jpg
├── Underoath - They're Only Chasing Safety (2004)/
│   ├── 01 - Young and Aspiring.flac
│   ├── ...
│   └── cover.jpg
└── capture.log
```

---

## Spotify Authentication

Transm uses the **Authorization Code Flow with PKCE** (the only flow Spotify allows for user-scoped endpoints as of late 2025). No client secret is stored locally.

```bash
# First-time setup: opens browser for Spotify OAuth
transm capture --spotify-login

# Stores refresh token in ~/.config/transm/spotify.json
# Subsequent runs use the refresh token automatically
```

Required scopes: `user-read-playback-state`, `user-read-currently-playing`

The PKCE flow means the user authorizes Transm to read their playback state. No premium account is needed for the API access itself (premium is needed on Spotify's end for lossless playback quality, but that's the user's concern, not the tool's).

---

## Dependencies

### Required

| Library | Purpose | License |
|---------|---------|---------|
| `soundcard` | Cross-platform loopback capture | BSD-3 |
| `pyflac` | Real-time FLAC streaming encoder | MIT |
| `numpy` | Audio frame buffering | BSD-3 |
| `mutagen` | FLAC metadata tagging | GPL-2.0+ |
| `requests` | Spotify API calls | Apache-2.0 |

### Optional

| Library | Purpose | License |
|---------|---------|---------|
| `pyaudiowpatch` | WASAPI loopback (Windows only) | MIT |
| `pyacoustid` | Audio fingerprinting (non-Spotify fallback) | MIT |
| `chromaprint` | Fingerprint generation (used by pyacoustid) | LGPL-2.1 |

### External (User-Installed)

| Software | Purpose | Platform |
|----------|---------|----------|
| BlackHole 2ch | Virtual audio loopback | macOS |
| (none needed) | PulseAudio/PipeWire built-in | Linux |
| (none needed) | WASAPI built-in | Windows |

### License Note

mutagen is GPL-2.0+, which is compatible with Transm's GPL-3.0 license (inherited from Pedalboard/Matchering in the main pipeline). The capture component could technically be split into a separate BSD/MIT-licensed package if desired, by swapping mutagen for a permissively-licensed tagger — but since the rest of Transm is already GPL-3.0, this is a non-issue.

---

## Quality Guarantees

**What you get:**
- Bit-perfect capture of whatever the streaming service decoded and output to the audio bus
- 24-bit FLAC encoding (no bit-depth reduction)
- Native sample rate preservation (44.1kHz from Spotify/Tidal CD-quality, 48kHz from some services, up to 192kHz if the source and virtual device support it)
- Zero lossy re-encoding — the path is: service decodes → PCM on audio bus → BlackHole passes bit-perfectly → pyFLAC encodes lossless

**What you don't get:**
- Better-than-source quality (if Spotify delivered 16-bit/44.1kHz, capturing in 24-bit just adds empty bits)
- Guaranteed lossless-from-studio quality (depends on the streaming tier the user pays for)
- DRM-free status by circumvention (the audio was already DRM-free at the point of capture — it was decoded PCM on the system audio bus)

---

## Roadmap

**v0.1 (2-3 weekends)**
- macOS only (BlackHole + SoundCard)
- Spotify API integration for track detection + metadata
- Real-time FLAC encoding via pyFLAC
- Basic CLI

**v0.2 (1 month)**
- Linux support (PulseAudio/PipeWire)
- Windows support (WASAPI via PyAudioWPatch)
- Silence-based fallback track detection
- Album art embedding

**v0.3 (2 months)**
- Chromaprint/AcoustID fingerprinting for non-Spotify sources
- `--process` flag to chain directly into Transm remastering pipeline
- Gradio UI for monitoring capture status
- Batch album mode (auto-stop after N tracks)

**v1.0**
- Robust error recovery (handle audio dropouts, API failures)
- Support for Apple Music / Tidal metadata APIs
- Real-time waveform/spectrogram display during capture
- Integration with music library managers (beets)
