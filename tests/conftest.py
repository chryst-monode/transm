"""Synthetic audio generators for testing. All return AudioBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from transm.types import AudioBuffer

SR = 44100


def generate_sine(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sr: int = SR,
    amplitude: float = 0.5,
    num_channels: int = 2,
) -> AudioBuffer:
    """Pure sine wave at a given frequency and amplitude."""
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    mono = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    data = np.column_stack([mono] * num_channels)
    return AudioBuffer(data=data, sample_rate=sr)


def generate_drum_hit(
    sr: int = SR,
    attack_ms: float = 2.0,
    decay_ms: float = 150.0,
    freq_hz: float = 80.0,
    amplitude: float = 0.9,
) -> AudioBuffer:
    """Synthetic drum hit: sine burst with gaussian attack and exponential decay."""
    duration_s = (attack_ms + decay_ms * 5) / 1000.0
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Gaussian attack envelope
    attack_samples = int(attack_ms / 1000.0 * sr)
    decay_samples = n_samples - attack_samples

    envelope = np.zeros(n_samples, dtype=np.float32)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)
    decay_tau = decay_ms / 1000.0
    envelope[attack_samples:] = np.exp(
        -np.arange(decay_samples, dtype=np.float32) / (decay_tau * sr)
    )

    mono = (amplitude * envelope * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    data = np.column_stack([mono, mono])
    return AudioBuffer(data=data, sample_rate=sr)


def generate_crushed_drum_pattern(
    num_hits: int = 8,
    sr: int = SR,
    clip_threshold: float = 0.3,
) -> AudioBuffer:
    """Multiple drum hits with hard clipping applied (simulates brickwall limiting)."""
    hit = generate_drum_hit(sr=sr)
    gap_samples = int(0.15 * sr)  # 150ms between hits
    gap = np.zeros((gap_samples, 2), dtype=np.float32)

    parts = []
    for _ in range(num_hits):
        parts.append(hit.data)
        parts.append(gap)

    pattern = np.concatenate(parts, axis=0)
    # Simulate brickwall limiting: hard clip
    pattern = np.clip(pattern, -clip_threshold, clip_threshold)
    # Normalize to near full scale
    peak = np.max(np.abs(pattern))
    if peak > 0:
        pattern = pattern / peak * 0.95
    return AudioBuffer(data=pattern.astype(np.float32), sample_rate=sr)


def generate_sibilant_vocal(
    sr: int = SR,
    duration_s: float = 2.0,
) -> AudioBuffer:
    """Voiced signal with boosted sibilance in the 6-9 kHz range."""
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Voiced component: fundamental + harmonics
    voiced = 0.3 * np.sin(2 * np.pi * 200 * t)
    voiced += 0.15 * np.sin(2 * np.pi * 400 * t)
    voiced += 0.1 * np.sin(2 * np.pi * 600 * t)

    # Sibilant component: band-limited noise in 6-9 kHz
    from scipy.signal import butter, sosfilt

    noise = np.random.default_rng(42).normal(0, 0.4, n_samples).astype(np.float32)
    sos = butter(4, [6000, 9000], btype="bandpass", fs=sr, output="sos")
    sibilance = sosfilt(sos, noise).astype(np.float32)

    mono = (voiced + sibilance).astype(np.float32)
    # Normalize
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.8

    data = np.column_stack([mono, mono])
    return AudioBuffer(data=data, sample_rate=sr)


def generate_signal_with_dynamics(
    sr: int = SR,
    duration_s: float = 4.0,
    loud_amplitude: float = 0.8,
    quiet_amplitude: float = 0.05,
) -> AudioBuffer:
    """Alternating loud and quiet sections for testing expansion."""
    n_samples = int(sr * duration_s)
    quarter = n_samples // 4
    t = np.arange(n_samples, dtype=np.float32) / sr

    signal = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    envelope = np.ones(n_samples, dtype=np.float32)

    # Alternate: loud, quiet, loud, quiet
    envelope[:quarter] = loud_amplitude
    envelope[quarter : 2 * quarter] = quiet_amplitude
    envelope[2 * quarter : 3 * quarter] = loud_amplitude
    envelope[3 * quarter :] = quiet_amplitude

    mono = signal * envelope
    data = np.column_stack([mono, mono])
    return AudioBuffer(data=data, sample_rate=sr)


@pytest.fixture
def sine_440() -> AudioBuffer:
    return generate_sine(freq_hz=440.0, duration_s=1.0)


@pytest.fixture
def drum_hit() -> AudioBuffer:
    return generate_drum_hit()


@pytest.fixture
def crushed_drums() -> AudioBuffer:
    return generate_crushed_drum_pattern()


@pytest.fixture
def sibilant_vocal() -> AudioBuffer:
    return generate_sibilant_vocal()


@pytest.fixture
def dynamic_signal() -> AudioBuffer:
    return generate_signal_with_dynamics()
