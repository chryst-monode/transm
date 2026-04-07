"""Core data types for Transm. Every module imports from here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class AudioBuffer:
    """Immutable container for audio data + sample rate.

    Data is always float32, shape (samples, channels). Mono is stored as (samples, 1).
    """

    data: NDArray[np.float32]
    sample_rate: int

    @staticmethod
    def from_array(data: NDArray[np.floating], sr: int) -> AudioBuffer:
        """Create an AudioBuffer, validating and converting to float32 (samples, channels)."""
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim != 2:
            msg = f"Audio data must be 1D or 2D, got {arr.ndim}D"
            raise ValueError(msg)
        if arr.shape[0] == 0:
            msg = "Audio data must have at least one sample"
            raise ValueError(msg)
        return AudioBuffer(data=arr, sample_rate=sr)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.data.shape[0] / self.sample_rate

    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def num_samples(self) -> int:
        return self.data.shape[0]


@dataclass(frozen=True)
class Metrics:
    """Complete analysis metrics for an audio buffer."""

    lufs_integrated: float
    loudness_range: float  # LRA in LU
    true_peak_dbtp: float
    peak_to_loudness_ratio: float  # PLR in dB
    crest_factor_db: float
    spectral_centroid_hz: float
    clipping_percent: float
    spectral_tilt: float  # dB/octave


@dataclass(frozen=True)
class MetricsDelta:
    """Before/after comparison of metrics."""

    before: Metrics
    after: Metrics

    @property
    def lufs_delta(self) -> float:
        return self.after.lufs_integrated - self.before.lufs_integrated

    @property
    def lra_delta(self) -> float:
        return self.after.loudness_range - self.before.loudness_range

    @property
    def plr_delta(self) -> float:
        return self.after.peak_to_loudness_ratio - self.before.peak_to_loudness_ratio

    @property
    def crest_factor_delta(self) -> float:
        return self.after.crest_factor_db - self.before.crest_factor_db

    @property
    def true_peak_delta(self) -> float:
        return self.after.true_peak_dbtp - self.before.true_peak_dbtp

    @property
    def clipping_delta(self) -> float:
        return self.after.clipping_percent - self.before.clipping_percent

    @property
    def spectral_centroid_delta(self) -> float:
        return self.after.spectral_centroid_hz - self.before.spectral_centroid_hz

    @property
    def spectral_tilt_delta(self) -> float:
        return self.after.spectral_tilt - self.before.spectral_tilt


@dataclass(frozen=True)
class StemQAReport:
    """Quality assessment for separated stems."""

    bleed_scores: dict[str, float]  # stem name → 0.0 (clean) to 1.0 (severe bleed)
    artifact_scores: dict[str, float]  # stem name → 0.0 (clean) to 1.0 (severe artifacts)
    reconstruction_error_db: float  # RMS diff between sum(stems) and original
    warnings: list[str] = field(default_factory=list)


# --- Preset parameter types ---


@dataclass(frozen=True)
class DrumsParams:
    transient_attack_db: float = 4.5
    transient_sustain_db: float = -2.0
    expander_threshold_db: float = -30.0
    expander_ratio: float = 1.5
    high_shelf_gain_db: float = -3.0
    high_shelf_freq_hz: float = 8000.0
    low_shelf_gain_db: float = 2.0
    low_shelf_freq_hz: float = 80.0


@dataclass(frozen=True)
class VocalsParams:
    deesser_freq_low_hz: float = 6000.0
    deesser_freq_high_hz: float = 9000.0
    expander_ratio: float = 1.2
    presence_gain_db: float = 1.5
    presence_freq_hz: float = 4000.0
    level_adjust_db: float = -1.5


@dataclass(frozen=True)
class BassParams:
    hp_freq_hz: float = 30.0
    mud_cut_freq_hz: float = 250.0
    mud_cut_gain_db: float = -3.0
    mud_cut_q: float = 2.0
    harmonic_freq_hz: float = 1000.0
    harmonic_gain_db: float = 2.0
    comp_ratio: float = 2.0
    comp_attack_ms: float = 30.0


@dataclass(frozen=True)
class OtherParams:
    mid_boost_low_hz: float = 500.0
    mid_boost_high_hz: float = 2000.0
    mid_boost_gain_db: float = 2.0
    high_shelf_gain_db: float = -2.0
    high_shelf_freq_hz: float = 10000.0
    stereo_width: float = 1.2


@dataclass(frozen=True)
class GlobalParams:
    intensity: float = 0.35
    target_lufs: float = -14.0
    target_true_peak_dbtp: float = -1.0


@dataclass(frozen=True)
class PresetParams:
    """All tunable parameters for a processing preset."""

    name: str
    description: str = ""
    drums: DrumsParams = field(default_factory=DrumsParams)
    vocals: VocalsParams = field(default_factory=VocalsParams)
    bass: BassParams = field(default_factory=BassParams)
    other: OtherParams = field(default_factory=OtherParams)
    global_params: GlobalParams = field(default_factory=GlobalParams)


@dataclass(frozen=True)
class StemSet:
    """Collection of separated stems, keyed by name."""

    vocals: AudioBuffer
    drums: AudioBuffer
    bass: AudioBuffer
    other: AudioBuffer

    def items(self) -> list[tuple[str, AudioBuffer]]:
        return [
            ("vocals", self.vocals),
            ("drums", self.drums),
            ("bass", self.bass),
            ("other", self.other),
        ]

    def __getitem__(self, key: str) -> AudioBuffer:
        return getattr(self, key)


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run."""

    input_path: Path
    output_path: Path
    input_metrics: Metrics
    output_metrics: Metrics
    delta: MetricsDelta
    stem_qa: StemQAReport
    processing_time_s: float
