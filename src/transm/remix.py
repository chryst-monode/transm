"""Stem remixing — sum processed stems with gain staging and polarity check."""

from __future__ import annotations

import numpy as np

from transm.dsp.common import db_to_linear
from transm.types import AudioBuffer, MixParams, StemSet


def remix_stems(
    stems: StemSet,
    original: AudioBuffer,
    mix_params: MixParams | None = None,
) -> AudioBuffer:
    """Sum processed stems back together with gain staging.

    Steps:
    1. Apply per-stem mix levels (dB → linear gain)
    2. Sum all stems: vocals + drums + bass + other
    3. Check polarity alignment -- if cross-correlation with original is negative,
       flip the mixed signal (unlikely but safety check)
    4. Headroom management: if peak exceeds 0.95, scale down proportionally
    5. Return remixed AudioBuffer
    """
    sr = original.sample_rate

    # Determine target length (use original as reference)
    target_len = original.num_samples
    n_channels = original.num_channels

    # Per-stem mix levels (0 dB = unity when mix_params is None)
    mix_levels: dict[str, float] = {}
    if mix_params is not None:
        mix_levels = {
            "drums": mix_params.drums_db,
            "vocals": mix_params.vocals_db,
            "bass": mix_params.bass_db,
            "other": mix_params.other_db,
        }

    # Sum all stems, aligning lengths
    mixed = np.zeros((target_len, n_channels), dtype=np.float32)

    for name, stem in stems.items():
        stem_data = stem.data
        # Align length
        min_len = min(target_len, stem_data.shape[0])
        # Align channels
        if stem_data.shape[1] < n_channels:
            # Duplicate mono to match channel count
            stem_data = np.column_stack(
                [stem_data[:, 0]] * n_channels
            )
        elif stem_data.shape[1] > n_channels:
            stem_data = stem_data[:, :n_channels]

        gain = db_to_linear(mix_levels.get(name, 0.0))
        mixed[:min_len] += stem_data[:min_len] * gain

    # Polarity check
    if not check_polarity(
        AudioBuffer(data=mixed, sample_rate=sr),
        original,
    ):
        mixed = -mixed

    # Headroom management
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    return AudioBuffer(data=mixed.astype(np.float32), sample_rate=sr)


def check_polarity(mixed: AudioBuffer, reference: AudioBuffer) -> bool:
    """Return True if signals are in-phase (positive correlation).

    Uses np.corrcoef on mono-summed signals.
    """
    # Mono-sum both signals
    mixed_mono = np.mean(mixed.data, axis=1)
    ref_mono = np.mean(reference.data, axis=1)

    # Align lengths
    min_len = min(len(mixed_mono), len(ref_mono))
    mixed_mono = mixed_mono[:min_len]
    ref_mono = ref_mono[:min_len]

    # Handle silence edge case
    if np.all(mixed_mono == 0) or np.all(ref_mono == 0):
        return True

    corr_matrix = np.corrcoef(mixed_mono, ref_mono)
    correlation = corr_matrix[0, 1]

    # NaN can occur if one signal is constant
    if np.isnan(correlation):
        return True

    return bool(correlation >= 0)
