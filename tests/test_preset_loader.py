"""Tests for preset loading, validation, and intensity handling."""

from __future__ import annotations

import numpy as np
import pytest

from transm.preset_loader import (
    effective_mix,
    list_presets,
    load_preset,
    scale_by_intensity,
    validate_preset,
)
from transm.remix import remix_stems
from transm.types import (
    AudioBuffer,
    DrumsParams,
    MixParams,
    PresetParams,
    StemSet,
)


class TestLoadBundledPreset:
    def test_load_bundled_preset(self) -> None:
        """Load '2000s-metalcore', verify all fields populated, name matches."""
        preset = load_preset("2000s-metalcore")

        assert preset.name == "2000s Metalcore"
        assert preset.description != ""

        # Verify drums fields populated from TOML
        assert preset.drums.transient_attack_db == 3.0
        assert preset.drums.high_shelf_freq_hz == 9000.0
        assert preset.drums.low_shelf_gain_db == 1.0

        # Verify vocals fields
        assert preset.vocals.deesser_freq_low_hz == 6500.0
        assert preset.vocals.presence_gain_db == 5.0
        assert preset.vocals.level_adjust_db == 2.0

        # Verify bass fields
        assert preset.bass.hp_freq_hz == 45.0
        assert preset.bass.harmonic_gain_db == 2.5

        # Verify other fields
        assert preset.other.mid_boost_gain_db == 4.0
        assert preset.other.high_shelf_gain_db == 0.0
        assert preset.other.stereo_width == 1.15

        # Verify mix levels
        assert preset.mix.drums_db == 1.5
        assert preset.mix.vocals_db == 2.0
        assert preset.mix.bass_db == 1.0
        assert preset.mix.other_db == 2.5

        # Verify global fields
        assert preset.global_params.intensity == 0.35
        assert preset.global_params.target_lufs == -14.0


class TestListPresets:
    def test_list_presets(self) -> None:
        """Should include '2000s-metalcore'."""
        names = list_presets()
        assert "2000s-metalcore" in names


class TestValidatePreset:
    def test_validate_preset_valid(self) -> None:
        """Default metalcore preset should produce no warnings."""
        preset = load_preset("2000s-metalcore")
        warnings = validate_preset(preset)
        assert warnings == []

    def test_validate_preset_extreme_gain(self) -> None:
        """Preset with gain > 12 dB should produce a warning."""
        preset = PresetParams(
            name="extreme",
            drums=DrumsParams(transient_attack_db=15.0),
        )
        warnings = validate_preset(preset)
        assert len(warnings) >= 1
        assert any("transient_attack_db" in w for w in warnings)
        assert any("±12" in w or "12" in w for w in warnings)


class TestScaleByIntensity:
    """`scale_by_intensity` is a pure override of `global_params.intensity`.

    Under the current architecture, gain scaling is performed inline by DSP
    modules via `params.global_params.intensity`, and mix-bus scaling is
    performed at remix time via `effective_mix`. Both read the intensity
    field set by this function, so there is no pre-multiplication here.
    Older versions pre-scaled gain fields, which caused a double-scale bug
    when the CLI `--intensity` override was used (x × intensity in
    `scale_by_intensity` followed by × intensity again in the DSP modules).
    """

    def test_scale_by_intensity_sets_intensity_field(self) -> None:
        """Scale at 0.5 -> only global_params.intensity changes."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.5)

        # Global intensity is the only changed field
        assert scaled.global_params.intensity == 0.5
        assert scaled.global_params.target_lufs == -14.0
        assert scaled.global_params.target_true_peak_dbtp == -1.0

    def test_scale_by_intensity_leaves_gains_unchanged(self) -> None:
        """Gain fields are NOT pre-scaled — DSP modules apply intensity inline."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.5)

        assert scaled.drums.transient_attack_db == preset.drums.transient_attack_db
        assert scaled.drums.high_shelf_gain_db == preset.drums.high_shelf_gain_db
        assert scaled.drums.low_shelf_gain_db == preset.drums.low_shelf_gain_db
        assert scaled.vocals.presence_gain_db == preset.vocals.presence_gain_db
        assert scaled.vocals.level_adjust_db == preset.vocals.level_adjust_db
        assert scaled.bass.mud_cut_gain_db == preset.bass.mud_cut_gain_db
        assert scaled.bass.harmonic_gain_db == preset.bass.harmonic_gain_db
        assert scaled.other.mid_boost_gain_db == preset.other.mid_boost_gain_db
        assert scaled.other.high_shelf_gain_db == preset.other.high_shelf_gain_db

    def test_scale_by_intensity_leaves_mix_unchanged(self) -> None:
        """Mix fields are NOT pre-scaled — `effective_mix` scales at remix time."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.5)

        assert scaled.mix.drums_db == preset.mix.drums_db
        assert scaled.mix.vocals_db == preset.mix.vocals_db
        assert scaled.mix.bass_db == preset.mix.bass_db
        assert scaled.mix.other_db == preset.mix.other_db

    def test_scale_by_intensity_leaves_structural_fields_unchanged(self) -> None:
        """Frequencies, ratios, thresholds, widths stay unchanged at any intensity."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.5)

        assert scaled.drums.high_shelf_freq_hz == 9000.0
        assert scaled.drums.low_shelf_freq_hz == 80.0
        assert scaled.drums.expander_ratio == 1.15
        assert scaled.vocals.deesser_freq_low_hz == 6500.0
        assert scaled.vocals.presence_freq_hz == 3500.0
        assert scaled.vocals.expander_ratio == 1.2
        assert scaled.bass.hp_freq_hz == 45.0
        assert scaled.bass.mud_cut_freq_hz == 220.0
        assert scaled.bass.comp_ratio == 2.0
        assert scaled.other.stereo_width == 1.15

    def test_scale_by_intensity_zero(self) -> None:
        """Scale at 0.0 -> only intensity changes; no field is zeroed here."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.0)

        assert scaled.global_params.intensity == 0.0
        # Everything else is the same object content as the source preset
        assert scaled.drums == preset.drums
        assert scaled.vocals == preset.vocals
        assert scaled.bass == preset.bass
        assert scaled.other == preset.other
        assert scaled.mix == preset.mix


class TestEffectiveMix:
    """`effective_mix` lerps mix-bus dB values toward 0 by intensity."""

    def test_effective_mix_unity(self) -> None:
        """At intensity=1.0, mix passes through unchanged."""
        mix = MixParams(drums_db=1.5, vocals_db=2.0, bass_db=1.0, other_db=2.5)
        scaled = effective_mix(mix, 1.0)

        assert scaled == mix

    def test_effective_mix_half(self) -> None:
        """At intensity=0.5, all dB values are halved."""
        mix = MixParams(drums_db=1.5, vocals_db=2.0, bass_db=1.0, other_db=2.5)
        scaled = effective_mix(mix, 0.5)

        assert scaled.drums_db == pytest.approx(0.75)
        assert scaled.vocals_db == pytest.approx(1.0)
        assert scaled.bass_db == pytest.approx(0.5)
        assert scaled.other_db == pytest.approx(1.25)

    def test_effective_mix_zero(self) -> None:
        """At intensity=0.0, all dB values are zero (unity linear gain)."""
        mix = MixParams(drums_db=1.5, vocals_db=2.0, bass_db=1.0, other_db=2.5)
        scaled = effective_mix(mix, 0.0)

        assert scaled.drums_db == 0.0
        assert scaled.vocals_db == 0.0
        assert scaled.bass_db == 0.0
        assert scaled.other_db == 0.0

    def test_intensity_zero_is_true_remix_passthrough(self) -> None:
        """With `effective_mix(mix, 0.0)`, `remix_stems(sum_of_stems)` equals sum_of_stems.

        This is the end-to-end contract that the master wetness knob is
        expected to honor: at intensity=0, there is no rebalance. Without
        `effective_mix`, a preset with any nonzero mix-bus dB value (like
        `2000s-metalcore.toml`) would apply an audible rebalance here.
        """
        sr = 44100
        n = sr  # 1 second
        rng = np.random.default_rng(0)

        def _stem() -> AudioBuffer:
            return AudioBuffer(
                data=(rng.standard_normal((n, 2)) * 0.05).astype(np.float32),
                sample_rate=sr,
            )

        drums, vocals, bass, other = _stem(), _stem(), _stem(), _stem()
        stems = StemSet(drums=drums, vocals=vocals, bass=bass, other=other)
        original = AudioBuffer(
            data=(drums.data + vocals.data + bass.data + other.data),
            sample_rate=sr,
        )

        preset = load_preset("2000s-metalcore")
        scaled_mix = effective_mix(preset.mix, 0.0)
        remixed = remix_stems(stems, original, mix_params=scaled_mix)

        # RMS difference should be zero (unity gains + in-phase stems + no clipping)
        rms_err = float(np.sqrt(np.mean((remixed.data - original.data) ** 2)))
        assert rms_err == pytest.approx(0.0, abs=1e-7)


class TestLoadPresetNotFound:
    def test_load_preset_not_found(self) -> None:
        """Should raise FileNotFoundError for nonexistent preset."""
        with pytest.raises(FileNotFoundError):
            load_preset("nonexistent-preset-that-does-not-exist")
