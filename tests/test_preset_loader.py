"""Tests for preset loading, validation, and intensity scaling."""

from __future__ import annotations

import pytest

from transm.preset_loader import (
    list_presets,
    load_preset,
    scale_by_intensity,
    validate_preset,
)
from transm.types import (
    BassParams,
    DrumsParams,
    GlobalParams,
    OtherParams,
    PresetParams,
    VocalsParams,
)


class TestLoadBundledPreset:
    def test_load_bundled_preset(self) -> None:
        """Load '2000s-metalcore', verify all fields populated, name matches."""
        preset = load_preset("2000s-metalcore")

        assert preset.name == "2000s Metalcore"
        assert preset.description != ""

        # Verify drums fields populated from TOML
        assert preset.drums.transient_attack_db == 4.5
        assert preset.drums.high_shelf_freq_hz == 8000.0
        assert preset.drums.low_shelf_gain_db == 2.0

        # Verify vocals fields
        assert preset.vocals.deesser_freq_low_hz == 6000.0
        assert preset.vocals.presence_gain_db == 1.5

        # Verify bass fields
        assert preset.bass.hp_freq_hz == 30.0
        assert preset.bass.harmonic_gain_db == 2.0

        # Verify other fields
        assert preset.other.mid_boost_gain_db == 2.0
        assert preset.other.stereo_width == 1.2

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
    def test_scale_by_intensity(self) -> None:
        """Scale at 0.5 -> all gain values halved, frequencies unchanged."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.5)

        # Gain fields should be halved
        assert scaled.drums.transient_attack_db == pytest.approx(4.5 * 0.5)
        assert scaled.drums.high_shelf_gain_db == pytest.approx(-3.0 * 0.5)
        assert scaled.drums.low_shelf_gain_db == pytest.approx(2.0 * 0.5)
        assert scaled.vocals.presence_gain_db == pytest.approx(1.5 * 0.5)
        assert scaled.vocals.level_adjust_db == pytest.approx(-1.5 * 0.5)
        assert scaled.bass.mud_cut_gain_db == pytest.approx(-3.0 * 0.5)
        assert scaled.bass.harmonic_gain_db == pytest.approx(2.0 * 0.5)
        assert scaled.other.mid_boost_gain_db == pytest.approx(2.0 * 0.5)
        assert scaled.other.high_shelf_gain_db == pytest.approx(-2.0 * 0.5)

        # Frequency fields should be unchanged
        assert scaled.drums.high_shelf_freq_hz == 8000.0
        assert scaled.drums.low_shelf_freq_hz == 80.0
        assert scaled.vocals.deesser_freq_low_hz == 6000.0
        assert scaled.vocals.presence_freq_hz == 4000.0
        assert scaled.bass.hp_freq_hz == 30.0
        assert scaled.bass.mud_cut_freq_hz == 250.0

        # Ratio fields should be unchanged
        assert scaled.drums.expander_ratio == 1.5
        assert scaled.vocals.expander_ratio == 1.2
        assert scaled.bass.comp_ratio == 2.0

        # Global intensity should be updated
        assert scaled.global_params.intensity == 0.5
        # Target LUFS/peak should be unchanged
        assert scaled.global_params.target_lufs == -14.0
        assert scaled.global_params.target_true_peak_dbtp == -1.0

    def test_scale_by_intensity_zero(self) -> None:
        """Scale at 0.0 -> all gains are 0, frequencies unchanged."""
        preset = load_preset("2000s-metalcore")
        scaled = scale_by_intensity(preset, 0.0)

        # All gain fields should be zero
        assert scaled.drums.transient_attack_db == 0.0
        assert scaled.drums.transient_sustain_db == 0.0
        assert scaled.drums.high_shelf_gain_db == 0.0
        assert scaled.drums.low_shelf_gain_db == 0.0
        assert scaled.vocals.presence_gain_db == 0.0
        assert scaled.vocals.level_adjust_db == 0.0
        assert scaled.bass.mud_cut_gain_db == 0.0
        assert scaled.bass.harmonic_gain_db == 0.0
        assert scaled.other.mid_boost_gain_db == 0.0
        assert scaled.other.high_shelf_gain_db == 0.0

        # Frequencies, ratios unchanged
        assert scaled.drums.high_shelf_freq_hz == 8000.0
        assert scaled.vocals.expander_ratio == 1.2
        assert scaled.bass.comp_attack_ms == 30.0
        assert scaled.other.stereo_width == 1.2

        # Intensity updated
        assert scaled.global_params.intensity == 0.0


class TestLoadPresetNotFound:
    def test_load_preset_not_found(self) -> None:
        """Should raise FileNotFoundError for nonexistent preset."""
        with pytest.raises(FileNotFoundError):
            load_preset("nonexistent-preset-that-does-not-exist")
