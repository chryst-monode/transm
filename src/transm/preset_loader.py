"""Preset loading, validation, and intensity scaling."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from transm.types import (
    BassParams,
    DrumsParams,
    GlobalParams,
    MixParams,
    OtherParams,
    PresetParams,
    VocalsParams,
)

_USER_PRESETS_DIR = Path.home() / ".config" / "transm" / "presets"

# Fields that represent gain/boost values and should be scaled by intensity.
# Frequencies, ratios, thresholds, Q values, widths, and ms values are NOT scaled.
_GAIN_FIELDS: dict[str, set[str]] = {
    "drums": {
        "transient_attack_db",
        "transient_sustain_db",
        "high_shelf_gain_db",
        "low_shelf_gain_db",
    },
    "vocals": {
        "presence_gain_db",
        "level_adjust_db",
    },
    "bass": {
        "mud_cut_gain_db",
        "harmonic_gain_db",
    },
    "other": {
        "mid_boost_gain_db",
        "high_shelf_gain_db",
    },
}


def _toml_to_preset(data: dict[str, Any]) -> PresetParams:
    """Convert parsed TOML dict to PresetParams."""
    metadata = data.get("metadata", {})
    return PresetParams(
        name=metadata.get("name", ""),
        description=metadata.get("description", ""),
        drums=DrumsParams(**data["drums"]) if "drums" in data else DrumsParams(),
        vocals=VocalsParams(**data["vocals"]) if "vocals" in data else VocalsParams(),
        bass=BassParams(**data["bass"]) if "bass" in data else BassParams(),
        other=OtherParams(**data["other"]) if "other" in data else OtherParams(),
        global_params=GlobalParams(**data["global"]) if "global" in data else GlobalParams(),
        mix=MixParams(**data["mix"]) if "mix" in data else MixParams(),
    )


def load_preset(name: str) -> PresetParams:
    """Load a preset by name from bundled presets.

    Uses importlib.resources to find presets/ directory.
    Falls back to ~/.config/transm/presets/ for user presets.
    """
    toml_name = f"{name}.toml"

    # Try bundled presets first
    presets_pkg = resources.files("transm.presets")
    bundled = presets_pkg / toml_name
    if bundled.is_file():
        raw = bundled.read_text(encoding="utf-8")
        data = tomllib.loads(raw)
        return _toml_to_preset(data)

    # Fall back to user presets directory
    user_path = _USER_PRESETS_DIR / toml_name
    if user_path.is_file():
        return load_preset_from_file(user_path)

    msg = (
        f"Preset '{name}' not found. "
        f"Searched bundled presets and {_USER_PRESETS_DIR}"
    )
    raise FileNotFoundError(msg)


def load_preset_from_file(path: Path) -> PresetParams:
    """Load a preset from an explicit TOML file path."""
    if not path.is_file():
        msg = f"Preset file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _toml_to_preset(data)


def list_presets() -> list[str]:
    """List available preset names (without .toml extension).

    Includes both bundled and user presets. Duplicates are removed,
    with bundled presets taking precedence in ordering.
    """
    seen: set[str] = set()
    result: list[str] = []

    # Bundled presets
    presets_pkg = resources.files("transm.presets")
    for item in presets_pkg.iterdir():
        name = item.name
        if name.endswith(".toml"):
            stem = name[: -len(".toml")]
            if stem not in seen:
                seen.add(stem)
                result.append(stem)

    # User presets
    if _USER_PRESETS_DIR.is_dir():
        for item in sorted(_USER_PRESETS_DIR.iterdir()):
            if item.suffix == ".toml":
                stem = item.stem
                if stem not in seen:
                    seen.add(stem)
                    result.append(stem)

    return result


def validate_preset(params: PresetParams) -> list[str]:
    """Validate preset parameters are within safe ranges.

    Returns list of warning messages. Empty = all good.
    Rules: gains +/- 12 dB, freqs 20-20000 Hz, ratios >= 1.0, intensity 0-1.
    """
    warnings: list[str] = []

    def _check_gain(section: str, field: str, value: float) -> None:
        if abs(value) > 12.0:
            warnings.append(
                f"{section}.{field} = {value} dB is outside safe range (±12 dB)"
            )

    def _check_freq(section: str, field: str, value: float) -> None:
        if value < 20.0 or value > 20000.0:
            warnings.append(
                f"{section}.{field} = {value} Hz is outside audible range (20-20000 Hz)"
            )

    def _check_ratio(section: str, field: str, value: float) -> None:
        if value < 1.0:
            warnings.append(
                f"{section}.{field} = {value} must be >= 1.0"
            )

    # Drums
    d = params.drums
    _check_gain("drums", "transient_attack_db", d.transient_attack_db)
    _check_gain("drums", "transient_sustain_db", d.transient_sustain_db)
    _check_gain("drums", "high_shelf_gain_db", d.high_shelf_gain_db)
    _check_gain("drums", "low_shelf_gain_db", d.low_shelf_gain_db)
    _check_freq("drums", "high_shelf_freq_hz", d.high_shelf_freq_hz)
    _check_freq("drums", "low_shelf_freq_hz", d.low_shelf_freq_hz)
    _check_ratio("drums", "expander_ratio", d.expander_ratio)

    # Vocals
    v = params.vocals
    _check_gain("vocals", "presence_gain_db", v.presence_gain_db)
    _check_gain("vocals", "level_adjust_db", v.level_adjust_db)
    _check_freq("vocals", "deesser_freq_low_hz", v.deesser_freq_low_hz)
    _check_freq("vocals", "deesser_freq_high_hz", v.deesser_freq_high_hz)
    _check_freq("vocals", "presence_freq_hz", v.presence_freq_hz)
    _check_ratio("vocals", "expander_ratio", v.expander_ratio)

    # Bass
    b = params.bass
    _check_gain("bass", "mud_cut_gain_db", b.mud_cut_gain_db)
    _check_gain("bass", "harmonic_gain_db", b.harmonic_gain_db)
    _check_freq("bass", "hp_freq_hz", b.hp_freq_hz)
    _check_freq("bass", "mud_cut_freq_hz", b.mud_cut_freq_hz)
    _check_freq("bass", "harmonic_freq_hz", b.harmonic_freq_hz)
    _check_ratio("bass", "comp_ratio", b.comp_ratio)

    # Other
    o = params.other
    _check_gain("other", "mid_boost_gain_db", o.mid_boost_gain_db)
    _check_gain("other", "high_shelf_gain_db", o.high_shelf_gain_db)
    _check_freq("other", "mid_boost_low_hz", o.mid_boost_low_hz)
    _check_freq("other", "mid_boost_high_hz", o.mid_boost_high_hz)
    _check_freq("other", "high_shelf_freq_hz", o.high_shelf_freq_hz)

    # Mix levels
    m = params.mix
    for stem_name in ("drums_db", "vocals_db", "bass_db", "other_db"):
        val = getattr(m, stem_name)
        if abs(val) > 6.0:
            warnings.append(
                f"mix.{stem_name} = {val} dB is outside expected range (±6 dB)"
            )

    # Global
    g = params.global_params
    if g.intensity < 0.0 or g.intensity > 1.0:
        warnings.append(
            f"global.intensity = {g.intensity} is outside valid range (0.0-1.0)"
        )

    return warnings


def _scale_dataclass(obj: object, gain_fields: set[str], intensity: float) -> dict[str, Any]:
    """Scale gain fields of a frozen dataclass, returning kwargs for a new instance."""
    import dataclasses

    result = {}
    for f in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if f.name in gain_fields:
            result[f.name] = value * intensity
        else:
            result[f.name] = value
    return result


def scale_by_intensity(params: PresetParams, intensity: float) -> PresetParams:
    """Return new PresetParams with all gain/boost values scaled by intensity.

    Non-gain params (frequencies, ratios, thresholds) are NOT scaled.
    The global_params.intensity is set to the new intensity value.
    """
    return PresetParams(
        name=params.name,
        description=params.description,
        drums=DrumsParams(**_scale_dataclass(params.drums, _GAIN_FIELDS["drums"], intensity)),
        vocals=VocalsParams(**_scale_dataclass(params.vocals, _GAIN_FIELDS["vocals"], intensity)),
        bass=BassParams(**_scale_dataclass(params.bass, _GAIN_FIELDS["bass"], intensity)),
        other=OtherParams(**_scale_dataclass(params.other, _GAIN_FIELDS["other"], intensity)),
        global_params=GlobalParams(
            intensity=intensity,
            target_lufs=params.global_params.target_lufs,
            target_true_peak_dbtp=params.global_params.target_true_peak_dbtp,
        ),
        mix=params.mix,  # Mix levels are absolute balance, not scaled by intensity
    )
