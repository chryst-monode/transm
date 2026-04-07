"""Wrapper around audio-separator for stem separation.

Bridges the file-based audio-separator API to our array-based AudioBuffer pipeline.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from transm.audio_io import read_audio
from transm.types import AudioBuffer, StemSet

logger = logging.getLogger(__name__)

# Default model filenames per backend
_DEFAULT_MODELS: dict[str, str] = {
    "demucs": "htdemucs_ft.yaml",
    "roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}

# Canonical stem names we expect and the substrings audio-separator uses in filenames
_STEM_PATTERNS: dict[str, list[str]] = {
    "vocals": ["(Vocals)", "vocals"],
    "drums": ["(Drums)", "drums"],
    "bass": ["(Bass)", "bass"],
    "other": ["(Other)", "other", "(No Vocals)", "instrumental"],
}


class StemSeparator:
    """Separate audio into 4 stems using audio-separator."""

    def __init__(
        self,
        backend: str = "demucs",
        model_name: str | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize separator.

        Args:
            backend: "demucs" or "roformer"
            model_name: specific model filename, or None for the backend default
            device: "auto", "cpu", "cuda", "mps"
        """
        if backend not in _DEFAULT_MODELS:
            msg = f"Unknown backend '{backend}'. Choose from: {list(_DEFAULT_MODELS)}"
            raise ValueError(msg)

        self.backend = backend
        self.model_name = model_name or _DEFAULT_MODELS[backend]
        self.device = self.detect_device() if device == "auto" else device

    def separate(self, input_path: Path) -> StemSet:
        """Separate audio into 4 stems.

        Reads the file via audio-separator, writes stems to a temp directory,
        reads them back as AudioBuffers, and returns a StemSet.
        """
        from audio_separator.separator import Separator

        input_path = Path(input_path)
        if not input_path.exists():
            msg = f"Input file not found: {input_path}"
            raise FileNotFoundError(msg)

        tmp_dir = tempfile.TemporaryDirectory()
        try:
            output_dir = Path(tmp_dir.name)
            logger.info(
                "Separating %s with %s (model=%s, device=%s)",
                input_path.name,
                self.backend,
                self.model_name,
                self.device,
            )

            separator = Separator(output_dir=str(output_dir))
            separator.load_model(model_filename=self.model_name)
            output_files = separator.separate(str(input_path))

            # output_files is a list of paths to the generated stem files
            stem_buffers = _match_stems(output_files, output_dir, input_path.stem)
            return stem_buffers
        finally:
            tmp_dir.cleanup()

    @staticmethod
    def detect_device() -> str:
        """Auto-detect best available device: cuda > mps > cpu."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.debug("torch not available, falling back to cpu")
        return "cpu"


def _match_stems(
    output_files: list[str],
    output_dir: Path,
    input_stem: str,
) -> StemSet:
    """Match separator output files to canonical stem names and read them.

    Args:
        output_files: list of file paths returned by separator.separate()
        output_dir: the temp directory containing the output files
        input_stem: the stem (filename without extension) of the input file

    Returns:
        StemSet with all 4 stems populated
    """
    # Build a mapping from canonical stem name to the output file path
    found: dict[str, Path] = {}

    # Prefer output_files list if provided
    candidates: list[Path] = [Path(f) for f in output_files] if output_files else []

    # Also scan the output dir for any files we might have missed
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.suffix.lower() in {".wav", ".flac", ".mp3"} and f not in candidates:
                candidates.append(f)

    for stem_name, patterns in _STEM_PATTERNS.items():
        for candidate in candidates:
            fname_lower = candidate.name.lower()
            for pattern in patterns:
                if pattern.lower() in fname_lower:
                    found[stem_name] = candidate
                    break
            if stem_name in found:
                break

    missing = {"vocals", "drums", "bass", "other"} - set(found)
    if missing:
        available = [p.name for p in candidates]
        msg = (
            f"Could not find stems: {missing}. "
            f"Available files: {available}"
        )
        raise RuntimeError(msg)

    # Read all stems as AudioBuffers
    stems: dict[str, AudioBuffer] = {}
    for stem_name in ("vocals", "drums", "bass", "other"):
        stems[stem_name] = read_audio(found[stem_name])

    return StemSet(
        vocals=stems["vocals"],
        drums=stems["drums"],
        bass=stems["bass"],
        other=stems["other"],
    )
