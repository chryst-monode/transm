"""Pipeline orchestrator — runs the full Transm remastering pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from transm.analysis import compute_delta, compute_metrics
from transm.audio_io import read_audio, write_audio
from transm.dsp.bass import process_bass
from transm.dsp.drums import process_drums
from transm.dsp.other import process_other
from transm.dsp.vocals import process_vocals
from transm.limiter import apply_final_limiter
from transm.preset_loader import load_preset
from transm.remix import remix_stems
from transm.separation import StemSeparator
from transm.stem_qa import assess_stems
from transm.types import (
    AudioBuffer,
    Metrics,
    PipelineResult,
    PresetParams,
    StemSet,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


class Pipeline:
    """Full Transm remastering pipeline."""

    def __init__(
        self,
        preset: PresetParams,
        backend: str = "demucs",
        output_format: str = "wav",
    ) -> None:
        """Initialize pipeline with preset and separation backend.

        Args:
            preset: Processing parameters for all stems and global settings.
            backend: Stem separation backend ("demucs" or "roformer").
            output_format: Output file format ("wav" or "flac").
        """
        self.preset = preset
        self.backend = backend
        self.output_format = output_format
        self._separator = StemSeparator(backend=backend)

    def run(
        self,
        input_path: Path,
        output_path: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Execute the full remastering pipeline.

        Steps:
        1. Read input audio
        2. Pre-analysis (compute_metrics)
        3. Stem separation (StemSeparator)
        4. Stem QA (assess_stems)
        5. Per-stem DSP (process_drums, process_vocals, process_bass, process_other)
        6. Remix (remix_stems)
        7. Final limiting (apply_final_limiter)
        8. Post-analysis (compute_metrics)
        9. Write output
        10. Return PipelineResult
        """
        t_start = time.monotonic()
        input_path = Path(input_path)

        # Resolve output path
        if output_path is None:
            output_path = (
                input_path.parent / f"{input_path.stem}_transm.{self.output_format}"
            )
        output_path = Path(output_path)

        # 1. Read input
        _report(progress, "Reading audio...", 0.02)
        original = read_audio(input_path)

        # 2. Pre-analysis
        _report(progress, "Analyzing input...", 0.05)
        input_metrics = compute_metrics(original)

        # 3. Stem separation
        _report(progress, "Separating stems...", 0.10)
        stems = self._separator.separate(input_path)
        _report(progress, "Stems separated", 0.70)

        # 4. Stem QA
        stem_qa = assess_stems(stems, original)
        if stem_qa.warnings:
            for w in stem_qa.warnings:
                logger.warning("Stem QA: %s", w)

        # 5. Per-stem DSP
        _report(progress, "Processing drums...", 0.72)
        processed_drums = process_drums(stems.drums, self.preset)

        _report(progress, "Processing vocals...", 0.76)
        processed_vocals = process_vocals(stems.vocals, self.preset)

        _report(progress, "Processing bass...", 0.80)
        processed_bass = process_bass(stems.bass, self.preset)

        _report(progress, "Processing other...", 0.84)
        processed_other = process_other(stems.other, self.preset)

        processed_stems = StemSet(
            vocals=processed_vocals,
            drums=processed_drums,
            bass=processed_bass,
            other=processed_other,
        )

        # 6. Remix
        _report(progress, "Remixing...", 0.88)
        remixed = remix_stems(processed_stems, original)

        # 7. Final limiting
        _report(progress, "Applying final limiter...", 0.92)
        limited = apply_final_limiter(
            remixed,
            target_lufs=self.preset.global_params.target_lufs,
            ceiling_dbtp=self.preset.global_params.target_true_peak_dbtp,
        )

        # 8. Post-analysis
        _report(progress, "Analyzing output...", 0.95)
        output_metrics = compute_metrics(limited)

        # 9. Write output
        _report(progress, "Writing output...", 0.98)
        write_audio(limited, output_path)

        _report(progress, "Done", 1.0)

        # 10. Build result
        processing_time = time.monotonic() - t_start
        delta = compute_delta(input_metrics, output_metrics)

        return PipelineResult(
            input_path=input_path,
            output_path=output_path,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            delta=delta,
            stem_qa=stem_qa,
            processing_time_s=processing_time,
        )

    def run_analysis_only(self, input_path: Path) -> Metrics:
        """Run only pre-analysis on input file."""
        input_path = Path(input_path)
        buffer = read_audio(input_path)
        return compute_metrics(buffer)

    def run_separation_only(
        self,
        input_path: Path,
        output_dir: Path | None = None,
    ) -> StemSet:
        """Run only stem separation. Optionally write stems to output_dir.

        Args:
            input_path: Path to the input audio file.
            output_dir: If provided, writes each stem as a WAV file to this directory.

        Returns:
            StemSet with all 4 separated stems.
        """
        input_path = Path(input_path)
        stems = self._separator.separate(input_path)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, stem in stems.items():
                stem_path = output_dir / f"{input_path.stem}_{name}.wav"
                write_audio(stem, stem_path)
                logger.info("Wrote stem: %s", stem_path)

        return stems


def _report(
    progress: ProgressCallback | None,
    message: str,
    fraction: float,
) -> None:
    """Fire progress callback if provided."""
    if progress is not None:
        progress(message, fraction)
