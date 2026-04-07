"""Report formatting for Transm metrics and QA results."""

from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO

from rich.console import Console
from rich.table import Table

from transm.types import Metrics, StemQAReport


def _make_console() -> Console:
    """Create a Console configured for consistent string capture."""
    return Console(file=StringIO(), force_terminal=True, width=100)


def format_metrics_table(metrics: Metrics) -> str:
    """Format metrics as a Rich table rendered to string."""
    table = Table(title="Audio Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("LUFS (Integrated)", f"{metrics.lufs_integrated:.1f} LUFS")
    table.add_row("Loudness Range", f"{metrics.loudness_range:.1f} LU")
    table.add_row("True Peak", f"{metrics.true_peak_dbtp:.1f} dBTP")
    table.add_row("Peak-to-Loudness Ratio", f"{metrics.peak_to_loudness_ratio:.1f} dB")
    table.add_row("Crest Factor", f"{metrics.crest_factor_db:.1f} dB")
    table.add_row("Spectral Centroid", f"{metrics.spectral_centroid_hz:.0f} Hz")
    table.add_row("Clipping", f"{metrics.clipping_percent:.2f}%")
    table.add_row("Spectral Tilt", f"{metrics.spectral_tilt:.2f} dB/oct")

    console = _make_console()
    console.print(table)
    return console.file.getvalue()


def _delta_arrow(delta: float) -> str:
    """Return an arrow character indicating direction of change."""
    if delta > 0.01:
        return "▲"
    if delta < -0.01:
        return "▼"
    return "─"


def format_comparison_table(before: Metrics, after: Metrics) -> str:
    """Format before/after comparison with deltas and improvement arrows."""
    table = Table(title="Before / After Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")

    rows = [
        ("LUFS (Integrated)", before.lufs_integrated, after.lufs_integrated, "LUFS"),
        ("Loudness Range", before.loudness_range, after.loudness_range, "LU"),
        ("True Peak", before.true_peak_dbtp, after.true_peak_dbtp, "dBTP"),
        ("Peak-to-Loudness Ratio", before.peak_to_loudness_ratio, after.peak_to_loudness_ratio, "dB"),
        ("Crest Factor", before.crest_factor_db, after.crest_factor_db, "dB"),
        ("Spectral Centroid", before.spectral_centroid_hz, after.spectral_centroid_hz, "Hz"),
        ("Clipping", before.clipping_percent, after.clipping_percent, "%"),
        ("Spectral Tilt", before.spectral_tilt, after.spectral_tilt, "dB/oct"),
    ]

    for label, b_val, a_val, unit in rows:
        delta = a_val - b_val
        arrow = _delta_arrow(delta)
        table.add_row(
            label,
            f"{b_val:.2f} {unit}",
            f"{a_val:.2f} {unit}",
            f"{arrow} {delta:+.2f} {unit}",
        )

    console = _make_console()
    console.print(table)
    return console.file.getvalue()


def format_stem_qa_table(qa: StemQAReport) -> str:
    """Format stem QA results with warnings highlighted."""
    table = Table(title="Stem QA Report", show_header=True, header_style="bold cyan")
    table.add_column("Stem", style="bold")
    table.add_column("Bleed Score", justify="right")
    table.add_column("Artifact Score", justify="right")
    table.add_column("Status")

    all_stems = sorted(set(qa.bleed_scores.keys()) | set(qa.artifact_scores.keys()))

    for stem in all_stems:
        bleed = qa.bleed_scores.get(stem, 0.0)
        artifact = qa.artifact_scores.get(stem, 0.0)

        if bleed > 0.5 or artifact > 0.5:
            status = "[bold red]⚠ WARNING[/bold red]"
        elif bleed > 0.3 or artifact > 0.3:
            status = "[yellow]FAIR[/yellow]"
        else:
            status = "[green]GOOD[/green]"

        table.add_row(stem, f"{bleed:.3f}", f"{artifact:.3f}", status)

    # Add reconstruction error row
    table.add_section()
    table.add_row(
        "Reconstruction Error",
        f"{qa.reconstruction_error_db:.2f} dB",
        "",
        "",
    )

    # Add warnings if any
    if qa.warnings:
        table.add_section()
        for warning in qa.warnings:
            table.add_row("[bold red]Warning[/bold red]", warning, "", "")

    console = _make_console()
    console.print(table)
    return console.file.getvalue()


def metrics_to_json(metrics: Metrics) -> str:
    """Serialize metrics to JSON string."""
    return json.dumps(asdict(metrics), indent=2)
