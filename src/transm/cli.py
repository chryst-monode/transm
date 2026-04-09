"""Transm CLI — AI-powered remastering for Loudness War-era metal/rock."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from transm.analysis import compute_metrics
from transm.audio_io import read_audio, write_audio
from transm.pipeline import Pipeline
from transm.preset_loader import load_preset, load_preset_from_file, scale_by_intensity
from transm.report import format_comparison_table, format_metrics_table, metrics_to_json
from transm.separation import StemSeparator, check_separator_available

_SEPARATOR_INSTALL_MSG = (
    "Stem separation requires additional dependencies.\n"
    "Install with: pip install transm[separator]\n"
    "Note: requires Python 3.11 or 3.12 (onnxruntime lacks 3.13 support)."
)

app = typer.Typer(
    name="transm",
    help="AI-powered remastering for Loudness War-era metal/rock.",
    no_args_is_help=True,
)


def _err_console() -> Console:
    """Create a Console writing to stderr (created per-call so it picks up redirects)."""
    return Console(stderr=True)


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="Audio file to analyze", exists=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Analyze audio file and display loudness/dynamic metrics."""
    try:
        buffer = read_audio(input_file)
        metrics = compute_metrics(buffer)

        if json_output:
            typer.echo(metrics_to_json(metrics))
        else:
            typer.echo(format_metrics_table(metrics))
    except Exception as exc:
        _err_console().print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


@app.command()
def separate(
    input_file: Path = typer.Argument(..., help="Audio file to separate", exists=True),
    backend: str = typer.Option("demucs", help="Separation backend: demucs, roformer"),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for stems"
    ),
    model: str = typer.Option(None, help="Specific model name/filename"),
) -> None:
    """Separate audio into stems (vocals, drums, bass, other).

    Requires: pip install transm[separator]
    """
    try:
        if not check_separator_available():
            _err_console().print(f"[bold red]Error:[/bold red] {_SEPARATOR_INSTALL_MSG}")
            raise SystemExit(1)

        if output_dir is None:
            output_dir = input_file.parent / f"{input_file.stem}_stems"

        sep = StemSeparator(backend=backend, model_name=model)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Separating stems...", total=None)
            stems = sep.separate(input_file)
            progress.update(task, description="Done", completed=1, total=1)

        # Write stems to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, buf in stems.items():
            stem_path = output_dir / f"{input_file.stem}_{name}.wav"
            write_audio(buf, stem_path)

        typer.echo(f"Stems written to: {output_dir}")
        for name, _buf in stems.items():
            typer.echo(f"  {input_file.stem}_{name}.wav")
    except Exception as exc:
        _err_console().print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Audio file to remaster", exists=True),
    preset: str = typer.Option("2000s-metalcore", help="Processing preset name"),
    preset_file: Path = typer.Option(None, "--preset-file", help="Path to custom preset TOML"),
    intensity: float = typer.Option(
        None, help="Processing intensity 0.0-1.0 (overrides preset)"
    ),
    backend: str = typer.Option("demucs", help="Separation backend"),
    model: str = typer.Option(None, help="Specific separator model name/filename"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    output_format: str = typer.Option("wav", "--format", help="Output format: wav, flac"),
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
) -> None:
    """Full remastering pipeline: separate, process stems, remix.

    Requires: pip install transm[separator]
    """
    try:
        if not check_separator_available():
            _err_console().print(f"[bold red]Error:[/bold red] {_SEPARATOR_INSTALL_MSG}")
            raise SystemExit(1)

        # 0. Validate output format
        valid_formats = ("wav", "flac")
        if output_format.lower() not in valid_formats:
            _err_console().print(
                f"[bold red]Error:[/bold red] Unsupported format '{output_format}'. "
                f"Choose from: {', '.join(valid_formats)}"
            )
            raise SystemExit(1)
        output_format = output_format.lower()

        # 1. Load preset
        if preset_file is not None:
            params = load_preset_from_file(preset_file)
        else:
            params = load_preset(preset)

        # 2. Override intensity if provided
        if intensity is not None:
            if not 0.0 <= intensity <= 1.0:
                _err_console().print(
                    "[bold red]Error:[/bold red] Intensity must be between 0.0 and 1.0"
                )
                raise SystemExit(1)
            params = scale_by_intensity(params, intensity)

        # 3. Resolve output path
        if output is None:
            output = input_file.parent / f"{input_file.stem}_transm.{output_format}"

        # 4. Create pipeline and run with progress bar
        pipeline = Pipeline(
            preset=params, backend=backend, output_format=output_format, model_name=model
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing...", total=100)

            def update_progress(message: str, fraction: float) -> None:
                progress.update(task, completed=int(fraction * 100), description=message)

            result = pipeline.run(input_file, output, progress=update_progress)

        # 5. Display results
        if json_output:
            result_data = {
                "input": str(result.input_path),
                "output": str(result.output_path),
                "processing_time_s": round(result.processing_time_s, 2),
                "before": asdict(result.input_metrics),
                "after": asdict(result.output_metrics),
            }
            typer.echo(json.dumps(result_data, indent=2))
        else:
            typer.echo(
                format_comparison_table(result.input_metrics, result.output_metrics)
            )
            typer.echo(f"\nOutput written to: {result.output_path}")
            typer.echo(f"Processing time: {result.processing_time_s:.1f}s")
    except SystemExit:
        raise
    except Exception as exc:
        _err_console().print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


@app.command()
def compare(
    file_a: Path = typer.Argument(
        ..., help="First audio file (typically original)", exists=True
    ),
    file_b: Path = typer.Argument(
        ..., help="Second audio file (typically remastered)", exists=True
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare metrics between two audio files side-by-side."""
    try:
        buffer_a = read_audio(file_a)
        buffer_b = read_audio(file_b)

        metrics_a = compute_metrics(buffer_a)
        metrics_b = compute_metrics(buffer_b)

        if json_output:
            result_data = {
                "file_a": str(file_a),
                "file_b": str(file_b),
                "metrics_a": asdict(metrics_a),
                "metrics_b": asdict(metrics_b),
            }
            typer.echo(json.dumps(result_data, indent=2))
        else:
            typer.echo(format_comparison_table(metrics_a, metrics_b))
    except Exception as exc:
        _err_console().print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


@app.command()
def capture(
    source: str = typer.Argument(None, help="Spotify URL or URI to capture"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory for captured FLAC"
    ),
    device: str = typer.Option(
        "BlackHole 2ch", "--device", "-d", help="Loopback audio device name"
    ),
    login_flag: bool = typer.Option(False, "--login", help="Authenticate with Spotify"),
    list_devices: bool = typer.Option(
        False, "--list-devices", help="List available audio input devices"
    ),
    do_analyze: bool = typer.Option(False, "--analyze", help="Analyze captured audio"),
) -> None:
    """Capture a track via loopback recording from a streaming service."""
    try:
        if list_devices:
            from transm.capture import list_loopback_devices

            devices = list_loopback_devices()
            typer.echo("Available audio input devices:")
            for d in devices:
                typer.echo(f"  {d}")
            return

        if login_flag:
            from transm.spotify_auth import login

            login()
            typer.echo("Spotify authentication successful.")
            return

        if source is None:
            _err_console().print(
                "[bold red]Error:[/bold red] Provide a Spotify URL to capture, "
                "or use --login / --list-devices."
            )
            raise SystemExit(1)

        from transm.capture import capture_track

        if output_dir is None:
            output_dir = Path.home() / "Music" / "transm-captures"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Capturing...", total=None)
            output_path = capture_track(
                spotify_url=source,
                output_dir=output_dir,
                device_name=device,
            )
            progress.update(task, description="Done", completed=1, total=1)

        typer.echo(f"Captured: {output_path}")

        if do_analyze:
            buffer = read_audio(output_path)
            metrics = compute_metrics(buffer)
            typer.echo(format_metrics_table(metrics))

    except SystemExit:
        raise
    except Exception as exc:
        _err_console().print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    app()
