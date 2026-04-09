# Agent Updates

Coordination file for AI agents working on this repo. Every agent **must** read this before starting and update it after completing work.

## Format

Add entries in reverse chronological order (newest first). Each entry must include:

```
### YYYY-MM-DD — <short description>
- **Agent**: <identity (e.g., Claude Code session, NanoClaw, etc.)>
- **Branch**: <branch name>
- **Files changed**: <list of files added/modified/deleted>
- **Summary**: <what was done and why>
- **State**: <clean / WIP / blocked — and why if not clean>
```

## Changelog

### 2026-04-09 — Fix capture: redirect URI + CoreAudio crash
- **Agent**: Claude Code (Opus 4.6)
- **Branch**: feat/capture
- **Files changed**:
  - `src/transm/spotify_auth.py` — redirect URI changed from `http://localhost` to `http://127.0.0.1` (Spotify Dashboard rejects http://localhost, and https://localhost causes self-signed cert issues)
  - `src/transm/capture.py` — chunk-based recording with stop event to fix CoreAudio thread teardown crash; non-daemon recording thread
- **Summary**: Two fixes for the capture module (PR #7). (1) Spotify PKCE redirect URI updated to `http://127.0.0.1:8765/callback` — the only form the Spotify Dashboard accepts for local dev. (2) CoreAudio crash fix: replaced single blocking `recorder.record()` call with 1-second chunk loop checking a `threading.Event`. On error/cancellation, the stop event is set, the chunk loop exits, and the recorder context manager closes cleanly — preventing the pthread_exit-from-workgroup crash. Recording thread changed from daemon to regular.
- **State**: clean

### 2026-04-07 — v0.1 Implementation Complete
- **Agent**: Claude Code (Opus 4.6) — orchestrator + 8 sub-agents
- **Branch**: feat/v0.1-implementation
- **Files changed**: 30 new files (src/transm/ + tests/)
- **Summary**: Full v0.1 CLI built via 4-wave parallel agent swarm. 103 tests passing, 0 lint errors. Modules: types, audio_io, analysis, DSP primitives (transient shaper, expander, de-esser), 4 stem processors (drums/vocals/bass/other), separation wrapper, stem QA, presets (TOML), remix, true-peak limiter, pipeline orchestrator, Typer CLI with Rich output, report formatting.
- **State**: clean

### 2026-04-07 — Wave 1-D: Stem Separation Wrapper & QA Modules
- **Agent**: Claude Code (Opus 4.6) — sub-agent
- **Branch**: (same as parent session)
- **Files changed**:
  - `src/transm/separation.py` (new) — StemSeparator class wrapping audio-separator
  - `src/transm/stem_qa.py` (new) — assess_stems, estimate_bleed, estimate_artifacts, check_reconstruction
  - `tests/test_separation.py` (new) — 7 tests (all @slow/@integration)
  - `tests/test_stem_qa.py` (new) — 8 unit tests using synthetic stems
- **Summary**: Implemented file-based audio-separator bridge (demucs/roformer backends, auto device detection, temp dir lifecycle, stem filename matching). Built spectral QA pipeline: bleed via STFT cross-correlation, artifacts via spectral flux in quiet passages, reconstruction error via RMS dB ratio. Tests NOT verified — sandbox denied pytest execution.
- **State**: WIP (tests written but not run; parent agent must verify with `pytest tests/test_stem_qa.py -v`)

### 2026-04-07 — v0.1 Implementation (Wave 0: Foundation)
- **Agent**: Claude Code (Opus 4.6)
- **Branch**: feat/v0.1-implementation
- **Files changed**: pyproject.toml, src/transm/__init__.py, src/transm/types.py, src/transm/audio_io.py, src/transm/dsp/__init__.py, src/transm/presets/__init__.py, tests/__init__.py, tests/conftest.py, tests/test_audio_io.py
- **Summary**: Wave 0 foundation — all type contracts (AudioBuffer, Metrics, PresetParams, StemSet, PipelineResult), audio I/O with soundfile/librosa fallback, synthetic test generators, 8 passing tests. Waves 1-4 build on these interfaces.
- **State**: in-progress (Wave 0 complete, Waves 1-4 pending)

### 2026-04-07 — Repository setup and fork workflow
- **Agent**: Claude Code (Opus 4.6)
- **Branch**: main
- **Files changed**: README.md, .gitignore, .co-authors, CLAUDE.md, agent-updates.md, docs/architecture.md, docs/feasibility-assessment.md
- **Summary**: Initial repo creation. Set up fork-based contribution model (TeeYum owns upstream, chryst-monode fork for dev). Added project docs, architecture, feasibility assessment, agent coordination, and co-authoring config.
- **State**: clean
