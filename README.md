# Transm (Transmute AI)

**Open-source AI-powered remastering for Loudness War-era metal and rock.**

Transm separates crushed masters into individual stems using state-of-the-art AI models, applies genre-aware DSP processing to restore dynamics and reduce fatigue, and remixes them with proper headroom. Built for audiophiles who want their 2000s metalcore to sound less like it was mastered inside a trash compactor.

## Demo: Before / After

All clips processed with the `2000s-metalcore` preset at intensity 0.7. Download and A/B compare.

### As I Lay Dying — "94 Hours" (10s clip, 0:30–0:40)

- [Original clip](docs/samples/aild_94hours_original_clip.wav) (-12.5 LUFS)
- [Remastered clip](docs/samples/aild_94hours_transm_70_clip.wav) (-14.0 LUFS, +4.4 dB crest factor)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Crest Factor | 12.4 dB | 16.8 dB | **+4.4 dB** |
| Peak-to-Loudness Ratio | 8.3 dB | 13.2 dB | **+4.8 dB** |
| True Peak | -4.2 dBTP | -1.0 dBTP | -3.2 dB |

### Darkest Hour — "Sound The Surrender" (10s clip, 0:45–0:55)

- [Original clip](docs/samples/dh_sound_the_surrender_original_clip.wav) (-8.5 LUFS, clipping at +1.5 dBTP)
- [Remastered clip](docs/samples/dh_sound_the_surrender_transm_70_clip.wav) (-14.0 LUFS, +2.1 dB crest factor)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Crest Factor | 12.9 dB | 15.0 dB | **+2.1 dB** |
| Peak-to-Loudness Ratio | 10.0 dB | 12.3 dB | **+2.3 dB** |
| True Peak | +1.5 dBTP | -1.7 dBTP | -3.1 dB |

---

Of Mice & Men — "O.G. Loko" (15s clip, 2:49–3:04). Processed with the `2000s-metalcore` preset at intensity 0.7.

- [Original clip](docs/samples/omm_ogloko_original_clip.wav) (brickwalled, -16.1 LUFS)
- [Remastered clip](docs/samples/omm_ogloko_transm_clip.wav) (restored dynamics, -16.0 LUFS, +6.3 dB crest factor)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Crest Factor | 11.1 dB | 17.4 dB | **+6.3 dB** |
| Peak-to-Loudness Ratio | 8.9 dB | 15.0 dB | **+6.1 dB** |
| True Peak | -7.1 dBTP | -1.0 dBTP | +6.1 dB |
| Clipping | 0.00% | 0.00% | -- |

> **Tuning note:** Screamed vocals are noticeably more muted in the remastered version. The de-esser (6–9 kHz) can't distinguish sibilance from scream harmonics, and the -1.5 dB vocal level cut compounds the effect. This is a known limitation of the current `2000s-metalcore` preset — see [TeeYum/transm#3](https://github.com/TeeYum/transm/issues/3) for tuning plans.

---

> Audio clips are short excerpts used for technical demonstration of audio processing. All rights belong to the original artists.

## Status

**Pre-alpha / Research Phase.** See the docs for the full architecture and feasibility assessment.

## Documentation

- [Architecture](docs/architecture.md) — Technical architecture, pipeline design, tech stack, and roadmap
- [Capture Component](docs/capture-component.md) — Lossless stream capture from Spotify, Apple Music, Tidal via system audio loopback
- [Feasibility Assessment](docs/feasibility-assessment.md) — Honest difficulty estimates, model fine-tuning feasibility, and adversarial analysis review

## Planned Stack

- **Separation**: `audio-separator` (Demucs, RoFormer, MDX-Net, ensemble)
- **DSP**: `pedalboard` + custom NumPy/SciPy (transient shaping, expansion, de-essing)
- **Analysis**: `pyloudnorm`, `librosa`
- **Mastering** (optional): `matchering`

## Contributing

This project uses a fork-based workflow. All development happens on the [chryst-monode fork](https://github.com/chryst-monode/transm) and is proposed to this repo via pull request. See [CLAUDE.md](CLAUDE.md) for agent-specific instructions.

All commits are co-created by [@TeeYum](https://github.com/TeeYum), [@chryst-monode](https://github.com/chryst-monode), and [Claude](https://claude.ai).

## License

GPL-3.0 (required by Pedalboard and Matchering dependencies)
