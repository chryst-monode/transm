# Transm (Transmute AI)

**Open-source AI-powered remastering for Loudness War-era metal and rock.**

Transm separates crushed masters into individual stems using state-of-the-art AI models, applies genre-aware DSP processing to restore dynamics and reduce fatigue, and remixes them with proper headroom. Built for audiophiles who want their 2000s metalcore to sound less like it was mastered inside a trash compactor.

## Status

**Pre-alpha / Research Phase.** See the docs for the full architecture and feasibility assessment.

## Documentation

- [Architecture](docs/architecture.md) — Technical architecture, pipeline design, tech stack, and roadmap
- [Feasibility Assessment](docs/feasibility-assessment.md) — Honest difficulty estimates, model fine-tuning feasibility, and adversarial analysis review

## Planned Stack

- **Separation**: `audio-separator` (Demucs, RoFormer, MDX-Net, ensemble)
- **DSP**: `pedalboard` + custom NumPy/SciPy (transient shaping, expansion, de-essing)
- **Analysis**: `pyloudnorm`, `librosa`
- **Mastering** (optional): `matchering`

## Contributors

This project is co-created by [@TeeYum](https://github.com/TeeYum), [@chryst-monode](https://github.com/chryst-monode), and [Claude](https://claude.ai).

## License

GPL-3.0 (required by Pedalboard and Matchering dependencies)
