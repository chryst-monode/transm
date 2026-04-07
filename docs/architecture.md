# Transm: An Open-Source AI Remastering Engine for Loudness War-Era Metal

## A Technical Architecture Document for Senior Engineers and Audiophiles

---

## The Problem, Stated Honestly

Early 2000s metalcore sits in a uniquely painful spot in audio history. Albums like Atreyu's *The Curse*, Underoath's *They're Only Chasing Safety*, Norma Jean's *Bless the Martyr and Kiss the Child*, and From Autumn to Ashes' *Too Bad You're Beautiful* were mastered during the peak of the Loudness War. Engineers applied brickwall limiting with ratios of 10:1 or higher, crushing dynamic range to compete for perceived loudness on radio and in-store playback. The result: clipped transients, collapsed spatial depth, fatiguing cymbal harshness, and kick drums that sound like cardboard boxes instead of cannons.

Here's the uncomfortable truth: **clipping is destructive and irreversible.** Once a brickwall limiter chops the peaks off a snare hit, that information is gone from the waveform permanently. No AI model can perfectly reconstruct what was there. What AI *can* do is make educated, musically-informed guesses about what those peaks probably looked like, and re-expand the dynamics in a way that sounds dramatically better than the crushed original. The goal isn't perfection. The goal is making over-compressed masters breathe again.

This document proposes **Transm** (Transmute AI), an open-source Python application that chains together proven, production-ready AI models and DSP tools into a single pipeline purpose-built for this use case.

---

## What Actually Exists (Separating Reality from Hype)

### Production-Ready (Build On These)

**Demucs v4 (HTDemucs) by Meta/Kyutai**
A strong baseline for open-source source separation. The Hybrid Transformer architecture processes audio simultaneously in time domain (waveform convolutions) and frequency domain (spectrogram analysis), with cross-attention Transformers bridging both. It achieves 9.20 dB SDR on MUSDB HQ. For metalcore specifically, the 4-stem mode (vocals, drums, bass, other) is the right choice because the 6-stem mode tends to thin out guitar tone in dense mixes. GitHub: `adefossez/demucs`. Installable via pip.

**BS-RoFormer / Mel-Band RoFormer**
Newer separation architectures that outperform Demucs v4 on MUSDB18HQ benchmarks. BS-RoFormer achieves 11.99 dB SDR with extra training data (vs. Demucs's 9.20). Mel-Band RoFormer further improves on individual stems. Both are available via ZFTurbo's training repo and the `audio-separator` library. MIT-licensed.

**audio-separator (by nomadkaraoke)**
A unified Python wrapper supporting Demucs, MDX-Net, MDXC/RoFormer, and VR Arch models behind a single API. Supports CUDA, CoreML on Apple Silicon, and CPU. Includes ensembling strategies (average, median, min/max, spectral methods). This is the right abstraction layer for the separation backend. GitHub: `nomadkaraoke/python-audio-separator`.

**Spotify Pedalboard**
A Python library for studio-grade audio effects processing. It can load VST3 and Audio Unit plugins, apply EQ, compression, and gain staging, and integrates cleanly into ML pipelines. Actively maintained by Spotify, tested through Python 3.14. Note: does NOT include transient shaper or expander primitives — those must be implemented in NumPy/SciPy. GitHub: `spotify/pedalboard`.

**Matchering 2.0**
An open-source audio matching and mastering library. Feed it your processed audio and a modern reference track (say, a Spiritbox or Sleep Token master), and it will match RMS levels, frequency response, peak amplitude, and stereo width. Should be used as an optional reference tone-match stage, NOT the default mastering step (it can re-crush dynamics). GitHub: `sergree/matchering`.

**pyloudnorm**
ITU-R BS.1770-4 compliant loudness measurement. Essential for measuring LUFS and dynamic range before and after processing, so you can quantify improvement. GitHub: `csteinmetz1/pyloudnorm`.

**Librosa**
The standard Python audio analysis library. Spectrogram computation, beat tracking, onset detection, feature extraction. Not a processing tool, but essential for the analysis layer. GitHub: `librosa/librosa`.

### Real But Experimental (Use With Caution)

**AudioSR (Audio Super Resolution)**
A diffusion-based model that upsamples degraded audio to 48 kHz by hallucinating missing high-frequency content. Its own README warns it "was not trained to handle other causes of high-frequency loss, such as MP3 compression" — it was trained on low-pass filtering only. Worth testing per-stem rather than on the full mix, but treat as experimental. GitHub: `haoheliu/versatile_audio_super_resolution`.

**Apollo (Music Repair)**
A generative model for converting lossy MP3 audio toward lossless quality. CC BY-SA 4.0 licensed with no GitHub releases. Active development but not production-ready. GitHub: `JusperLee/Apollo`.

**Audio Declipping (Various)**
Multiple research implementations exist (`rajmic/declipping2020_codes`, `kripton/audio-declipper`, `pierreaguie/C-OMP-for-Audio-Declipping`). All are research-grade. None are production-ready for music. The ICASSP 2026 Music Source Restoration Challenge found the winning system achieved only 0.29 dB improvement on percussion restoration, quantifying how hard this problem remains.

### Not Worth Building On

**Open-Unmix**: 3 dB SDR behind Demucs. No reason to use it.
**Spleeter (Deezer)**: Obsolete. Demucs superseded it entirely.
**VoiceFixer**: Speech-focused. Does not generalize to music well.

---

## Architecture: The Transm Pipeline

```
                         Transm Pipeline
                         ===============

[Input: Crushed Master]
         |
         v
  +------+-------+
  | PRE-ANALYSIS |  <-- pyloudnorm, librosa
  | LUFS, LRA,   |      Measure: LUFS, DR, crest factor,
  | PLR, true     |      spectral centroid, clipping %,
  | peak, spectral|      true peak, spectral tilt
  | fingerprint   |
  +------+-------+
         |
         v
  +------+---------+
  | STEM SPLIT     |  <-- audio-separator (backend-agnostic)
  | Backend:       |      Default: Demucs
  | - Demucs       |      Advanced: RoFormer, MDX-Net
  | - RoFormer     |      Optional: Ensemble mode
  | - MDX-Net      |
  | - Ensemble     |
  +------+---------+
         |
         v
  +------+-------+
  | STEM QA      |  Bleed/artifact estimates,
  |              |  user warning if separation
  |              |  quality is poor
  +------+-------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
  [Vox][Drm][Bas][Gtr][Optional: AudioSR per-stem]
    |    |    |    |
    v    v    v    v
  +------+-------+
  | PER-STEM DSP |  <-- Pedalboard + custom NumPy/SciPy
  |              |
  | Drums:       |      Transient emphasis, clipped-region
  |              |      soft reconstruction, cymbal de-harsh
  | Vocals:      |      De-ess, level rebalance,
  |              |      avoid over-presence
  | Bass:        |      Mud removal, harmonic audibility,
  |              |      phase-safe low-end control
  | Guitars:     |      Fizz suppression, harsh resonance
  |              |      control, optional M/S width
  +------+-------+
         |
         v
  +------+-------+
  | REMIX / SUM  |  <-- Gain staging per stem,
  | Headroom     |      polarity/latency alignment,
  | management   |      summing with headroom
  +------+-------+
         |
         v
  +------+---------+
  | FINAL STAGE    |  Default: true-peak limiter,
  |                |  target -14 LUFS / -1.0 dBTP
  | Optional:      |  Optional: Matchering reference
  | ref EQ match   |  EQ match (not loudness match)
  +------+---------+
         |
         v
  +------+-------+
  | POST-ANALYSIS|  <-- pyloudnorm, librosa
  | Before/after |      Quantify improvement:
  | metrics +    |      delta LUFS, delta LRA, delta PLR,
  | A/B export   |      spectral comparison, artifact score
  +------+-------+
         |
         v
[Output: Remastered WAV/FLAC]
```

---

## Recommended Tech Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Separation | `audio-separator` | latest | Backend-agnostic stem splitting |
| DSP | `pedalboard` | 0.9.x | EQ, compression, limiting |
| Custom DSP | `numpy`, `scipy` | latest | Transient shaper, expander, de-esser |
| Loudness | `pyloudnorm` | latest | LUFS/LRA/PLR measurement |
| Analysis | `librosa` | latest | Spectral analysis, onset detection |
| I/O | `soundfile` | latest | Audio file read/write |

### Optional / Experimental

| Component | Library | Purpose |
|-----------|---------|---------|
| Mastering | `matchering` | Optional reference EQ matching |
| Super-resolution | `audiosr` | High-frequency restoration (experimental) |
| Visualization | `matplotlib` | Waveform/spectrogram plots |
| Model training | `Music-Source-Separation-Training` | Custom separator fine-tuning |

### GUI Options

For the initial release, ship as a **CLI tool**. Add a **Gradio** web interface once the pipeline is stable and tuned. For a polished v2, **PySide6 (Qt for Python)** with **pyqtgraph** for real-time waveform visualization.

---

## Per-Stem Processing Strategy (The Audiophile Part)

This is where domain knowledge matters more than model architecture. A senior audio engineer would process each stem differently because the Loudness War damaged each element in different ways.

### Drums

The drums suffer the most from brickwall limiting. Snare transients are chopped flat, kick drums lose their sub-bass thump, and cymbals become a wash of harsh sibilance.

Processing chain:
1. **Transient emphasis** (custom NumPy envelope follower, attack +3-6 dB, sustain -2 dB)
2. **Downward expander** (custom SciPy implementation, threshold ~-30 dB, ratio 1.5:1)
3. **High-shelf EQ cut** (-2 to -4 dB above 8 kHz) to tame cymbal harshness
4. **Low-shelf boost** (+2 dB below 80 Hz) to restore kick weight

### Vocals

Metalcore vocals (both clean and screamed) tend to be over-compressed and pushed forward in the mix at the expense of everything else.

Processing chain:
1. **De-esser** (custom band-specific compressor, 6-9 kHz) for sibilance control
2. **Light downward expansion** (ratio 1.2:1) to create more dynamic variation
3. **Presence EQ** (gentle +1-2 dB shelf around 3-5 kHz) for clarity without harshness
4. **Reduce level by 1-2 dB** relative to other stems (vocals were often too loud in 2000s masters)

### Bass

Bass guitar in early metalcore was often buried or turned into a low-mid mud puddle.

Processing chain:
1. **High-pass at 30 Hz** to remove sub-rumble
2. **Tight parametric cut** at 200-300 Hz (-3 dB, narrow Q) to clear mud
3. **Harmonic enhancement** around 800 Hz-1.2 kHz (+2 dB) for note definition
4. **Light compression** (2:1, slow attack) to even out the performance

### Guitars (Other Stem)

The "other" stem will contain guitars, synths, and anything that isn't vocals, drums, or bass. For metalcore, this is predominantly distorted guitar.

Processing chain:
1. **Mid-scoop reduction** (gently boost 500 Hz-2 kHz to undo the "scooped mids" of the era)
2. **High-shelf rolloff** (-2 dB above 10 kHz) to reduce fizz
3. **Stereo width enhancement** (subtle M/S processing to push guitars wider)
4. **Dynamic EQ** to tame resonant peaks that the limiter was masking

---

## Genre-Specific Presets

One of Transm's differentiators should be **genre-aware preset profiles**. The Loudness War hit different subgenres differently. A few starting presets:

**"2000s Metalcore" (Default)**
Target albums: Atreyu, Underoath, Killswitch Engage, As I Lay Dying
Characteristics: Extreme brickwall limiting, harsh cymbals, buried bass, vocals too loud
Strategy: Aggressive drum transient restoration, significant cymbal taming, bass recovery

**"Nu-Metal / Post-Grunge"**
Target albums: Deftones, Chevelle, Breaking Benjamin
Characteristics: Over-compressed but less clipped than metalcore, muddy low-mids
Strategy: Moderate expansion, mid-range clarity, sub-bass restoration

**"Pop-Punk / Emo"**
Target albums: Taking Back Sunday, Brand New, The Used
Characteristics: Bright and fatiguing, drums sound papery, bass is thin
Strategy: High-frequency taming, drum weight restoration, bass fullness

**"Post-Hardcore"**
Target albums: Glassjaw, Thursday, At the Drive-In
Characteristics: Dynamic range somewhat preserved but still compressed, chaotic mixes
Strategy: Light touch - mostly expansion and spatial enhancement

---

## CLI Interface (Ship This First)

```bash
transm analyze input.flac
transm separate input.flac --backend demucs
transm process input.flac --preset 2000s-metalcore --intensity 0.35
transm compare input.flac output.flac
```

Default preset should be conservative. Metalcore gets ugly fast if you over-expand separation artifacts or boost fake transients.

---

## What This Will and Won't Do

### It Will

- Meaningfully improve the listening experience of over-compressed 2000s metal on revealing headphones
- Restore some sense of dynamic range by separating stems and re-expanding them individually
- Allow per-stem EQ and gain adjustments that are impossible on a brickwalled master
- Optionally match the tonal profile of modern, well-mastered tracks
- Give you quantifiable before/after metrics (LUFS, LRA, PLR, true peak, spectral comparison)
- Run entirely on your own hardware with no cloud dependencies

### It Won't

- Perfectly reconstruct what the original, unmastered mix sounded like (that information is gone)
- Sound identical to a proper remix from the original multitrack session files
- Eliminate all separation artifacts (expect faint "watery" textures in quiet passages)
- Work equally well on all source material (MP3 sources will have more artifacts than CD rips)
- Replace the ears of a mastering engineer for critical listening decisions

### Honest Expectation Setting

Targets +2 to +5 dB improvement in peak-to-loudness ratio on suitable tracks, subject to artifact scoring and human A/B testing. That's the difference between "fatiguing after two songs" and "I can enjoy this whole album." It won't sound like it was mixed yesterday at Sterling Sound. But it will sound meaningfully, audibly better, especially on revealing audiophile-grade equipment where the flaws of the original master are most exposed.

---

## Legal Notes

Remastering music you personally own (CD rips, purchased digital files) for personal listening is legally straightforward. You cannot redistribute AI-remastered versions of copyrighted recordings without permission from the rights holders. The U.S. Copyright Office has explicitly stated that mechanical changes like noise reduction and format conversion do not create a new copyrightable work. Transm's documentation should make clear that it's for personal use with legally obtained source material.

---

## Roadmap

**v0.1 - Proof of Concept (4-6 Weekends)**
- CLI with `audio-separator` backend (Demucs default, RoFormer optional)
- Custom transient shaper and downward expander in NumPy/SciPy
- Conservative true-peak limiter (-14 LUFS / -1.0 dBTP)
- Before/after metrics (LUFS, LRA, PLR, true peak)
- Single "2000s Metalcore" preset with conservative defaults

**v0.2 - Evaluate and Tune (2-3 Months)**
- Evaluation harness with synthetic test data
- A/B testing framework
- Multiple genre presets
- Gradio web UI
- Batch processing for full albums

**v0.3 - Advanced Features (3-6 Months)**
- Ensemble separation mode
- Optional Matchering reference EQ matching
- Custom preset editor
- A/B comparison player in the UI
- FLAC/ALAC output options

**v0.4 - Custom Models (3-6 Months, Optional)**
- Fine-tune Mel-Band RoFormer on metalcore via ZFTurbo's training repo
- Source training data from open-license multitracks + synthetic generation

**v1.0 - Desktop Application**
- PySide6 native desktop app
- Real-time preview of DSP changes
- Plugin architecture for community presets and processing modules

---

## Key GitHub Repositories

| Project | URL | License |
|---------|-----|---------|
| audio-separator | github.com/nomadkaraoke/python-audio-separator | MIT |
| Demucs v4 | github.com/adefossez/demucs | MIT |
| BS-RoFormer | github.com/lucidrains/BS-RoFormer | MIT |
| Music-Source-Separation-Training | github.com/ZFTurbo/Music-Source-Separation-Training | MIT |
| Pedalboard | github.com/spotify/pedalboard | GPL-3.0 |
| Matchering | github.com/sergree/matchering | GPL-3.0 |
| pyloudnorm | github.com/csteinmetz1/pyloudnorm | MIT |
| Librosa | github.com/librosa/librosa | ISC |
| AudioSR | github.com/haoheliu/versatile_audio_super_resolution | - |
| Apollo | github.com/JusperLee/Apollo | CC BY-SA 4.0 |

Note: Pedalboard and Matchering are GPL-3.0. If included as dependencies, Transm must be GPL-3.0 compatible. Model weights have separate licenses that need auditing independently from code.

---

## Hardware Requirements

- **Minimum**: 16 GB RAM, any modern CPU (separation runs on CPU, ~5-10 minutes per track)
- **Recommended**: 16+ GB RAM, NVIDIA GPU with 6+ GB VRAM (10-50x faster separation via CUDA)
- **Apple Silicon**: CoreML acceleration supported via `audio-separator`
- **Storage**: ~4 GB for model weights, plus ~500 MB per track being processed (stems are large)

---

*This document was prepared as an architectural recommendation. The proposed stack uses only actively-maintained, open-source components with proven track records. No vaporware, no hype, just tools that work.*
