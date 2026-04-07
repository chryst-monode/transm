# Transm: Feasibility Assessment

## Can You Take an Existing Model and Improve It for Metalcore Remastering?

**Yes, and here's the honest difficulty breakdown.**

There are three layers to this problem, each with different feasibility profiles.

---

## Layer A: Stem Separation (Feasible, Medium Difficulty)

The separation models are the most mature part of the stack and the most amenable to improvement.

### Fine-tuning Demucs v4

Officially supported by Meta. The repo includes training documentation and a fine-tuning command (`dora run -f 955717e8 -d dset=auto_mus variant=finetune`). The catch is compute: the original model was trained on 8x V100 GPUs (32GB each), 600 epochs. Fine-tuning on a single consumer GPU is possible with reduced batch sizes but will be slow and require careful hyperparameter management. You'd need multi-track metalcore recordings (isolated stems) as training data, which is the real bottleneck since MUSDB18 is only 10 hours and skews pop/rock.

### Better Option: ZFTurbo's Music-Source-Separation-Training

This is MIT-licensed, actively maintained, and supports training BS-RoFormer, Mel-Band RoFormer, SCNet, MDX23C, and even Demucs4HT. It's the most practical path to training a genre-specific separator. BS-RoFormer achieves 11.99 dB SDR on MUSDB18HQ (vs. Demucs v4's 9.20 dB) when trained with extra data. Mel-Band RoFormer beats even that on individual stems.

### Difficulty Estimate

If you have access to a machine with a single A6000 (48GB) or equivalent, and can source or create 50-100 multi-track metalcore songs, you could fine-tune a RoFormer variant in 2-4 weeks of training time. The engineering work to set up the training pipeline is maybe 2-3 weekends. Getting the training data is the hard part.

### Ensembling as a Shortcut

The `audio-separator` library supports multiple ensembling strategies. Running Demucs + RoFormer + MDX-Net and ensembling the results often outperforms any single model, including fine-tuned ones. For a v1 product, offering ensemble mode is higher-ROI than training a custom model.

---

## Layer B: Per-Stem Restoration/Enhancement (Feasible, Low-Medium Difficulty)

### The DSP Gap

Pedalboard does NOT have transient shaper or expander classes. It has Compressor, various filters, Gain, Limiter, and the ability to load VST3/AU plugins. For an open-source tool that can't depend on proprietary plugins, you need to implement transient shaping and expansion yourself in NumPy/SciPy.

This is well-understood DSP: envelope following via `scipy.signal`, onset detection via `librosa.onset.onset_detect()`, and gain modulation based on the envelope. It's maybe 200-400 lines of Python for a solid transient shaper and downward expander. Not rocket science, but it needs careful tuning.

### Difficulty Estimate

A competent Python developer with some DSP knowledge could implement the per-stem processing chain in 2-3 weeks. The hard part isn't the code, it's tuning the parameters to sound good on metalcore specifically. That requires ears and iteration.

---

## Layer C: Declipping / Waveform Reconstruction (Not Feasible Today)

This is where the dream hits reality hardest. There are NO production-ready open-source neural declipping models for music as of April 2026. The most recent paper (February 2026, arXiv:2602.22279) shows a self-supervised approach that trains on clipped measurements only, but it's pure research with no shipping code.

### The Hard Numbers

The ICASSP 2026 Music Source Restoration Challenge found that percussion restoration averages only 0.29 dB improvement across competing teams, vs. 4.59 dB for bass. That tells you how hard drum transient reconstruction is. The honest answer: stem separation + expansion can create the *illusion* of restored dynamics, but the actual transient information in clipped drums cannot be recovered with current technology.

### The Promising Research Direction

The SonicMaster paper (arXiv:2508.03448) demonstrates taking clean audio, applying simulated degradation (EQ, compression, limiting, distortion), and training a model to reverse it. You could take well-mastered modern metalcore (Spiritbox, Knocked Loose, Sleep Token), apply simulated 2003-era brickwall limiting to create degraded versions, and train a restoration model on the pairs.

This is the most interesting research direction, but it's a research project, not a weekend hack. Expect 3-6 months of work and access to serious GPU compute (4-8x A100s for the diffusion-based approaches).

---

## Difficulty Summary

| Component | Difficulty | Time | Compute | Blocking? |
|-----------|-----------|------|---------|-----------|
| Fine-tune separator (RoFormer) on metalcore | Medium | 4-6 weeks | 1x A6000+ | No, existing models work OK |
| Implement custom DSP (transient shaper, expander) | Low-Medium | 2-3 weeks | CPU | No, basic EQ works as MVP |
| Train genre-specific restoration model | High | 3-6 months | 4-8x A100 | Yes, this is the novel contribution |
| Collect multi-track metalcore training data | Medium-Hard | Ongoing | N/A | Yes, data is the real bottleneck |

**Bottom line:** You can ship a useful tool in 4-6 weekends using existing models as-is with custom DSP. Making it *great* by fine-tuning the models specifically for metalcore is a 3-6 month project requiring meaningful GPU access and training data curation. The declipping/restoration layer is a genuine research problem that could take a year or more to solve well.

---

## Assessment of Codex's Adversarial Analysis

**Codex's critique was largely correct and well-researched.** Here's the point-by-point evaluation:

### Codex Was Right About

1. **"Do not make Demucs the only separator"** — Correct and important. Demucs v4's PyPI release is from September 2023 and it's no longer SOTA. BS-RoFormer (11.99 dB SDR) and Mel-Band RoFormer both outperform it on MUSDB18HQ. The `audio-separator` library by nomadkaraoke is the right abstraction layer: it wraps Demucs, MDX-Net, MDXC/RoFormer, and VR Arch models behind a unified API with ensembling support. Making the separator backend swappable from day one is the correct architectural decision.

2. **"Replace the fake DSP claims with implementable DSP"** — Correct. The original architecture described transient shapers and downward expanders but the proof-of-concept used static EQ and gain because Pedalboard doesn't have those primitives. Either implement them in NumPy/SciPy or make VST3 loading an explicit optional dependency. The core app should work without proprietary plugins.

3. **"Move Matchering out of the default path"** — Correct and subtle. Matchering matches RMS, frequency response, peak amplitude, AND stereo width to a reference. If your reference track is a modern, loud master (-8 LUFS), Matchering will push your carefully expanded output right back to being loud and crushed. The default final stage should be a conservative true-peak limiter targeting -14 LUFS (streaming standard) with a -1.0 dBTP ceiling. Reference matching should be opt-in.

4. **"Stop promising DR6 to DR12"** — Fair criticism. The original document said you could take a DR6 and make it "feel like" a DR10-12. That's aspirational marketing, not engineering. Codex's reframe of "+2 to +5 dB improvement in peak-to-loudness ratio on suitable tracks, subject to artifact scoring and human A/B testing" is more honest.

5. **"Add an evaluation harness before building a fancy UI"** — Correct. Shipping metrics first (LUFS, LRA, PLR, true peak, clipping count, spectral tilt, stem leakage) and a CLI that produces before/after comparisons is the right engineering sequence.

6. **"AudioSR and Apollo as experimental toggles only"** — Correct. AudioSR's own README says it "was not trained to handle other causes of high-frequency loss, such as MP3 compression." Apollo is CC BY-SA 4.0 with no GitHub releases. Neither belongs in the default pipeline.

7. **"Fix licensing posture"** — Correct. Pedalboard is GPL-3.0, Matchering is GPL-3.0. The combined work must be GPL-3.0. Model weights have separate licenses that need auditing independently from code.

### Codex Was Partially Wrong About

1. **"Lyria 3 is not relevant"** — Codex is right that Lyria doesn't belong in the restoration pipeline, but dismissed the synthetic test data generation angle too quickly. Using a music generation model to create genre-appropriate test fixtures that you can then degrade and use as training pairs is a legitimate approach that avoids copyright issues entirely.

### What Codex Missed

1. **The ZFTurbo training repo is the real unlock.** This MIT-licensed repo supports training BS-RoFormer, Mel-Band RoFormer, SCNet, and others on custom datasets. It's the most practical path to a genre-specific separator.

2. **The ICASSP 2026 MSR Challenge results quantify the hard limits.** Percussion restoration averages only 0.29 dB improvement. This fundamentally constrains what Transm can promise for drum transient restoration.

3. **Ensembling is probably more valuable than fine-tuning for v1.** Running multiple separation models and ensembling the results often outperforms any single model.

---

## Revised Phased Recommendation

### Phase 1: Ship Something (4-6 Weekends)

Build a CLI tool that:
- Uses `audio-separator` as the separation backend (Demucs default, RoFormer optional, ensemble available)
- Implements custom transient shaping and downward expansion in NumPy/SciPy (no VST dependency)
- Has a conservative final limiter targeting -14 LUFS / -1.0 dBTP
- Produces before/after metrics (LUFS, LRA, PLR, true peak, spectral comparison)
- Includes 3-4 genre presets with conservative default parameters
- Ships as a pip-installable CLI: `transm process track.flac --preset metalcore`

Skip Gradio, skip Matchering-as-default, skip AudioSR. Just make the core pipeline work and sound good.

### Phase 2: Evaluate and Tune (2-3 Months)

- Build the evaluation harness
- Create synthetic test data: take modern well-mastered metalcore, apply simulated brickwall limiting, measure how well Transm recovers
- A/B test with real humans on real headphones
- Tune the DSP parameters obsessively
- Add Gradio UI once the pipeline is stable

### Phase 3: Train Custom Models (3-6 Months, Optional)

- Use ZFTurbo's repo to fine-tune a Mel-Band RoFormer on metalcore stems
- Source training data from: Cambridge/Telefunken multitracks (open license), Mixing Secrets multitracks, synthetic generation
- This only makes sense if Phase 2 shows that separation quality is the bottleneck

### Phase 4: Research the Restoration Model (6-12 Months, Ambitious)

- Follow the SonicMaster approach: build paired datasets of clean + degraded metalcore
- Train a diffusion-based restoration model specifically for brickwall-limited audio
- This is the novel scientific contribution and the hardest part
- Only pursue if Phases 1-3 prove the product concept has demand

---

## The One-Line Summary

The tool is buildable, the separation models are good enough today, the hard part is tuning the DSP to sound right on metalcore specifically, and the declipping/restoration dream is still a research problem. Ship the pipeline with existing models first, then improve.
