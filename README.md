<div align="center">

# 🎧 SEGAN

### Speech Enhancement with a 1D-CNN U-Net on Mel-Spectrograms

*Denoising speech using a conditional GAN, trained with UrbanSound8K noise and evaluated with PESQ, STOI and SNR.*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep_Learning-GAN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Audio](https://img.shields.io/badge/Audio-Denoising-6C63FF?style=for-the-badge)

</div>

---

## 📖 About

Real-world speech is rarely clean: voice calls, hearing aids, conferencing and voice assistants all operate in noisy environments. Noise hurts both **intelligibility** (can you understand it?) and **perceived quality** (does it sound clean?).

This project, built for the course **Data Mining and Learning Systems (Group 22)**, learns a mapping from noisy speech to an estimate of the clean signal, without distorting the speech itself.

## 💡 What Is SEGAN?

**SEGAN** (Speech Enhancement GAN) was introduced by Pascual et al. (2017) as the first end-to-end GAN that maps a noisy waveform directly to a clean one.

| 🎯 The Original Idea | 🔧 Our Adaptation |
|---|---|
| A Generator turns noisy speech into clean speech | Works on **mel-spectrograms** instead of raw waveforms |
| A Discriminator tells real clean speech from the Generator's output | Generator is a **1D-CNN U-Net** that predicts a **gain mask** |
| G tries to fool D, D tries not to be fooled | Discriminator is a **conditional PatchGAN**, conditioned on the noisy input |

## 🧮 Problem Definition

```
y = x + α·n
```

- `x`: clean speech waveform
- `n`: additive urban noise waveform
- `α`: scaling factor that sets the signal-to-noise ratio
- **Training SNR: 5 dB (fixed)**

The goal is to learn `Gθ : y ↦ x̂` so that `x̂` is as close as possible to `x`, judged on both waveform-level (SNR) and perceptual (PESQ, STOI) criteria.

## 🏗️ Approach

### 🔊 Why Mel-Spectrograms?
- Perceptually motivated, compressed time-frequency representation
- Much shorter sequences than raw waveforms
- Focuses capacity on the frequencies that matter for speech perception
- Trade-off: a magnitude-to-waveform reconstruction step is needed at inference (the noisy phase is reused)

### 🥊 Why a Conditional GAN?
Pure L1/L2 regression tends to over-smooth and blurs harmonic structure. The Discriminator pushes the Generator toward outputs that are distributionally indistinguishable from real clean speech, not just numerically close.

```
min_G max_D  E[log D(y,x)] + E[log(1 − D(y,G(y)))] + λ·E[|G(y) − x|]
```

### ⚙️ Configuration

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz |
| Mel bins | 64 |
| Training SNR | 5 dB (fixed) |
| Train / validation split | 90 / 10 (seeded) |

## 🧱 Architecture

- 🟦 **Generator:** 1D-CNN U-Net that predicts a gain mask `M ∈ [0,1]` per mel-bin (sigmoid), applied to the noisy signal's own power: `enhanced_power = M × noisy_power`
- 🟥 **Discriminator:** 1D-CNN PatchGAN, conditioned on the noisy input

### 🛡️ Stability & Loss
- Spectral normalization on every Discriminator conv layer
- Smaller Discriminator capacity (32–512 channels)
- One-sided label smoothing (0.9)
- G:D update ratio of 4:1
- 5-epoch L1-only pretraining for the Generator
- Feature Matching loss (λ = 10)
- Mask Suppression loss for silent regions (λ = 5)
- Multi-resolution spectral loss (λ = 0.5) and differentiable waveform loss (λ = 0.5)

## 🐛 The Debugging Journey

The first versions made every metric worse on every sample:

| Metric | Noisy | Enhanced (buggy) | Δ |
|---|---|---|---|
| SNR (dB) | 5.00 | 1.64 | −3.36 |
| PESQ | 1.59 | 1.29 | −0.30 |
| STOI | 0.895 | 0.856 | −0.04 |

> 🔍 **Key insight:** a model that simply copies its noisy input would score Δ = 0, not a large negative delta. A consistent negative delta across all three metrics, on every sample, points to a **systematic bug**, not undertraining.

**Root cause: a normalization mismatch** between training and inference.

**The fix:**
1. 📏 One shared, fixed normalization `(mel_db + 80) / 40`, used identically in the dataset, training and inference, replacing per-utterance, per-bin mean/std
2. 🎭 Mask-based Generator output that preserves natural harmonic texture where speech dominates
3. 🔁 Multi-resolution losses kept, now stable thanks to the normalization fix
4. ⚖️ Better GAN balance (4:1 update ratio, Feature Matching, Mask Suppression)
5. ✅ Held-out validation split, so reported metrics never use data seen in training

## 📊 Results

Evaluated on **50 held-out samples** at 5 dB input SNR:

| Metric | Noisy | Enhanced | Δ |
|---|---|---|---|
| 🗣️ **PESQ** | 1.594 | **2.169** | **+0.575** |
| 🧠 **STOI** | 0.895 | **0.910** | **+0.015** |
| 📉 **SNR (dB)** | 5.000 | 2.847 | −2.153 |

- ✅ PESQ and STOI improved, so perceptual quality and intelligibility are better
- ⚠️ SNR drops because mask-based suppression slightly reduces overall power
- 🔉 Residual background noise is attenuated but not fully eliminated

## 🎬 Live Demo

The notebook includes a live demo cell:

1. 📂 Upload a WAV file, or pick a held-out validation sample with the slider
2. 🎚️ Set the input SNR (default: 5 dB)
3. ▶️ Run inference with `denoise_waveform()`, the full pipeline end-to-end
4. 🎧 Compare **Noisy**, **Clean** and **Enhanced** audio side by side

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <this-repository-url>
   ```
2. **Open the notebook** in Jupyter or Google Colab
3. **Run the cells in order** to load the data, train the model and evaluate it
4. **Run the Live Demo cell** to listen to the results

## 🔮 Future Work

- 🎙️ Neural vocoder (e.g. HiFi-GAN) instead of reusing the noisy phase
- 📈 Progressive SNR curriculum, from easy to hard
- 🌆 Larger and more diverse noise corpus beyond UrbanSound8K
- 🧩 Conformer / Transformer blocks for longer temporal context

## 💭 What We Learned

- A consistent negative delta across every metric is the signature of a systematic bug, not undertraining
- Normalization mismatches between training and inference are a silent, severe failure mode, so use one shared transform everywhere
- GAN stability tricks matter, but cannot compensate for a broken data pipeline
- Mask-based (Wiener-style) output preserves natural speech timbre better than absolute magnitude prediction

## 📚 Datasets & References

**Datasets:** LibriSpeech (test-clean subset) · UrbanSound8K

**References**
- Pascual et al. (2017). *SEGAN: Speech Enhancement Generative Adversarial Network*
- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*
- Miyato et al. (2018). *Spectral Normalization for Generative Adversarial Networks*
- Salimans et al. (2016). *Improved Techniques for Training GANs*
- ITU-T P.862. *Perceptual Evaluation of Speech Quality (PESQ)*
- Allen (1977). *Short-Time Spectral Analysis, Synthesis, and Modification by DFT*

---

<div align="center">
Made with ☕ and a lot of debugging
</div>
