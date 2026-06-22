# Physics-Informed Models for Microscopy Image Enhancement

A detailed reference on models that incorporate the physics of light propagation,
optical systems, and microscope-specific knowledge into deep learning architectures.

This document serves as the theoretical foundation for the physics-informed
enhancement work described in the [manuscript](../manuscript/ms_manuscript.tex)
and [scientific report](REPORT.md).

---

## 1. What Are Physics-Informed Models?

### 1.1 Definition

Physics-informed models are neural networks that incorporate **known physical laws**
into their architecture, loss function, or training process. Instead of learning
everything from scratch (purely data-driven), they are **constrained by physics**
to produce results that are physically consistent.

### 1.2 The Core Idea

In microscopy, image formation follows well-known physics:

```
Image = (Object ⊗ PSF) + Noise
```

where:
- **⊗** = convolution operation
- **PSF** = Point Spread Function (how the microscope blurs a point of light)
- **Noise** = photon noise, detector noise, etc.

A **pure data-driven model** (like standard CARE) learns:
```
Restored = f(Noisy Image) → learned from examples
```

A **physics-informed model** learns:
```
Restored = f(Noisy Image, PSF, Noise Model) → constrained by physics
```

### 1.3 Types of Physics Integration

| Integration Level | How Physics Is Used | Example |
|-------------------|---------------------|---------|
| **Loss function** | Physics-based penalty terms in training loss | Deconvolution consistency loss |
| **Architecture** | Network layers implement physical operations | Convolution with known PSF |
| **Data augmentation** | Simulate realistic degradation using physics | Forward model for training pairs |
| **Parameter estimation** | Network learns physical parameters (PSF, noise level) | Blind deconvolution |
| **Hybrid pipeline** | Physics-based preprocessing + neural network | Richardson-Lucy + CNN |

---

## 2. Key Physics Concepts in Microscopy

### 2.1 Point Spread Function (PSF)

The PSF describes how the microscope images a **point source of light**.
It is the fundamental building block of image formation.

```
I(x,y) = O(x,y) * PSF(x,y) + N(x,y)
```

where I = observed image, O = true object, * = convolution, N = noise.

**For phase-contrast microscopy specifically:**
- The PSF has a characteristic **ringing pattern** (phase halo)
- The halo creates the bright/dark edges around cells
- Knowledge of this PSF can guide restoration

### 2.2 Optical Transfer Function (OTF)

The OTF is the Fourier transform of the PSF. It tells us which spatial
frequencies are preserved and which are attenuated.

```
OTF(u,v) = FFT(PSF(x,y))
```

**Key insight**: Frequencies where OTF ≈ 0 are **lost** — no amount of
processing can recover them. Physics-informed models know this and don't try.

### 2.3 Noise Models

| Noise Type | Source | Distribution | Physics Known? |
|------------|--------|-------------|----------------|
| **Photon (shot) noise** | Quantum nature of light | Poisson | Yes |
| **Detector noise** | Electronics | Gaussian | Yes |
| **Readout noise** | Sensor readout | Gaussian | Yes |
| **Autofluorescence** | Sample_background | Mixed | Partially |

Physics-informed models use the **correct noise model** rather than assuming
Gaussian noise. This matters because Poisson noise is signal-dependent — brighter
regions have more noise.

### 2.4 Computational Forward Model

The complete image formation model:

```
g(x,y) = η [ h(x,y) * f(x,y) ] + n(x,y)
```

where:
- f(x,y) = true sample
- h(x,y) = PSF
- η = photon count (Poisson process)
- n(x,y) = detector noise
- g(x,y) = observed image

Physics-informed models use this model to **constrain** the solution.

---

## 3. Key Physics-Informed Models

### 3.1 DeBCR (Denoising, Deblurring, and optical Deconvolution)

- **Paper**: Li et al., bioRxiv 2024 / ResearchGate
- **URL**: https://github.com/leeroyhannover/DeBCR
- **Architecture**: Wavelet-based CNN with physics constraints
- **Physics used**:
  - Wavelet decomposition separates scales (physics of frequency content)
  - Optical deconvolution using estimated PSF
  - Noise model: Poisson + Gaussian混合
- **Key innovation**: Combines wavelet theory (mathematical physics) with CNN learning
- **Advantages**: Light model, fast runtime, works on both light microscopy and cryo-ET
- **Training**: Supervised with paired data
- **Code**: Available on GitHub

**How it works:**
```
1. Wavelet decomposition → separate image into frequency bands
2. Physics-informed noise estimation → estimate noise per band
3. CNN denoising → clean each band with learned filters
4. Physics-based deconvolution → apply PSF correction
5. Wavelet reconstruction → recombine bands
```

### 3.2 Physics-Informed Denoising Diffusion Probabilistic Model (PI-DDPM)

- **Paper**: Nature Communications Engineering, 2024
- **DOI**: 10.1038/s44172-024-00331-z
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11683148
- **Architecture**: Conditional Diffusion Model with physics loss
- **Physics used**:
  - Forward model: g = h*f + n (image formation equation)
  - PSF-aware conditioning
  - Poisson noise model in the diffusion process
- **Key innovation**: The diffusion process (gradual denoising) is constrained
  to produce images that are consistent with the known PSF and noise model
- **Advantages**: State-of-the-art quality, physically consistent, handles
  uncertainty (can generate multiple plausible restorations)
- **Limitations**: Slow inference (diffusion requires many steps)

**How it works:**
```
1. Start with pure noise
2. At each denoising step:
   a. Neural network predicts less noisy image
   b. Physics constraint: check consistency with PSF and noise model
   c. Project onto physically plausible solution space
3. Final output: restored image that satisfies both data and physics
```

### 3.3 Physics-Informed PSF Learning (CVPR 2025)

- **Paper**: "A Physics-Informed Blur Learning Framework for Imaging Systems"
- **Conference**: CVPR 2025
- **Architecture**: Seidel/wavefront PSF model + learning
- **Physics used**:
  - Wavefront aberration theory (Seidel aberrations)
  - PSF is parameterized by Zernike polynomials
  - Spatially varying PSF across field of view
- **Key innovation**: Instead of learning PSF from scratch, uses physical
  parameterization (Zernike coefficients) → fewer parameters, more robust
- **Advantages**: High accuracy PSF estimation, no lens parameters needed,
  generalizable across microscope types

### 3.4 CryoGEM (NeurIPS 2024)

- **Paper**: "CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy"
- **Conference**: NeurIPS 2024
- **Physics used**: CTF (Contrast Transfer Function) of electron microscopy
- **Innovation**: Physics-informed generative model for cryo-EM
- **Relevance**: Demonstrates physics-informed approach for electron microscopy

### 3.5 DeepInMiniscope (Science Advances)

- **Paper**: "DeepInMiniscope: Deep learning–powered physics-informed..."
- **Journal**: Science Advances
- **Physics used**: Light-field propagation physics for miniscope imaging
- **Application**: 3D imaging from 2D miniscope data
- **Innovation**: Physics of wave propagation used to constrain 3D reconstruction

---

## 4. Physics-Informed Loss Functions

The key innovation in physics-informed models is the **loss function**. Instead of
just comparing to a ground truth image, they add physics-based constraints.

### 4.1 Standard Loss (Data-Driven)
```
L = ||f_θ(x) - y||²
```
where f_θ = neural network, x = input, y = target

### 4.2 Physics-Informed Loss
```
L = ||f_θ(x) - y||²          ← Data fidelity (match target)
  + λ₁ ||h * f_θ(x) - x||²  ← Forward consistency (convolved output should match input)
  + λ₂ R(f_θ(x))              ← Physics regularization (smoothness, non-negativity)
  + λ₃ N(f_θ(x), x)           ← Noise model consistency
```

**Forward consistency** is the key: if we take the restored image and convolve
it with the known PSF, we should get something close to the original noisy image.
This constrains the network to produce solutions that are physically consistent.

### 4.3 Noise Model Loss

For Poisson-Gaussian noise:
```
L_noise = Σ [f_θ(x) + σ² - x · log(f_θ(x) + σ² + ε)]
```

This is the **negative log-likelihood** of the Poisson-Gaussian noise model.
The network learns to produce images that, when passed through the noise model,
would generate the observed noisy image.

---

## 5. Advantages of Physics-Informed Models

### 5.1 vs. Pure Data-Driven Models

| Aspect | Pure Data-Driven | Physics-Informed |
|--------|-----------------|------------------|
| **Training data needed** | Large (thousands of pairs) | Smaller (physics constrains) |
| **Generalization** | Limited to training distribution | Better across conditions |
| **Physical consistency** | Not guaranteed | Guaranteed |
| **Interpretability** | Black box | Parameters have physical meaning |
| **Noise handling** | Learned implicitly | Explicit noise model |
| **PSF knowledge** | Learned implicitly | Used explicitly |
| **Out-of-distribution** | May fail badly | More robust |

### 5.2 vs. Traditional Image Processing

| Aspect | Traditional (RL, Wiener, etc.) | Physics-Informed ML |
--------|-------------------------------|---------------------|
| **Prior knowledge** | Hand-crafted regularizers | Learned + physics |
| **Adaptability** | Fixed | Adapts to data |
| **Noise model** | Often assumed Gaussian | Can use correct model |
| **Speed** | Fast | Moderate to fast |
| **Quality** | Limited by hand-crafted priors | Can exceed traditional methods |
| **Blind deconvolution** | Difficult | Easier with CNN |

---

## 6. Implementation for Your Project

### 6.1 For Our LIVECell Phase-Contrast Images

**What physics do we know?**
1. **PSF**: Phase-contrast has a characteristic PSF with halo
2. **Noise**: Photon noise (Poisson) + detector noise (Gaussian)
3. **Image formation**: Phase contrast converts phase shifts to intensity

**How to apply physics-informed models:**

**Option A: PSF-aware denoising**
```python
# 1. Estimate PSF from phase-contrast images
#    (e.g., from bead images or theoretical model)
# 2. Use PSF in loss function:
#    L = ||f_θ(x) - y||² + λ||PSF * f_θ(x) - x||²
# 3. Train on our LIVECell HQ/LQ pairs
```

**Option B: Noise model-aware restoration**
```python
# 1. Use Poisson-Gaussian noise model
# 2. Loss = negative log-likelihood of observed noise
# 3. Network learns to denoise with correct noise statistics
```

**Option C: Use DeBCR (simplest)**
```python
# 1. Install from GitHub
# 2. Train on our HQ/LQ pairs
# 3. Wavelet decomposition handles physics implicitly
```

### 6.2 Practical Steps

1. **Estimate your PSF**: Use sub-resolution beads or theoretical phase-contrast PSF
2. **Choose noise model**: Poisson-Gaussian (standard for microscopy)
3. **Select model**: DeBCR (easiest) or PI-DDPM (best quality)
4. **Train on our pairs**: LIVECell HQ → LQ (our synthetic degradations)
5. **Compare**: Raw vs. Enhanced vs. Enhanced+Filter

### 6.3 Expected Improvements

| Scenario | Bandpass Filter Only | Physics-Informed Only | Combined |
|----------|---------------------|----------------------|----------|
| Noise σ=50 | +0.003 IoU | +0.08-0.12 IoU | +0.10-0.18 IoU |
| Defocus σ=4 | +0.005 IoU | +0.05-0.10 IoU | +0.08-0.15 IoU |
| Shading α=0.5 | +0.010 IoU | +0.03-0.06 IoU | +0.05-0.10 IoU |

---

## 7. References

1. Li et al. (2024). DeBCR: Denoising, Deblurring, and optical Deconvolution
   for cryo-ET and light microscopy. bioRxiv. https://github.com/leeroyhannover/DeBCR

2. Physics-informed denoising diffusion probabilistic model for microscopy.
   Nature Communications Engineering, 3:186, 2024.
   DOI: 10.1038/s44172-024-00331-z

3. A Physics-Informed Blur Learning Framework for Imaging Systems.
   CVPR 2025. https://cvpr.thecvf.com/virtual/2025/poster/32605

4. CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy.
   NeurIPS 2024.

5. DeepInMiniscope: Deep learning–powered physics-informed...
   Science Advances. https://www.science.org/doi/10.1126/sciadv.adr6687

6. Physics informed image restoration under low illumination.
   Optics Express, 33(3):6121. https://opg.optica.org/oe/abstract.cfm?uri=oe-33-3-6121

7. Roadmap on Deep Learning for Microscopy. EPFL, 2024.
   https://bigwww.epfl.ch/preprints/volpe2501p.pdf
