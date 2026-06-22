# Tutorial 3: Physics-Informed Enhancement Models for Microscopy Image Restoration

## Overview

This tutorial covers the three physics-informed deep learning models implemented for enhancing low-quality phase-contrast microscopy images. These models incorporate knowledge of the image formation process, point spread function (PSF), and noise statistics into the network architecture and loss function.

## Image Formation Model

The forward model of microscopy image formation:

```
g(x,y) = η · [h(x,y) * f(x,y)] + n(x,y)
```

where:
- $f(x,y)$ = true sample
- $h(x,y)$ = point spread function (PSF)
- $\eta$ = photon count (Poisson process)
- $n(x,y)$ = detector noise (Gaussian)
- $g(x,y)$ = observed image

Physics-informed models use this forward model to constrain the solution space.

## Model 1: DeBCR (Denoising, Deblurring, optical Deconvolution by CNN and Wavelets)

### Architecture

```
Input → Wavelet Decomposition → Per-band CNN Denoising → Wavelet Reconstruction → Deconvolution → Output
```

### Key Components

**Wavelet Decomposition**: 4-level Dual-Tree Complex Wavelet Transform (DTCWT) separates the image into frequency bands. This is physics-informed because wavelet decomposition implicitly respects the frequency content structure of microscopy images.

**Per-band CNN Denoising**: Each wavelet band is denoised by a separate CNN branch. The CNN learns noise statistics specific to each frequency band.

**Richardson-Lucy Deconvolution**: After denoising, the estimated PSF is used for deconvolution:

```
f^(k+1)(x,y) = f^(k)(x,y) · [ (g / (f^(k) * h)) * h^T ](x,y)
```

### Implementation

```python
import torch
import torch.nn as nn
import pywt

class DeBCRInspired(nn.Module):
    def __init__(self, n_levels=4, n_channels=64):
        super().__init__()
        self.n_levels = n_levels
        
        # Per-band denoising CNNs
        self.denoise_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, n_channels, 3, padding=1),  # 2 for complex wavelet
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_channels, 2, 3, padding=1)
            ) for _ in range(n_levels * 6)  # 6 subbands per level
        ])
        
    def forward(self, x):
        # Wavelet decomposition
        coeffs = pywt.wavedec2(x, 'db4', level=self.n_levels)
        
        # Denoise each band
        denoised_coeffs = []
        for i, band in enumerate(coeffs):
            if isinstance(band, tuple):
                band_denoised = tuple(
                    self.denoise_nets[i*6+j](band[j].unsqueeze(0)).squeeze(0)
                    for j, b in enumerate(band)
                )
                denoised_coeffs.append(band_denoised)
            else:
                denoised_coeffs.append(
                    self.denoise_nets[i](band.unsqueeze(0)).squeeze(0)
                )
        
        # Wavelet reconstruction
        restored = pywt.waverec2(denoised_coeffs, 'db4')
        
        # Richardson-Lucy deconvolution (5 iterations)
        psf = self.estimate_psf(x)
        restored = self.richardson_lucy(restored, psf, iterations=5)
        
        return restored
    
    def estimate_psf(self, x):
        """Estimate PSF from image statistics."""
        # Simplified: use Gaussian PSF with estimated sigma
        sigma = self.psf_sigma_net(x.mean(dim=0))
        return self.gaussian_psf(sigma)
    
    def richardson_lucy(self, img, psf, iterations=5):
        """Richardson-Lucy deconvolution."""
        estimate = img.clone()
        psf_mirror = torch.flip(psf, dims=(-2, -1))
        for _ in range(iterations):
            reblurred = torch.nn.functional.conv2d(estimate, psf, padding='same')
            ratio = img / (reblurred + 1e-8)
            estimate = estimate * torch.nn.functional.conv2d(ratio, psf_mirror, padding='same')
        return estimate
```

### Training

```python
def train_debcr(model, train_loader, epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        for lq, hq in train_loader:
            restored = model(lq)
            
            # Combined loss: L1 + perceptual + forward consistency
            loss_l1 = nn.L1Loss()(restored, hq)
            loss_perceptual = perceptual_loss(restored, hq)
            loss_forward = forward_consistency_loss(restored, lq, model.psf)
            
            loss = loss_l1 + 0.1 * loss_perceptual + 0.01 * loss_forward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
```

## Model 2: PI-DDPM (Physics-Informed Denoising Diffusion Probabilistic Model)

### Architecture

The diffusion model gradually denoises an image through $T$ steps:

```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε,  ε ~ N(0, I)
```

At each step, the network predicts the noise $\epsilon_\theta(x_t, t)$.

### Physics Constraint

The key innovation is adding a physics-based loss term:

```
L_total = L_diffusion + λ · ||h * f_θ(x_t, t) - g||²
```

where $h$ is the PSF and $g$ is the observed noisy image. This ensures the restored image, when convolved with the PSF, is consistent with the observation.

### Implementation

```python
class PIDDPMInspired(nn.Module):
    def __init__(self, T=1000, channels=64):
        super().__init__()
        self.T = T
        
        # Noise schedule
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # U-Net denoiser
        self.denoiser = UNetDenoiser(channels=channels)
        
    def forward_diffusion(self, x_0, t):
        """Add noise at timestep t."""
        noise = torch.randn_like(x_0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def reverse_step(self, x_t, t, psf=None, observed=None):
        """Single denoising step with physics constraint."""
        # Predict noise
        noise_pred = self.denoiser(x_t, t)
        
        # Standard DDPM reverse step
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        
        x_prev = (1 / torch.sqrt(alpha)) * (
            x_t - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
        )
        
        # Physics constraint: project onto consistent solution
        if psf is not None and observed is not None:
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            consistency = torch.nn.functional.conv2d(x_0_pred, psf, padding='same')
            correction = observed - consistency
            x_prev = x_prev + 0.01 * correction
        
        return x_prev
    
    def sample(self, x_T, psf=None, observed=None):
        """Full reverse diffusion sampling."""
        x = x_T
        for t in reversed(range(self.T)):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            x = self.reverse_step(x, t_batch, psf, observed)
        return x
```

## Model 3: PSF-Learning (Zernike-Parameterized PSF Estimation)

### Zernike Polynomials

The PSF is parameterized using Zernike polynomials:

```
h(x,y) = |FT[exp(i · Σ_k a_k · Z_k(ρ, θ))]|²
```

where $Z_k$ are Zernike polynomials and $a_k$ are the coefficients to be learned.

### Implementation

```python
class PSFLearning(nn.Module):
    def __init__(self, n_zernike=15):
        super().__init__()
        self.n_zernike = n_zernike
        
        # Encoder for PSF parameter estimation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, n_zernike)  # Zernike coefficients
        )
        
    def forward(self, x):
        # Estimate Zernike coefficients
        coeffs = self.encoder(x)
        
        # Generate PSF from coefficients
        psf = self.zernike_to_psf(coeffs, x.shape[-2:])
        
        return psf, coeffs
    
    def zernike_to_psf(self, coeffs, size):
        """Convert Zernike coefficients to PSF."""
        # Create polar coordinate grid
        M, N = size
        u = torch.linspace(-1, 1, M)
        v = torch.linspace(-1, 1, N)
        U, V = torch.meshgrid(u, v, indexing='ij')
        R = torch.sqrt(U**2 + V**2)
        Theta = torch.atan2(V, U)
        
        # Compute Zernike polynomials
        Z = self.zernike_polynomials(R, Theta)  # (n_zernike, M, N)
        
        # Wavefront
        wavefront = torch.sum(coeffs.unsqueeze(-1).unsqueeze(-1) * Z.unsqueeze(0), dim=1)
        
        # Pupil function
        pupil = (R <= 1.0).float() * torch.exp(1j * wavefront)
        
        # PSF = |FT(pupil)|²
        psf = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil))) ** 2
        psf = psf / psf.sum()
        
        return psf
```

## Training Protocol

### Data Preparation

```python
# Training pairs: (LQ, HQ) images
# LQ images generated by synthetic degradation pipeline
# HQ images from LIVECell annotated subset

train_pairs = [
    (synthetic_lq, original_hq)
    for lq, hq in zip(synthetic_lq_images, hq_images)
]
```

### Training Configuration

| Parameter | DeBCR | PI-DDPM | PSF-Learning |
|-----------|-------|---------|--------------|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 16 | 8 | 16 |
| Epochs | 100 | 200 | 100 |
| Loss | L1 + Perceptual + Forward | Diffusion + Physics | MSE + PSF consistency |
| GPU memory | ~8 GB | ~16 GB | ~6 GB |
| Training time | ~4 hours | ~24 hours | ~3 hours |

## Evaluation Results

| Model | Mean IoU (noise_50) | Mean IoU (combined) | Δ vs Raw |
|-------|---------------------|---------------------|----------|
| Raw | 0.2837 | 0.2590 | — |
| DeBCR | 0.2772 | 0.2511 | -0.007 |
| PI-DDPM | 0.2782 | 0.2557 | -0.002 |
| DoG filter | 0.2810 | 0.2801 | +0.022 |
| **DeBCR+DoG** | — | **0.3162** | **+0.057** |

## Key Findings

1. **DeBCR alone does not improve segmentation** — wavelet denoising smooths cell boundaries
2. **Enhancement + filtering synergy** — DeBCR+DoG achieves 2× improvement over DoG alone
3. **Physics constraint matters** — forward consistency loss improves restoration quality
4. **Combined pipeline is essential** — no single model solves the quality gap alone

## Source Code

- DeBCR: `src/phaseA_physics_models.py`
- PI-DDPM: `src/ws1_physics_models.py`
- PSF-Learning: `src/phaseA_physics_models.py`
- Training: `src/ws1_physics_models.py`
- Visual comparison: `src/phaseB_visual_comparison.py`

## References

1. Li, X. et al. (2024). DeBCR: Denoising, Deblurring, and optical Deconvolution. bioRxiv.
2. Wang, Y. et al. (2024). Physics-informed DDPM for microscopy. Nat. Commun. Eng., 3, 186.
3. CVPR 2025. Physics-Informed Blur Learning Framework.
