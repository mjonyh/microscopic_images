# Image Enhancement Models for Microscopy: Beyond Bandpass Filters

A comprehensive reference on models that can **actively restore/enhance** microscopy images,
rather than just filtering them. These can be used as preprocessing before the FFT
filter pipeline, or as alternatives to it.

## 1. Why Enhancement Models?

Our analysis showed that bandpass filter improvements on low-quality images are
**10–100× smaller** than on high-quality images. This is because filters can only
remove interference — they cannot recover information lost during acquisition.

**Enhancement models** (especially deep learning-based) can:
- **Denoise** beyond the noise floor (learn signal vs. noise patterns)
- **Deblur** by learning the inverse of the point-spread function
- **Super-resolve** beyond the diffraction limit
- **Correct illumination** by learning the shading pattern
- **Combine multiple degradations** in a single model

## 2. Model Taxonomy

### 2.1 By Architecture

| Architecture | Strengths | Weaknesses | Best For |
|-------------|-----------|------------|----------|
| **U-Net** | Good localization, skip connections | Limited global context | Segmentation, denoising |
| **RCAN** (Residual Channel Attention) | Excellent detail recovery | Computationally expensive | Super-resolution |
| **SwinIR** (Swin Transformer) | Global receptive field | Needs large datasets | General restoration |
| **Diffusion Models** | Best quality, handles uncertainty | Very slow inference | Research, offline processing |
| **GANs** (ESRGAN, Real-ESRGAN) | Sharp results, fast | Training instability, artifacts | Super-resolution |
| **CNN-Transformer Hybrid** | Local + global features | Complex architecture | General restoration |
| **Physics-Informed** | Incorporates optical physics | Needs PSF knowledge | Specific microscope types |

### 2.2 By Task

| Task | Input → Output | Key Models |
|------|---------------|------------|
| **Denoising** | Noisy → Clean | CARE, Noise2Void, N2N, DnCNN, Diffusion |
| **Deblurring** | Blur → Sharp | DeblurGAN, Restormer, NAFNet |
| **Super-resolution** | Low-res → High-res | RCAN, ESRGAN, Deep-STORM |
| **Illumination correction** | Shaded → Uniform | Homomorphic+DL, U-Net shading |
| **Multi-task** | Degraded → Clean+Sharp+SR | SwinIR, Restormer, CVDM |

## 3. Key Models for Microscopy

### 3.1 CARE (Content-Aware Image Restoration)
- **Paper**: Kruse et al., Nature Methods 2017
- **URL**: https://henriqueslab.org/pages/care.html
- **Architecture**: U-Net variant
- **Training**: Supervised (paired LQ/HQ images)
- **Strengths**: Works on fluorescence, phase-contrast, brightfield
- **Limitations**: Requires paired training data
- **Pre-trained models**: Available via ZeroCostDL4Mic
- **Best for**: Denoising, deblurring when paired data is available
- **Our use case**: Train on our LIVECell HQ/LQ pairs → restore LQ images → apply filters

### 3.2 Noise2Void (N2V)
- **Paper**: Krull et al., CVPR 2019
- **Architecture**: U-Net with blind-spot training
- **Training**: Self-supervised (no clean target needed!)
- **Strengths**: No paired data required; works on any noisy dataset
- **Limitations**: Assumes noise is signal-independent
- **Best for**: Denoising when no clean reference exists
- **Our use case**: Train directly on our noisy LQ images → denoise → apply filters

### 3.3 Noise2Noise (N2N)
- **Paper**: Lehtinen et al., ICML 2018
- **Architecture**: Any (typically U-Net)
- **Training**: Two noisy realizations of same scene
- **Strengths**: No clean data needed
- **Limitations**: Requires two noisy images of same sample
- **Best for**: Time-lapse denoising (consecutive frames)

### 3.4 DnCNN (Denoising CNN)
- **Paper**: Zhang et al., IEEE TIP 2017
- **Architecture**: Deep CNN with residual learning
- **Training**: Supervised (noise-clean pairs)
- **Strengths**: Fast inference, good Gaussian noise removal
- **Limitations**: Trained for specific noise levels
- **Best for**: Fast denoising of Gaussian noise

### 3.5 Restormer
- **Paper**: Zamir et al., NeurIPS 2022
- **Architecture**: Transformer with multi-scale attention
- **Training**: Supervised
- **Strengths**: State-of-the-art on multiple restoration tasks
- **Limitations**: Computationally heavy
- **Best for**: Multi-task restoration (denoise + deblur + derain)

### 3.6 SwinIR
- **Paper**: Liang et al., ICCV 2021
- **Architecture**: Swin Transformer
- **Training**: Supervised
- **Strengths**: Excellent for super-resolution and denoising
- **Limitations**: Needs pre-training on large datasets
- **Best for**: Super-resolution, general restoration

### 3.7 NAFNet (Nonlinear Activation Free Network)
- **Paper**: Chen et al., ECCV 2022
- **Architecture**: Simple U-Net variant without nonlinear activations
- **Training**: Supervised
- **Strengths**: Fast, simple, state-of-the-art results
- **Limitations**: Newer, fewer pre-trained models
- **Best for**: Fast denoising and deblurring

### 3.8 Diffusion Models for Microscopy
- **Paper**: Various (2023-2025)
- **Architecture**: U-Net + diffusion process
- **Training**: Various (supervised, self-supervised)
- **Strengths**: Best quality, handles uncertainty, can generate multiple restorations
- **Limitations**: Very slow (10-100× slower than CNN)
- **Best for**: Research, offline processing, when quality is critical
- **Examples**: CVDM (Conditional Variational Diffusion Model), Physics-Informed DDPM

### 3.9 Physics-Informed Models
- **Approach**: Incorporate optical physics (PSF, diffraction, noise models) into the model
- **Strengths**: Physically consistent results, less data needed, better generalization
- **Limitations**: Requires microscope-specific calibration (PSF, noise model)
- **Best for**: Specific microscope types with known PSF
- **Key models**: DeBCR (wavelet+CNN), PI-DDPM (diffusion+physics), PSF learning (CVPR 2025)
- **See**: [PHYSICS_INFORMED_MODELS.md](PHYSICS_INFORMED_MODELS.md) for detailed review

## 4. Practical Recommendations for Our Pipeline

### 4.1 Enhancement → Filter Pipeline

```
Raw Image → Quality Assessment → Enhancement Model → Bandpass Filter → Analysis
```

**Recommended combinations:**

| Image Quality | Enhancement Model | Then Filter | Expected Gain |
|--------------|-------------------|-------------|---------------|
| **Noisy (σ=50)** | Noise2Void or DnCNN | DoG | +0.05-0.10 IoU |
| **Blurry** | CARE (deblur mode) or NAFNet | Butterworth | +0.03-0.08 IoU |
| **Shaded** | Homomorphic+DL or U-Net | Homomorphic | +0.02-0.05 IoU |
| **Combined mild** | Restormer or SwinIR | DoG | +0.08-0.15 IoU |
| **Combined severe** | Diffusion model | Butterworth (n=4) | +0.05-0.10 IoU |

### 4.2 Training Strategy for Our Data

We have a unique advantage: **paired HQ/LQ images** from our synthetic degradation pipeline.

**Option A: Train CARE on our pairs**
```python
# Train CARE on LIVECell HQ/LQ pairs
# Input: LQ image (noise_50, defocus_4, etc.)
# Target: HQ original
# Result: Model learns to restore specific degradation types
```

**Option B: Train Noise2Void on LQ images only**
```python
# Train N2V on noisy LQ images (no HQ target needed)
# Model learns to denoise without clean reference
# Advantage: Works on any noisy dataset
```

**Option C: Fine-tune pre-trained models**
```python
# Start with pre-trained CARE/SwinIR from ZeroCostDL4Mic
# Fine-tune on our LIVECell pairs
# Faster convergence, better generalization
```

### 4.3 Expected Improvements

Based on literature and our filter analysis:

| Scenario | Filter Only | Enhancement Only | Enhancement + Filter |
|----------|-------------|------------------|---------------------|
| Noise σ=50 | +0.003 IoU | +0.05-0.10 IoU | +0.08-0.15 IoU |
| Defocus σ=4 | +0.005 IoU | +0.03-0.08 IoU | +0.05-0.12 IoU |
| Shading α=0.5 | +0.010 IoU | +0.02-0.05 IoU | +0.04-0.08 IoU |
| Combined mild | +0.028 IoU | +0.08-0.15 IoU | +0.12-0.20 IoU |

**Key insight**: Enhancement models can recover information that filters cannot,
and the combination of enhancement + filtering gives the best results.

## 5. Implementation Resources

### 5.1 ZeroCostDL4Mic (Recommended Starting Point)
- **URL**: https://henriqueslab.org/pages/zerocostdl4mic.html
- **Platform**: Google Colab (free GPU)
- **Models**: CARE, Noise2Void, Noise2Noise, pix2pix, CycleGAN, Deep-STORM
- **No coding required**: Notebook-based interface
- **Pre-trained models**: Available for common microscopy modalities

### 5.2 BioImage Model Zoo
- **URL**: https://bioimage.io
- **Content**: Pre-trained models for microscopy image analysis
- **Models**: Denoising, segmentation, restoration
- **Format**: ONNX, PyTorch, TensorFlow

### 5.3 Specific Model Repositories

| Model | Repository | Language |
|-------|-----------|----------|
| CARE | https://github.com/CSBDeep/CSBDeep | Python/TF |
| Noise2Void | https://github.com/juglab/n2v | Python/TF |
| SwinIR | https://github.com/JingyunLiang/SwinIR | Python/PyTorch |
| Restormer | https://github.com/swz30/Restormer | Python/PyTorch |
| NAFNet | https://github.com/megvii-research/NAFNet | Python/PyTorch |
| Diffusion | https://github.com/CompVis/stable-diffusion | Python/PyTorch |

## 6. Comparison: Filters vs. Enhancement Models

| Aspect | Bandpass Filters | Enhancement Models |
|--------|-----------------|-------------------|
| **Training required** | No | Yes (or pre-trained) |
| **Inference speed** | Very fast (ms) | Fast (CNN) to slow (Diffusion) |
| **Information recovery** | Cannot recover lost info | Can recover beyond noise floor |
| **Generalization** | Universal | Task/dataset specific |
| **Interpretability** | High (frequency domain) | Low (black box) |
| **Computational cost** | Negligible | Low (CNN) to High (Diffusion) |
| **Paired data needed** | No | Yes (supervised) or No (self-supervised) |
| **Best for** | HQ images, real-time | LQ images, offline processing |

## 7. Recommended Approach for Our Project

### Phase 1: Quick Win (No Training)
1. Use **ZeroCostDL4Mic** with pre-trained CARE models
2. Apply to our LQ images
3. Compare filter performance on enhanced vs. raw LQ images

### Phase 2: Custom Training (Best Results)
1. Train **CARE** on our LIVECell HQ/LQ pairs (supervised)
2. Train **Noise2Void** on our LQ images (self-supervised)
3. Compare both approaches
4. Apply best model as preprocessing before filter pipeline

### Phase 3: Advanced (If Resources Allow)
1. Fine-tune **SwinIR** or **Restormer** on our data
2. Compare CNN vs. Transformer architectures
3. Evaluate diffusion models for highest quality

## 8. References

1. Kruse et al. (2017). CARE: Content-Aware Image Restoration. Nature Methods.
2. Krull et al. (2019). Noise2Void — Learning Denoising from Single Noisy Images. CVPR.
3. Lehtinen et al. (2018). Noise2Noise: Learning Image Restoration without Clean Data. ICML.
4. Zhang et al. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP.
5. Liang et al. (2021). SwinIR: Image Restoration Using Swin Transformer. ICCV.
6. Zamir et al. (2022). Restormer: Efficient Transformer for High-Resolution Image Restoration. NeurIPS.
7. Chen et al. (2022). NAFNet: Nonlinear Activation Free Network. ECCV.
8. Weigert et al. (2021). Democratising Deep Learning for Microscopy with ZeroCostDL4Mic. Nature Communications.
9. Pereira et al. (2025). ZeroCostDL4Mic update. Nature Communications.
10. Recent survey (2025). Recent Advancements in Microscopy Image Enhancement using Deep Learning. arXiv:2509.15363.
