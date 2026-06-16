#!/usr/bin/env python3
"""
Phase A: Implement and compare physics-informed enhancement models.
Tests DeBCR (wavelet+CNN) and PI-DDPM (diffusion+physics) on our LIVECell LQ images.
Compares: Raw vs. Enhanced vs. Enhanced+Filter
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from filters import apply_filter
from common import load_image, list_images, load_annotations, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

# ── Install required packages ──────────────────────────────
print("Checking dependencies...")
try:
    import pywt
    print("  PyWavelets: OK")
except ImportError:
    print("  Installing PyWavelets...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pywt"], check=True)
    import pywt

# ── Model 1: DeBCR-inspired (Wavelet + Physics Loss) ───────
class DeBCRInspired:
    """
    Simplified DeBCR implementation.
    Uses wavelet decomposition + physics-informed denoising.
    
    Architecture:
    1. Wavelet decomposition → multi-scale frequency separation
    2. Physics-informed thresholding → noise estimation per band
    3. Soft thresholding → denoise each wavelet coefficient
    4. PSF-aware reconstruction → deconvolve using known/estimated PSF
    
    Physics used:
    - Wavelet theory: separates signal from noise by frequency
    - Poisson noise model: variance proportional to signal
    - Forward consistency: H*f ≈ g (restored image convolved with PSF ≈ input)
    """
    
    def __init__(self, wavelet='db4', levels=3, lambda_physics=0.1):
        self.wavelet = wavelet
        self.levels = levels
        self.lambda_physics = lambda_physics
    
    def estimate_noise(self, coeffs):
        """Physics-informed noise estimation using MAD (Median Absolute Deviation).
        Assumes Gaussian noise in wavelet domain."""
        return np.median(np.abs(coeffs)) / 0.6745
    
    def denoise_coefficients(self, coeffs, noise_sigma):
        """Soft thresholding with physics-informed threshold."""
        threshold = noise_sigma * np.sqrt(2 * np.log(len(coeffs)))
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def forward_consistency_loss(self, restored, original, psf_sigma=1.0):
        """Physics constraint: H*f ≈ g.
        Restored image convolved with PSF should approximate original."""
        from scipy.ndimage import gaussian_filter
        restored_blurred = gaussian_filter(restored, sigma=psf_sigma)
        return np.mean((restored_blurred - original)**2)
    
    def enhance(self, image):
        """Full DeBCR-inspired enhancement pipeline."""
        import pywt
        
        # Step 1: Normalize
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img / 255.0
        
        # Step 2: Wavelet decomposition
        coeffs = pywt.wavedec2(img_norm, self.wavelet, level=self.levels)
        
        # Step 3: Denoise each level
        denoised_coeffs = [coeffs[0]]  # Keep approximation
        for i, detail in enumerate(coeffs[1:], 1):
            # Estimate noise per subband
            noise_est = self.estimate_noise(detail[0])
            # Apply soft thresholding
            denoised_detail = tuple(
                self.denoise_coefficients(d, noise_est) for d in detail
            )
            denoised_coeffs.append(denoised_detail)
        
        # Step 4: Wavelet reconstruction
        restored = pywt.waverec2(denoised_coeffs, self.wavelet)
        
        # Step 5: Physics-informed refinement (forward consistency)
        # Iteratively adjust to satisfy H*f ≈ g
        for iteration in range(5):
            consistency_loss = self.forward_consistency_loss(
                restored, img_norm, psf_sigma=1.0
            )
            # Gradient step toward consistency
            from scipy.ndimage import gaussian_filter
            restored_blurred = gaussian_filter(restored, sigma=1.0)
            residual = img_norm - restored_blurred
            restored = restored + 0.1 * residual
            restored = np.clip(restored, 0, 1)
        
        # Step 6: Denormalize
        result = restored * (img_max - img_min) + img_min
        return np.clip(result, 0, 255).astype(np.uint8)


# ── Model 2: PI-DDPM-inspired (Iterative Physics-Diffusion) ─
class PIDDPMInspired:
    """
    Simplified PI-DDPM (Physics-Informed Diffusion) implementation.
    
    Instead of full diffusion (which requires training), we implement
    the physics-informed iterative refinement that diffusion models perform.
    
    Physics used:
    - Forward model: g = H*f + n
    - Poisson noise model
    - Data fidelity: f should match observed when passed through forward model
    - Prior: f should be smooth, non-negative
    """
    
    def __init__(self, n_steps=50, lr=0.01, lambda_physics=0.5):
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_physics = lambda_physics
    
    def poisson_likelihood_grad(self, restored, observed):
        """Gradient of Poisson negative log-likelihood.
        L = Σ [restored - observed * log(restored + ε)]
        dL/df = 1 - observed / (restored + ε)
        """
        eps = 1e-8
        return 1.0 - observed / (restored + eps)
    
    def forward_model(self, image, psf_sigma=1.0):
        """Apply forward model (convolution with PSF)."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=psf_sigma)
    
    def enhance(self, image):
        """PI-DDPM-inspired iterative enhancement."""
        from scipy.ndimage import gaussian_filter
        
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        
        # Initialize: start with noisy image (or slightly smoothed)
        x = img.copy()
        
        # Estimate PSF sigma from image properties
        psf_sigma = 0.8  # Typical for phase-contrast
        
        for step in range(self.n_steps):
            # Step A: Data fidelity gradient (Poisson)
            forward_x = self.forward_model(x, psf_sigma)
            data_grad = self.poisson_likelihood_grad(forward_x, img)
            
            # Back-project the gradient
            data_grad_back = gaussian_filter(data_grad, sigma=psf_sigma)
            
            # Step B: Prior gradient (smoothness regularization)
            laplacian = (
                gaussian_filter(x, sigma=1.0) - x
            )
            prior_grad = -self.lambda_physics * laplacian
            
            # Step C: Update
            x = x - self.lr * (data_grad_back + prior_grad)
            
            # Step D: Non-negativity constraint
            x = np.maximum(x, 0)
            
            # Step E: Diffusion noise reduction (gradual)
            noise_scale = 1.0 - (step / self.n_steps)
            if noise_scale > 0.01:
                x = x + noise_scale * 0.5 * np.random.randn(*x.shape)
                x = gaussian_filter(x, sigma=0.3)
        
        return np.clip(x, img_min, img_max).astype(np.uint8)


# ── Model 3: Physics-Informed PSF Learning ─────────────────
class PSFLearningPhysics:
    """
    Physics-informed PSF learning and deconvolution.
    
    Models the PSF using Zernike polynomials (wavefront aberrations),
    then performs blind deconvolution.
    
    Physics used:
    - Zernike polynomial PSF parameterization
    - Wavefront aberration theory
    - Spatially varying PSF
    """
    
    def __init__(self, zernike_order=4):
        self.zernike_order = zernike_order
    
    def zernike_psf(self, shape, coeffs, zernike_order=4):
        """Generate PSF from Zernike coefficients."""
        h, w = shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        R = np.clip(R, 0, 1)
        Theta = np.arctan2(Y, X)
        
        # Zernike polynomials (simplified)
        Z = np.ones_like(R)
        if zernike_order >= 1:
            Z = Z + coeffs.get('defocus', 0) * 2 * R**2 - 1
        if zernike_order >= 2:
            Z = Z + coeffs.get('astigmatism_0', 0) * R**2 * np.cos(2 * Theta)
            Z = Z + coeffs.get('astigmatism_45', 0) * R**2 * np.sin(2 * Theta)
        if zernike_order >= 3:
            Z = Z + coeffs.get('coma_x', 0) * (3*R**3 - 2*R) * np.cos(Theta)
            Z = Z + coeffs.get('coma_y', 0) * (3*R**3 - 2*R) * np.sin(Theta)
        if zernike_order >= 4:
            Z = Z + coeffs.get('spherical', 0) * 6*R**4 - 6*R**2 + 1
        
        # Convert wavefield to PSF
        pupil = np.exp(1j * 2 * np.pi * Z)
        pupil[R > 1] = 0
        psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil)))**2
        psf = psf / (psf.sum() + 1e-10)
        return psf
    
    def richardson_lucy(self, image, psf, iterations=20):
        """Richardson-Lucy deconvolution (physics-based)."""
        from scipy.signal import fftconvolve
        
        estimate = image.copy().astype(np.float64) + 1.0
        psf_mirror = psf[::-1, ::-1]
        
        for _ in range(iterations):
            reblurred = fftconvolve(estimate, psf, mode='same')
            reblurred = np.maximum(reblurred, 1e-8)
            relative_blur = image.astype(np.float64) / reblurred
            estimate = estimate * fftconvolve(relative_blur, psf_mirror, mode='same')
            estimate = np.maximum(estimate, 0)
        
        return estimate
    
    def enhance(self, image):
        """PSF learning + Richardson-Lucy deconvolution."""
        from scipy.ndimage import gaussian_filter
        
        img = image.astype(np.float64)
        
        # Estimate PSF from image (simplified: use theoretical phase-contrast PSF)
        # Phase-contrast PSF has central peak + ring
        psf_sigma = 1.0
        psf = self.zernical_psf(
            (21, 21),
            {'defocus': 0.3, 'astigmatism_0': 0.1},
            self.zernike_order
        )
        psf = psf / psf.sum()
        
        # Richardson-Lucy deconvolution
        restored = self.richardson_lucy(img, psf, iterations=15)
        
        # Post-processing: remove artifacts
        restored = gaussian_filter(restored, sigma=0.5)
        
        return np.clip(restored, 0, 255).astype(np.uint8)

    def zernical_psf(self, *args, **kwargs):
        """Alias for zernike_psf."""
        return self.zernike_psf(*args, **kwargs)


# ── Evaluation function ────────────────────────────────────
def segment_iou(image, gt_bboxes):
    if not gt_bboxes:
        return 0.0
    try:
        thresh = threshold_otsu(image)
        pred = image > thresh
    except ValueError:
        return 0.0
    gt = np.zeros_like(image, dtype=bool)
    for bbox in gt_bboxes:
        x, y, w, h = [int(v) for v in bbox]
        gt[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])] = True
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


# ── Main comparison ────────────────────────────────────────
print("=" * 60)
print("Physics-Informed Model Comparison")
print("=" * 60)

# Initialize models
model_debcr = DeBCRInspired(wavelet='db4', levels=3, lambda_physics=0.1)
model_piddpm = PIDDPMInspired(n_steps=30, lr=0.02, lambda_physics=0.3)
model_psf = PSFLearningPhysics(zernike_order=4)

annotations = load_annotations()
cell_lines = ["MCF7", "SHSY5Y", "BV2", "SkBr3"]  # 4 representative lines

# Select test images: 20 per cell line, annotated
test_images = []
for cl in cell_lines:
    cl_imgs = [p for p in list_images(cl) if p.stem in annotations][:20]
    test_images.extend(cl_imgs)

print(f"  Test images: {len(test_images)}")
print(f"  Models: DeBCR-inspired, PI-DDPM-inspired, PSF-Learning")
print(f"  Degradation types: noise_50, combined_mild")

DEGRADATIONS = ["noise_50", "combined_mild"]
BEST_FILTER = {"dog": dict(sigma1=0.05, sigma2=0.20)}

records = []

for i, path in enumerate(test_images):
    if i % 10 == 0:
        print(f"  Processing {i+1}/{len(test_images)}...")

    img_hq = load_image(path)
    ann = annotations.get(path.stem, {})
    bboxes = ann.get("bboxes", [])
    cell_line = path.stem.split("_")[0]

    for deg_name in DEGRADATIONS:
        # Load degraded image
        deg_path = Path("data/mixed_quality") / "synthetic_low" / deg_name / f"{path.stem}.tif"
        if not deg_path.exists():
            continue

        from PIL import Image as PILImage
        img_lq = np.array(PILImage.open(deg_path)).astype(np.float64)

        # 1. Raw (no enhancement)
        iou_raw = segment_iou(img_lq.astype(np.uint8), bboxes)
        records.append({
            "filename": path.stem, "cell_line": cell_line,
            "degradation": deg_name, "method": "raw",
            "iou": iou_raw, "improvement": 0.0
        })

        # 2. DeBCR enhancement
        try:
            img_debcr = model_debcr.enhance(img_lq.astype(np.uint8))
            iou_debcr = segment_iou(img_debcr, bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "degradation": deg_name, "method": "DeBCR",
                "iou": iou_debcr, "improvement": iou_debcr - iou_raw
            })
        except Exception as e:
            print(f"    DeBCR error: {e}")

        # 3. PI-DDPM enhancement
        try:
            img_piddpm = model_piddpm.enhance(img_lq.astype(np.uint8))
            iou_piddpm = segment_iou(img_piddpm, bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "degradation": deg_name, "method": "PI-DDPM",
                "iou": iou_piddpm, "improvement": iou_piddpm - iou_raw
            })
        except Exception as e:
            print(f"    PI-DDPM error: {e}")

        # 4. PSF-Learning enhancement
        try:
            img_psf = model_psf.enhance(img_lq.astype(np.uint8))
            iou_psf = segment_iou(img_psf, bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "degradation": deg_name, "method": "PSF-Learning",
                "iou": iou_psf, "improvement": iou_psf - iou_raw
            })
        except Exception as e:
            print(f"    PSF error: {e}")

        # 5. Best bandpass filter (DoG) on raw
        try:
            img_dog = apply_filter(
                img_lq.astype(np.uint8), "dog",
                sigma1=0.05, sigma2=0.20
            )
            iou_dog = segment_iou(img_dog, bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "degradation": deg_name, "method": "DoG_filter",
                "iou": iou_dog, "improvement": iou_dog - iou_raw
            })
        except Exception as e:
            print(f"    DoG error: {e}")

        # 6. DeBCR + DoG (enhancement + filter)
        try:
            img_debcr_dog = apply_filter(img_debcr, "dog", sigma1=0.05, sigma2=0.20)
            iou_debcr_dog = segment_iou(img_debcr_dog, bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "degradation": deg_name, "method": "DeBCR+DoG",
                "iou": iou_debcr_dog, "improvement": iou_debcr_dog - iou_raw
            })
        except:
            pass

        # 7. Raw (HQ baseline)
        iou_hq = segment_iou(img_hq.astype(np.uint8), bboxes)
        records.append({
            "filename": path.stem, "cell_line": cell_line,
            "degradation": "HQ_reference", "method": "raw_HQ",
            "iou": iou_hq, "improvement": iou_hq - iou_raw
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "physics_model_comparison.csv", index=False)

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

summary = df.groupby(["degradation", "method"]).agg(
    mean_iou=("iou", "mean"),
    mean_improvement=("improvement", "mean"),
    n=("filename", "count")
).round(4)

print("\nAll methods by degradation:")
print(summary.to_string())

# Best method per degradation
print("\n=== BEST METHOD PER DEGRADATION ===")
for deg in DEGRADATIONS:
    sub = df[df["degradation"] == deg]
    best = sub.groupby("method")["iou"].mean().sort_values(ascending=False)
    print(f"\n  {deg}:")
    for method, iou in best.items():
        print(f"    {method:20s}: IoU={iou:.4f}")

# Comparison table
print("\n=== COMPARISON: ENHANCEMENT vs FILTER vs COMBINED ===")
for deg in DEGRADATIONS:
    sub = df[df["degradation"] == deg]
    raw_iou = sub[sub["method"] == "raw"]["iou"].mean()
    best_enhance = sub[sub["method"].isin(["DeBCR", "PI-DDPM", "PSF-Learning"])].groupby("method")["iou"].mean()
    best_enhance_name = best_enhance.idxmax()
    best_enhance_iou = best_enhance.max()
    dog_iou = sub[sub["method"] == "DoG_filter"]["iou"].mean()
    combined_iou = sub[sub["method"] == "DeBCR+DoG"]["iou"].mean()
    hq_iou = sub[sub["method"] == "raw_HQ"]["iou"].mean()

    print(f"\n  {deg}:")
    print(f"    Raw LQ:           {raw_iou:.4f}")
    print(f"    Best enhance:     {best_enhance_iou:.4f} ({best_enhance_name}, Δ={best_enhance_iou-raw_iou:+.4f})")
    print(f"    Best filter:      {dog_iou:.4f} (DoG, Δ={dog_iou-raw_iou:+.4f})")
    print(f"    Enhance+Filter:   {combined_iou:.4f} (DeBCR+DoG, Δ={combined_iou-raw_iou:+.4f})")
    print(f"    HQ reference:     {hq_iou:.4f}")
    gap_raw = hq_iou - raw_iou
    gap_best = hq_iou - max(best_enhance_iou, dog_iou, combined_iou)
    print(f"    Gap closed:       {(1 - gap_best/gap_raw)*100:.1f}%")

# ── Figures ────────────────────────────────────────────────
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Physics-Informed Model Comparison for Microscopy Image Enhancement",
             fontsize=13, fontweight="bold")

# (a) IoU comparison by method
ax = axes[0, 0]
methods = ["raw", "DeBCR", "PI-DDPM", "PSF-Learning", "DoG_filter", "DeBCR+DoG"]
for deg in DEGRADATIONS:
    ious = []
    for method in methods:
        val = df[(df["degradation"] == deg) & (df["method"] == method)]["iou"].mean()
        ious.append(val if not np.isnan(val) else 0)
    ax.plot(methods, ious, marker="o", label=deg, linewidth=1.5)
ax.set_ylabel("Mean IoU")
ax.set_title("(a) IoU by Method")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

# (b) Improvement comparison
ax = axes[0, 1]
for deg in DEGRADATIONS:
    imprs = []
    for method in methods:
        val = df[(df["degradation"] == deg) & (df["method"] == method)]["improvement"].mean()
        imprs.append(val if not np.isnan(val) else 0)
    ax.plot(methods, imprs, marker="s", label=deg, linewidth=1.5)
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("IoU Improvement")
ax.set_title("(b) Improvement over Raw")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

# (c) Gap closure to HQ
ax = axes[1, 0]
gap_data = []
for deg in DEGRADATIONS:
    sub = df[df["degradation"] == deg]
    raw_iou = sub[sub["method"] == "raw"]["iou"].mean()
    hq_iou = sub[sub["method"] == "raw_HQ"]["iou"].mean()
    gap_raw = hq_iou - raw_iou

    for method in methods:
        method_iou = sub[sub["method"] == method]["iou"].mean()
        gap_remaining = hq_iou - method_iou
        gap_closed = (1 - gap_remaining / gap_raw) * 100 if gap_raw > 0 else 0
        gap_data.append({
            "degradation": deg, "method": method, "gap_closed": gap_closed
        })

df_gap = pd.DataFrame(gap_data)
for deg in DEGRADATIONS:
    sub = df_gap[df_gap["degradation"] == deg]
    ax.bar(sub["method"], sub["gap_closed"], alpha=0.7, label=deg)
ax.axhline(100, color="green", linestyle="--", alpha=0.5, label="Perfect restoration")
ax.set_ylabel("Gap Closed (%)")
ax.set_title("(c) HQ Gap Closure")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 110)

# (d) Improvement factor (enhancement vs filter)
ax = axes[1, 1]
for deg in DEGRADATIONS:
    sub = df[df["degradation"] == deg]
    raw_iou = sub[sub["method"] == "raw"]["iou"].mean()

    enhance_methods = ["DeBCR", "PI-DDPM", "PSF-Learning"]
    enhance_best = sub[sub["method"].isin(enhance_methods)].groupby("method")["iou"].mean().max()

    filter_best = sub[sub["method"] == "DoG_filter"]["iou"].mean()
    combined_best = sub[sub["method"] == "DeBCR+DoG"]["iou"].mean()

    ratios = [
        (enhance_best - raw_iou) / (filter_best - raw_iou) if (filter_best - raw_iou) > 0 else 0,
        (combined_best - raw_iou) / (filter_best - raw_iou) if (filter_best - raw_iou) > 0 else 0,
    ]
    ax.bar([f"{deg}\nenhance/filter", f"{deg}\ncombined/filter"], ratios,
           alpha=0.7)
ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="Filter baseline")
ax.set_ylabel("Improvement Ratio")
ax.set_title("(d) Enhancement vs Filter")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "report_physics_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: report_physics_models.png")

print("\nDone.")
