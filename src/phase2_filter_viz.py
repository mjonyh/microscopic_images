#!/usr/bin/env python3
"""Phase 2: Generate filter response visualizations."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift

sys.path.insert(0, str(Path(__file__).parent))
from filters import FILTER_REGISTRY, apply_filter
from common import load_image, list_images, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9, "figure.dpi": 150,
})

SHAPE = (520, 704)
N_FILTERS = 12

PARAMS = {
    "ideal":       dict(d_low=0.02, d_high=0.30),
    "butterworth": dict(d_low=0.02, d_high=0.30, order=2),
    "gaussian":    dict(d_low=0.02, d_high=0.30),
    "chebyshev1":  dict(d_low=0.02, d_high=0.30, order=2, ripple_db=0.5),
    "chebyshev2":  dict(d_low=0.02, d_high=0.30, order=2, attenuation_db=40),
    "elliptic":    dict(d_low=0.02, d_high=0.30, order=2, ripple_db=0.5, attenuation_db=40),
    "laplacian":   dict(d_low=0.02, d_high=0.30),
    "homomorphic": dict(d0=0.10, gamma_l=0.5, gamma_h=2.0, c=1.0),
    "gabor":       dict(center_freq=0.10, sigma_u=0.05, sigma_v=0.05, theta=0),
    "dog":         dict(sigma1=0.05, sigma2=0.20),
    "trapezoidal": dict(d1=0.01, d2=0.03, d3=0.25, d4=0.35),
    "cosine":      dict(d_low=0.02, d_high=0.30, transition_width=0.04),
}
PARAMS["parametric"] = dict(beta=2, sigma=0.15)

NAMES = list(PARAMS.keys())[:N_FILTERS]
COLORS = plt.cm.tab20(np.linspace(0, 1, N_FILTERS))

# ── Helper: radial profile ──────────────────────────────────
def radial_profile(H, n_bins=100):
    h, w = H.shape
    u = np.fft.fftshift(np.fft.fftfreq(w))
    v = np.fft.fftshift(np.fft.fftfreq(h))
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2).flatten()
    Hf = H.flatten()
    bins = np.linspace(0, 0.5, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (D >= bins[i]) & (D < bins[i+1])
        if mask.sum() > 0:
            profile[i] = Hf[mask].mean()
    return centers, profile

# ── Figure 1: 1D Radial Profiles (all 12 overlay) ───────────
print("Figure 1: 1D radial profiles...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, name in enumerate(NAMES):
    H = FILTER_REGISTRY[name]["fn"](SHAPE, **PARAMS[name])
    freqs, prof = radial_profile(H)
    label = name.replace("_", " ").title()
    ax.plot(freqs, prof, label=label, color=COLORS[i], linewidth=1.2)

ax.axvspan(0.02, 0.30, alpha=0.1, color="green", label="Passband region")
ax.set_xlabel("Spatial Frequency (cycles/pixel)")
ax.set_ylabel("Filter Response H(u,v)")
ax.set_title("Bandpass Filter Comparison — Radial Frequency Response", fontweight="bold")
ax.legend(fontsize=6, ncol=2, loc="upper right")
ax.set_xlim(0, 0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_radial_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_radial_profiles.png")

# ── Figure 2: 2D Heatmaps (3×4 grid) ───────────────────────
print("Figure 2: 2D heatmaps...")

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle("Bandpass Filter Frequency Response — 2D Heatmaps", fontsize=13, fontweight="bold")

for idx, name in enumerate(NAMES):
    ax = axes[idx // 4, idx % 4]
    H = FILTER_REGISTRY[name]["fn"](SHAPE, **PARAMS[name])
    vmax = max(abs(H.max()), abs(H.min()), 0.01)
    im = ax.imshow(H, cmap="viridis", extent=[-0.5, 0.5, -0.5, 0.5],
                   vmin=min(0, H.min()), vmax=vmax)
    ax.set_title(name.replace("_", " ").title(), fontsize=9)
    ax.set_xlabel("u"), ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_2d_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_2d_heatmaps.png")

# ── Figure 3: Impulse Response ──────────────────────────────
print("Figure 3: Impulse responses...")

impulse = np.zeros(SHAPE)
impulse[SHAPE[0]//2, SHAPE[1]//2] = 1.0

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle("Filter Impulse Response (Spatial Domain) — Shows Ringing Artifacts",
             fontsize=13, fontweight="bold")

cy, cx = SHAPE[0]//2, SHAPE[1]//2

for idx, name in enumerate(NAMES):
    ax = axes[idx // 4, idx % 4]
    H = FILTER_REGISTRY[name]["fn"](SHAPE, **PARAMS[name])
    F = fftshift(fft2(impulse))
    response = np.real(ifft2(ifftshift(F * H)))
    crop = response[cy-40:cy+40, cx-40:cx+40]
    vmax = max(abs(crop.min()), abs(crop.max()), 0.001)
    ax.imshow(crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(name.replace("_", " ").title(), fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_impulse_responses.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_impulse_responses.png")

# ── Figure 4: Applied to Sample Image ───────────────────────
print("Figure 4: Applied to sample image...")

sample_path = list_images("MCF7")[len(list_images("MCF7"))//2]
img = load_image(sample_path)

fig, axes = plt.subplots(4, 4, figsize=(14, 12))
fig.suptitle(f"Filters Applied to Phase-Contrast Image ({sample_path.stem[:18]}...)",
             fontsize=13, fontweight="bold")

axes[0, 0].imshow(img, cmap="gray", vmin=30, vmax=220)
axes[0, 0].set_title("Original", fontsize=9)
axes[0, 0].axis("off")

for idx, name in enumerate(NAMES):
    row, col = (idx + 1) // 4, (idx + 1) % 4
    filtered = apply_filter(img, name, **PARAMS[name])
    axes[row, col].imshow(filtered, cmap="gray", vmin=30, vmax=220)
    axes[row, col].set_title(name.replace("_", " ").title(), fontsize=9)
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_applied_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_applied_sample.png")

print("\nPhase 2 complete: 4 visualization figures generated.")
