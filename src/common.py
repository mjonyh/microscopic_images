"""
Common utilities for LIVECell FFT analysis.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage


DATA_DIR = Path(__file__).parent.parent / "data"
IMAGE_DIR = DATA_DIR / "livecell_train_val_images" / "livecell_train_val_images"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
ANNOT_DIR = DATA_DIR


# ── Image I/O ──────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load a TIFF image as float64 numpy array."""
    img = Image.open(path)
    return np.array(img, dtype=np.float64)


def list_images(cell_line: str = None) -> list[Path]:
    """List all TIFF images, optionally filtered by cell line."""
    tifs = sorted(IMAGE_DIR.glob("*.tif"))
    if cell_line:
        tifs = [t for t in tifs if t.stem.startswith(cell_line + "_")]
    return tifs


# ── FFT ────────────────────────────────────────────────────

def compute_fft(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D FFT and return shifted power spectrum + frequency axes.
    Returns: (power_spectrum, freq_x, freq_y)
    """
    # Subtract mean to remove DC component
    img = image - image.mean()
    # Windowing to reduce edge artifacts
    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img_windowed = img * window
    # FFT
    ft = np.fft.fft2(img_windowed)
    ft_shifted = np.fft.fftshift(ft)
    power = np.abs(ft_shifted) ** 2
    # Frequency axes (cycles/pixel)
    freq_x = np.fft.fftshift(np.fft.fftfreq(w))
    freq_y = np.fft.fftshift(np.fft.fftfreq(h))
    return power, freq_x, freq_y


def radial_profile(power: np.ndarray, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute azimuthally averaged radial power profile.
    Returns: (freq_bins, power_profile)
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cy, cx)
    # Bin and average
    bins = np.linspace(0, r_max, n_bins + 1)
    freqs = (bins[:-1] + bins[1:]) / 2
    profile = ndimage.mean(power, r, index=np.arange(1, n_bins + 1))
    # Normalize
    profile = profile / profile.max() if profile.max() > 0 else profile
    return freqs, profile


def azimuthal_profile(power: np.ndarray, n_bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radially averaged azimuthal power profile.
    Returns: (angle_bins_deg, power_profile)
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    theta = np.degrees(np.arctan2(y - cy, x - cx)) % 180  # 0-180 due to symmetry
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = min(cy, cx) * 0.8  # Exclude very center and edges
    mask = r < r_max
    bins = np.linspace(0, 180, n_bins + 1)
    angles = (bins[:-1] + bins[1:]) / 2
    profile = ndimage.mean(power[mask], theta[mask], index=np.arange(1, n_bins + 1))
    profile = profile / profile.max() if profile.max() > 0 else profile
    return angles, profile


def spectral_features(power: np.ndarray, freqs_x: np.ndarray, freqs_y: np.ndarray) -> dict:
    """
    Extract scalar features from a 2D power spectrum.
    """
    freq_r, radial = radial_profile(power)
    # Spectral centroid (mean frequency)
    total = radial.sum()
    if total == 0:
        return {"centroid": 0, "bandwidth": 0, "skewness": 0, "kurtosis": 0,
                "total_power": 0, "low_power": 0, "mid_power": 0, "high_power": 0}
    centroid = np.average(freq_r, weights=radial)
    # Bandwidth (std of frequency)
    bandwidth = np.sqrt(np.average((freq_r - centroid) ** 2, weights=radial))
    # Skewness and kurtosis
    if bandwidth > 0:
        skewness = np.average(((freq_r - centroid) / bandwidth) ** 3, weights=radial)
        kurtosis = np.average(((freq_r - centroid) / bandwidth) ** 4, weights=radial)
    else:
        skewness = 0
        kurtosis = 0
    # Frequency band power (low: 0-25%, mid: 25-75%, high: 75-100%)
    n = len(freq_r)
    low = radial[:n // 4].sum() / total
    mid = radial[n // 4:3 * n // 4].sum() / total
    high = radial[3 * n // 4:].sum() / total
    return {
        "centroid": float(centroid),
        "bandwidth": float(bandwidth),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "total_power": float(np.log10(power.sum() + 1)),
        "low_power": float(low),
        "mid_power": float(mid),
        "high_power": float(high),
    }


# ── Filename parsing ───────────────────────────────────────

def get_cell_line(filename: str) -> str:
    """Extract cell line name from filename."""
    return filename.split("_")[0]


def parse_time(filename: str) -> float:
    """
    Extract time in hours from filename.
    Format: ..._02d08h00m_3 → 2*24 + 8 = 56 hours
    """
    m = re.search(r'(\d+)d(\d+)h(\d+)m', filename)
    if m:
        d, h, mi = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return d * 24 + h + mi / 60
    return 0.0


def get_well_id(filename: str) -> str:
    """Extract well identifier (CellLine_Plate_Well)."""
    parts = filename.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else filename


# ── Annotations ────────────────────────────────────────────

def load_annotations() -> dict:
    """
    Load the largest COCO annotation file.
    Returns dict: image_filename -> {cell_count, areas, ...}
    """
    ann_files = sorted(ANNOT_DIR.glob("*percent.json"))
    if not ann_files:
        return {}
    # Pick largest
    best = max(ann_files, key=lambda f: len(json.load(open(f)).get("annotations", [])))
    with open(best) as f:
        coco = json.load(f)
    # Build image_id -> filename map
    img_map = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    # Aggregate annotations per image
    result = defaultdict(lambda: {"cell_count": 0, "areas": [], "bboxes": []})
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        fname = img_map.get(img_id, "")
        # Remove .tif extension for matching
        key = fname.replace(".tif", "")
        result[key]["cell_count"] += 1
        if "area" in ann:
            result[key]["areas"].append(ann["area"])
        if "bbox" in ann:
            result[key]["bboxes"].append(ann["bbox"])
    return dict(result)


# ── Bandpass filter ────────────────────────────────────────

def bandpass_filter(image: np.ndarray, low_cut: float = 0.01, high_cut: float = 0.3) -> np.ndarray:
    """
    Apply frequency-domain bandpass filter.
    low_cut, high_cut: fraction of max frequency (0-0.5)
    """
    h, w = image.shape
    ft = np.fft.fft2(image - image.mean())
    ft_shifted = np.fft.fftshift(ft)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
    mask = (r >= low_cut) & (r <= high_cut)
    ft_filtered = ft_shifted * mask
    result = np.real(np.fft.ifft2(np.fft.ifftshift(ft_filtered)))
    return result + image.mean()
