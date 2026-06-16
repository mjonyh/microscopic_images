"""
Bandpass filter library for FFT-based image analysis.
Implements 12 filter types with consistent interface.

Each filter function takes:
    shape: (height, width) tuple
    params: filter-specific parameters
Returns:
    2D numpy array (the filter mask in frequency domain)
"""
import numpy as np
from scipy.signal import cheby1, cheby2, ellip
from scipy.fft import fftfreq


def _distance_matrix(shape):
    """Create distance-from-center matrix for frequency domain."""
    h, w = shape
    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    U, V = np.meshgrid(u, v)
    return np.sqrt(U**2 + V**2)


def _distance_matrix_shifted(shape):
    """Create centered distance-from-center matrix."""
    h, w = shape
    u = np.fft.fftshift(np.fft.fftfreq(w))
    v = np.fft.fftshift(np.fft.fftfreq(h))
    U, V = np.meshgrid(u, v)
    return np.sqrt(U**2 + V**2)


# ── 1. Ideal Bandpass Filter ────────────────────────────────

def ideal_bandpass(shape, d_low, d_high):
    """
    Ideal (brick-wall) bandpass filter.

    H(u,v) = 1,  if d_low <= D(u,v) <= d_high
             0,  otherwise

    Parameters
    ----------
    shape : tuple (h, w)
    d_low : float — inner cutoff (fraction of Nyquist, 0–0.5)
    d_high : float — outer cutoff (fraction of Nyquist, 0–0.5)

    Returns
    -------
    H : 2D numpy array, shape (h, w)
    """
    D = _distance_matrix_shifted(shape)
    return ((D >= d_low) & (D <= d_high)).astype(np.float64)


# ── 2. Butterworth Bandpass Filter ──────────────────────────

def butterworth_bandpass(shape, d_low, d_high, order=2):
    """
    Butterworth bandpass filter (cascade of LP and HP).

    H_BP = H_LP(d_high) × H_LP(d_low) complement
         = 1/(1+(D/d_high)^(2n)) × 1/(1+(d_low/D)^(2n))

    Parameters
    ----------
    shape : tuple (h, w)
    d_low : float — inner cutoff
    d_high : float — outer cutoff
    order : int — filter order (default 2)

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    D = np.maximum(D, 1e-10)  # avoid division by zero
    H_lp = 1.0 / (1.0 + (D / d_high) ** (2 * order))
    H_hp = 1.0 / (1.0 + (d_low / D) ** (2 * order))
    return H_lp * H_hp


# ── 3. Gaussian Bandpass Filter ─────────────────────────────

def gaussian_bandpass(shape, d_low, d_high):
    """
    Gaussian bandpass filter (cascade of LP and HP Gaussians).

    H_BP = exp(-D²/(2σ_h²)) × (1 - exp(-D²/(2σ_l²)))

    where σ_l = d_low / sqrt(2*ln(2)), σ_h = d_high / sqrt(2*ln(2))

    Parameters
    ----------
    shape : tuple (h, w)
    d_low : float — inner cutoff (–3dB point)
    d_high : float — outer cutoff (–3dB point)

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    sigma_l = d_low / np.sqrt(2 * np.log(2))
    sigma_h = d_high / np.sqrt(2 * np.log(2))
    sigma_l = max(sigma_l, 1e-10)
    sigma_h = max(sigma_h, 1e-10)
    H_lp = np.exp(-D**2 / (2 * sigma_h**2))
    H_hp = 1.0 - np.exp(-D**2 / (2 * sigma_l**2))
    return H_lp * H_hp


# ── 4. Chebyshev Type I Bandpass Filter ─────────────────────

def chebyshev1_bandpass(shape, d_low, d_high, order=2, ripple_db=0.5):
    """
    Chebyshev Type I bandpass filter (equiripple passband).

    Uses 1D Chebyshev polynomial extended radially to 2D.

    Parameters
    ----------
    shape : tuple (h, w)
    d_low, d_high : float — cutoff frequencies
    order : int — filter order
    ripple_db : float — passband ripple in dB

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    D = np.maximum(D, 1e-10)
    epsilon = np.sqrt(10**(ripple_db / 10) - 1)
    # Normalized frequency for Chebyshev polynomial
    D0 = np.sqrt(d_low * d_high)  # geometric center
    W = d_high - d_low  # bandwidth
    # Argument for Chebyshev polynomial
    x = (D**2 - D0**2) / (D * W + 1e-10)
    # Chebyshev polynomial of first kind
    T = np.cosh(order * np.arccosh(np.clip(np.abs(x), 1, None)))
    H = 1.0 / np.sqrt(1 + epsilon**2 * T**2)
    return H


# ── 5. Chebyshev Type II Bandpass Filter ────────────────────

def chebyshev2_bandpass(shape, d_low, d_high, order=2, attenuation_db=40):
    """
    Chebyshev Type II bandpass filter (equiripple stopband).

    Parameters
    ----------
    shape : tuple (h, w)
    d_low, d_high : float — cutoff frequencies
    order : int — filter order
    attenuation_db : float — stopband attenuation in dB

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    D = np.maximum(D, 1e-10)
    epsilon = 1.0 / np.sqrt(10**(attenuation_db / 10) - 1)
    D0 = np.sqrt(d_low * d_high)
    W = d_high - d_low
    x = (D**2 - D0**2) / (D * W + 1e-10)
    T = np.cosh(order * np.arccosh(np.clip(np.abs(x), 1, None)))
    H = 1.0 / np.sqrt(1 + 1.0 / (epsilon**2 * T**2 + 1e-20))
    return H


# ── 6. Elliptic (Cauer) Bandpass Filter ─────────────────────

def elliptic_bandpass(shape, d_low, d_high, order=2, ripple_db=0.5, attenuation_db=40):
    """
    Elliptic (Cauer) bandpass filter (ripple in both bands).

    Approximation using Jacobi elliptic functions.
    For simplicity, uses a product of Chebyshev I and II approximations.

    Parameters
    ----------
    shape : tuple (h, w)
    d_low, d_high : float — cutoff frequencies
    order : int — filter order
    ripple_db : float — passband ripple in dB
    attenuation_db : float — stopband attenuation in dB

    Returns
    -------
    H : 2D numpy array
    """
    H1 = chebyshev1_bandpass(shape, d_low, d_high, order, ripple_db)
    H2 = chebyshev2_bandpass(shape, d_low, d_high, order, attenuation_db)
    return H1 * H2


# ── 7. Laplacian-Bandpass Filter ────────────────────────────

def laplacian_bandpass(shape, d_low, d_high):
    """
    Laplacian operator constrained to a frequency band.

    H(u,v) = -4π²(u² + v²),  if d_low <= D <= d_high
             0,               otherwise

    Parameters
    ----------
    shape : tuple (h, w)
    d_low, d_high : float — frequency band limits

    Returns
    -------
    H : 2D numpy array
    """
    h, w = shape
    u = np.fft.fftshift(np.fft.fftfreq(w))
    v = np.fft.fftshift(np.fft.fftfreq(h))
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    H = -4 * np.pi**2 * (U**2 + V**2)
    H[(D < d_low) | (D > d_high)] = 0
    return H


# ── 8. Homomorphic Filter ───────────────────────────────────

def homomorphic_filter(shape, d0, gamma_l=0.5, gamma_h=2.0, c=1.0):
    """
    Homomorphic filter for illumination correction.

    H(u,v) = (gamma_h - gamma_l) × [1 - exp(-c·D²/D0²)] + gamma_l

    Parameters
    ----------
    shape : tuple (h, w)
    d0 : float — cutoff frequency
    gamma_l : float — low-frequency gain (< 1 to suppress illumination)
    gamma_h : float — high-frequency gain (> 1 to enhance structure)
    c : float — sharpness of transition

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    H = (gamma_h - gamma_l) * (1.0 - np.exp(-c * D**2 / (d0**2 + 1e-10))) + gamma_l
    return H


# ── 9. Gabor Bandpass Filter ────────────────────────────────

def gabor_bandpass(shape, center_freq, sigma_u, sigma_v, theta=0):
    """
    Gabor bandpass filter (orientation-selective).

    H(u,v) = exp(-½((u'-u₀')²/σ_u² + (v'-v₀')²/σ_v²))

    Parameters
    ----------
    shape : tuple (h, w)
    center_freq : float — center frequency magnitude
    sigma_u, sigma_v : float — bandwidth along principal axes
    theta : float — orientation in degrees

    Returns
    -------
    H : 2D numpy array
    """
    h, w = shape
    u = np.fft.fftshift(np.fft.fftfreq(w))
    v = np.fft.fftshift(np.fft.fftfreq(h))
    U, V = np.meshgrid(u, v)
    theta_rad = np.radians(theta)
    u0 = center_freq * np.cos(theta_rad)
    v0 = center_freq * np.sin(theta_rad)
    # Rotated coordinates
    u_rot = (U - u0) * np.cos(theta_rad) + (V - v0) * np.sin(theta_rad)
    v_rot = -(U - u0) * np.sin(theta_rad) + (V - v0) * np.cos(theta_rad)
    sigma_u = max(sigma_u, 1e-10)
    sigma_v = max(sigma_v, 1e-10)
    H = np.exp(-0.5 * (u_rot**2 / sigma_u**2 + v_rot**2 / sigma_v**2))
    return H


# ── 10. Difference of Gaussians (DoG) Bandpass ──────────────

def dog_bandpass(shape, sigma1, sigma2):
    """
    Difference of Gaussians bandpass filter.

    H(u,v) = exp(-D²/(2σ₁²)) - exp(-D²/(2σ₂²)),  σ₁ < σ₂

    Parameters
    ----------
    shape : tuple (h, w)
    sigma1 : float — inner Gaussian width (pixels in freq domain)
    sigma2 : float — outer Gaussian width (pixels in freq domain)

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    sigma1 = max(sigma1, 1e-10)
    sigma2 = max(sigma2, 1e-10)
    H = np.exp(-D**2 / (2 * sigma1**2)) - np.exp(-D**2 / (2 * sigma2**2))
    return H


# ── 11. Trapezoidal Bandpass Filter ─────────────────────────

def trapezoidal_bandpass(shape, d1, d2, d3, d4):
    """
    Trapezoidal bandpass filter (linear ramp transitions).

    Parameters
    ----------
    shape : tuple (h, w)
    d1, d2 : float — lower transition band edges
    d3, d4 : float — upper transition band edges

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    H = np.zeros(shape, dtype=np.float64)
    # Lower ramp
    mask = (D >= d1) & (D < d2)
    H[mask] = (D[mask] - d1) / (d2 - d1 + 1e-10)
    # Passband
    mask = (D >= d2) & (D <= d3)
    H[mask] = 1.0
    # Upper ramp
    mask = (D > d3) & (D <= d4)
    H[mask] = (d4 - D[mask]) / (d4 - d3 + 1e-10)
    return H


# ── 12. Cosine-Tapered (Hann) Bandpass Filter ───────────────

def cosine_tapered_bandpass(shape, d_low, d_high, transition_width):
    """
    Cosine-tapered (Hann window) bandpass filter.

    Parameters
    ----------
    shape : tuple (h, w)
    d_low, d_high : float — passband edges
    transition_width : float — width of cosine taper

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    H = np.zeros(shape, dtype=np.float64)
    t = transition_width
    # Lower taper
    mask = (D >= d_low - t/2) & (D < d_low + t/2)
    H[mask] = 0.5 * (1 + np.cos(np.pi * (d_low - D[mask]) / t))
    # Passband
    mask = (D >= d_low + t/2) & (D <= d_high - t/2)
    H[mask] = 1.0
    # Upper taper
    mask = (D > d_high - t/2) & (D <= d_high + t/2)
    H[mask] = 0.5 * (1 + np.cos(np.pi * (D[mask] - d_high) / t))
    return H


# ── 13. Wiener / Parametric Power Spectrum Filter ───────────

def parametric_bandpass(shape, beta, sigma):
    """
    Parametric power-spectrum bandpass filter.

    H(u,v) = D(u,v)^β × exp(-D²/(2σ²))

    Parameters
    ----------
    shape : tuple (h, w)
    beta : float — power exponent (negative=LP, positive=HP, 0=BP)
    sigma : float — Gaussian width parameter

    Returns
    -------
    H : 2D numpy array
    """
    D = _distance_matrix_shifted(shape)
    D = np.maximum(D, 1e-10)
    sigma = max(sigma, 1e-10)
    H = D**beta * np.exp(-D**2 / (2 * sigma**2))
    return H


# ── Registry ────────────────────────────────────────────────

FILTER_REGISTRY = {
    "ideal":       {"fn": ideal_bandpass,          "params": ["d_low", "d_high"]},
    "butterworth": {"fn": butterworth_bandpass,    "params": ["d_low", "d_high", "order"]},
    "gaussian":    {"fn": gaussian_bandpass,       "params": ["d_low", "d_high"]},
    "chebyshev1":  {"fn": chebyshev1_bandpass,     "params": ["d_low", "d_high", "order", "ripple_db"]},
    "chebyshev2":  {"fn": chebyshev2_bandpass,     "params": ["d_low", "d_high", "order", "attenuation_db"]},
    "elliptic":    {"fn": elliptic_bandpass,       "params": ["d_low", "d_high", "order", "ripple_db", "attenuation_db"]},
    "laplacian":   {"fn": laplacian_bandpass,      "params": ["d_low", "d_high"]},
    "homomorphic": {"fn": homomorphic_filter,      "params": ["d0", "gamma_l", "gamma_h", "c"]},
    "gabor":       {"fn": gabor_bandpass,          "params": ["center_freq", "sigma_u", "sigma_v", "theta"]},
    "dog":         {"fn": dog_bandpass,            "params": ["sigma1", "sigma2"]},
    "trapezoidal": {"fn": trapezoidal_bandpass,    "params": ["d1", "d2", "d3", "d4"]},
    "cosine":      {"fn": cosine_tapered_bandpass, "params": ["d_low", "d_high", "transition_width"]},
    "parametric":  {"fn": parametric_bandpass,     "params": ["beta", "sigma"]},
}


def apply_filter(image, filter_name, **kwargs):
    """
    Apply a named filter to an image.

    Parameters
    ----------
    image : 2D numpy array
    filter_name : str — one of FILTER_REGISTRY keys
    **kwargs : filter-specific parameters

    Returns
    -------
    filtered_image : 2D numpy array (same shape as input)
    """
    from numpy.fft import fft2, ifftshift, ifft2, fftshift

    shape = image.shape
    entry = FILTER_REGISTRY[filter_name]
    H = entry["fn"](shape, **kwargs)

    # Apply in frequency domain
    F = fft2(image - image.mean())
    F_shifted = fftshift(F)
    F_filtered = F_shifted * H
    result = np.real(ifft2(ifftshift(F_filtered)))
    return result + image.mean()
