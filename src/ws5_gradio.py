#!/usr/bin/env python3
"""
Workstream 5: Real-Time Demo (Gradio Interface)
Depends on: WS1 (models), WS2 (BBBC005), WS3 (U-Net)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

print("=" * 60)
print("WS5: Real-Time Demo")
print("=" * 60)

try:
    import gradio as gr
    print("  Gradio available")
except ImportError:
    print("  Installing Gradio...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "gradio"], check=True)
    import gradio as gr

import numpy as np
from PIL import Image
from filters import apply_filter, FILTER_REGISTRY
from phaseA_physics_models import DeBCRInspired, PIDDPMInspired

# Initialize models
model_debcr = DeBCRInspired(wavelet='db4', levels=3)
model_piddpm = PIDDPMInspired(n_steps=20, lr=0.02, lambda_physics=0.3)

def process_image(image, filter_type, d_low, d_high, order, enhance):
    """Process image with selected filter and enhancement."""
    if image is None:
        return None, "No image uploaded"

    img = np.array(image)

    # Apply enhancement if requested
    if enhance == "DeBCR":
        img = model_debcr.enhance(img)
    elif enhance == "PI-DDPM":
        img = model_piddpm.enhance(img)

    # Apply filter
    if filter_type == "DoG":
        result = apply_filter(img, "dog", sigma1=d_low, sigma2=d_high)
    elif filter_type == "Butterworth":
        result = apply_filter(img, "butterworth", d_low=d_low, d_high=d_high, order=int(order))
    elif filter_type == "Gaussian":
        result = apply_filter(img, "gaussian", d_low=d_low, d_high=d_high)
    elif filter_type == "Homomorphic":
        result = apply_filter(img, "homomorphic", d0=d_high, gamma_l=0.5, gamma_h=2.0, c=1.0)
    else:
        result = img

    return result, f"Applied: {enhance} + {filter_type}"

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Microscopy Image"),
        gr.Dropdown(["DoG", "Butterworth", "Gaussian", "Homomorphic"], value="DoG", label="Filter Type"),
        gr.Slider(0.005, 0.1, 0.02, label="Low Cutoff (d_low)"),
        gr.Slider(0.1, 0.5, 0.3, label="High Cutoff (d_high)"),
        gr.Slider(1, 4, 2, label="Filter Order"),
        gr.Dropdown(["None", "DeBCR", "PI-DDPM"], value="None", label="Enhancement"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Processing Info"),
    ],
    title="Microscopy Image Enhancement Demo",
    description="Upload a microscopy image and apply enhancement + filtering",
)

print("  Launching Gradio demo...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
