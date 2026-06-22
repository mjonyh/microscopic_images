#!/usr/bin/env python3
"""
Workstream 5: Real-Time Demo (Gradio Interface)
Steps:
  5.1 — Gradio interface setup
  5.2 — Model serving (UNet, enhancement models, filters)
  5.3 — Interactive controls (sliders, dropdowns, checkboxes)
  5.4 — Deployment (requirements.txt, usage docs)

Usage:
  python src/ws5_gradio.py          # launch on localhost:7860
  python src/ws5_gradio.py --share # create public Gradio link
  python src/ws5_gradio.py --prep  # only prepare models, don't launch
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

plt_rcparams = {
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
}


def load_models(device="cpu"):
    """Load all enhancement and segmentation models."""
    import torch
    import torch.nn as nn
    from phaseA_physics_models import DeBCRInspired, PIDDPMInspired
    from filters import apply_filter

    models = {}

    # U-Net
    class UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
            super().__init__()
            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            self.pool = nn.MaxPool2d(2, 2)
            for feature in features:
                self.encoder.append(self._block(in_channels, feature))
                in_channels = feature
            self.bottleneck = self._block(features[-1], features[-1] * 2)
            for feature in reversed(features):
                self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
                self.decoder.append(self._block(feature * 2, feature))
            self.final_conv = nn.Conv2d(features[0], out_channels, 1)

        def _block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            skip = []
            for down in self.encoder:
                x = down(x); skip.append(x); x = self.pool(x)
            x = self.bottleneck(x); skip = skip[::-1]
            for idx in range(0, len(self.decoder), 2):
                x = self.decoder[idx](x)
                s = skip[idx // 2]
                if x.shape != s.shape:
                    x = nn.functional.interpolate(x, size=s.shape[2:])
                x = torch.cat([s, x], dim=1)
                x = self.decoder[idx + 1](x)
            return torch.sigmoid(self.final_conv(x))

    use_cuda = torch.cuda.is_available() and device != "cpu"
    dev = torch.device("cuda" if use_cuda else "cpu")

    unet = UNet().to(dev)
    pth = OUTPUT_DIR / "unet_fold1.pth"
    if pth.exists():
        try:
            unet.load_state_dict(torch.load(pth, map_location=dev))
            print(f"  Loaded UNet from {pth}")
        except Exception as e:
            print(f"  UNet load failed ({e}), using random weights")
    else:
        print("  No UNet checkpoint found, using random weights")
    unet.eval()

    models["unet"] = unet
    models["device"] = dev
    models["debcr"] = DeBCRInspired(wavelet="db4", levels=3)
    models["piddpm"] = PIDDPMInspired(n_steps=20, lr=0.02, lambda_physics=0.3)
    models["apply_filter"] = apply_filter
    models["use_cuda"] = use_cuda

    print(f"  Models loaded. Device: {dev}")
    return models


def process_pipeline(image, filter_type, d_low, d_high, order,
                     enhancement, show_fft, use_unet, models):
    """
    Full processing pipeline:
      1. Enhancement (optional)
      2. Bandpass filtering (optional)
      3. Segmentation (Otsu or U-Net)
      4. FFT visualization (optional)
    Returns: (processed_image, segmentation_mask, info_text, fft_image_or_None)
    """
    import torch
    from skimage.filters import threshold_otsu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image is None:
        return None, None, "No image uploaded", None

    # Convert to numpy
    if isinstance(image, np.ndarray):
        img = image.copy().astype(np.float64)
    else:
        img = np.array(image, dtype=np.float64)

    # Grayscale if RGB
    if img.ndim == 3:
        img = img.mean(axis=2)

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    info_parts = [f"Input: {img_uint8.shape[1]}x{img_uint8.shape[0]}"]

    # Step 1: Enhancement
    if enhancement == "DeBCR":
        enhanced = models["debcr"].enhance(img_uint8)
        info_parts.append("Enhancement: DeBCR")
    elif enhancement == "PI-DDPM":
        enhanced = models["piddpm"].enhance(img_uint8)
        info_parts.append("Enhancement: PI-DDPM")
    else:
        enhanced = img_uint8.copy()
        info_parts.append("Enhancement: None")

    # Step 2: Filtering
    if filter_type and filter_type != "None":
        try:
            if filter_type == "DoG":
                filtered = models["apply_filter"](enhanced, "dog", sigma1=d_low, sigma2=d_high)
            elif filter_type == "Butterworth":
                filtered = models["apply_filter"](enhanced, "butterworth", d_low=max(d_low, 0.001), d_high=d_high, order=int(order))
            elif filter_type == "Gaussian":
                filtered = models["apply_filter"](enhanced, "gaussian", d_low=d_low, d_high=d_high)
            elif filter_type == "Homomorphic":
                filtered = models["apply_filter"](enhanced, "homomorphic", d0=d_high, gamma_l=0.5, gamma_h=2.0, c=1.0)
            else:
                filtered = enhanced
        except Exception as e:
            filtered = enhanced
            info_parts.append(f"Filter error: {e}")
        else:
            info_parts.append(f"Filter: {filter_type} (low={d_low:.3f}, high={d_high:.3f}, order={order:.0f})")
    else:
        filtered = enhanced
        info_parts.append("Filter: None")

    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    # Step 3: Segmentation
    if use_unet:
        try:
            dev = models["device"]
            unet = models["unet"]
            tensor = torch.FloatTensor(filtered.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = unet(tensor)
            seg_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            info_parts.append(f"Segmentation: U-Net ({'GPU' if models['use_cuda'] else 'CPU'})")
        except Exception as e:
            # Fallback to Otsu
            try:
                thresh = threshold_otsu(filtered)
                seg_mask = ((filtered > thresh).astype(np.uint8)) * 255
                info_parts.append(f"Segmentation: Otsu (U-Net failed: {e})")
            except:
                seg_mask = np.zeros_like(filtered)
                info_parts.append("Segmentation: failed")
    else:
        try:
            thresh = threshold_otsu(filtered)
            seg_mask = ((filtered > thresh).astype(np.uint8)) * 255
            info_parts.append("Segmentation: Otsu")
        except:
            seg_mask = np.zeros_like(filtered)
            info_parts.append("Segmentation: failed")

    # Step 4: FFT (optional)
    fft_img = None
    if show_fft:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax1.imshow(filtered, cmap="gray")
            ax1.set_title("Filtered Image")
            ax1.axis("off")

            f = np.fft.fft2(filtered.astype(np.float64) - filtered.mean())
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            ax2.imshow(magnitude, cmap="viridis")
            ax2.set_title("FFT Log-Power Spectrum")
            ax2.axis("off")

            plt.tight_layout()
            fig.canvas.draw()
            fft_img = np.frombuffer(fig.canvas.tobytes_rgb(), dtype=np.uint8)
            fft_img = fft_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            info_parts.append("FFT: shown")
        except Exception as e:
            info_parts.append(f"FFT error: {e}")

    info = " | ".join(info_parts)
    return filtered, seg_mask, info, fft_img


def build_gradio(models):
    """Build and return the Gradio interface."""
    import gradio as gr

    # Get filter list
    from filters import FILTER_REGISTRY
    filter_choices = ["None"] + sorted(FILTER_REGISTRY.keys())

    demo = gr.Blocks(title="Microscopy Image Enhancement Demo")

    with demo:
        gr.Markdown("# Microscopy Image Enhancement & Segmentation Demo")
        gr.Markdown(
            "Upload a microscopy image and apply enhancement + filtering + segmentation. "
            "Powered by physics-informed models (DeBCR, PI-DDPM), 12 bandpass filters, and U-Net segmentation."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                input_image = gr.Image(type="numpy", label="Upload Microscopy Image")

                with gr.Row():
                    enhancement = gr.Dropdown(
                        ["None", "DeBCR", "PI-DDPM"], value="None",
                        label="Enhancement Model"
                    )
                    filter_type = gr.Dropdown(
                        filter_choices, value="DoG",
                        label="Bandpass Filter"
                    )

                d_low = gr.Slider(0.001, 0.1, value=0.02, step=0.001,
                                  label="Low Cutoff Frequency (d_low)")
                d_high = gr.Slider(0.05, 0.5, value=0.3, step=0.01,
                                   label="High Cutoff Frequency (d_high)")
                order = gr.Slider(1, 8, value=2, step=1,
                                  label="Filter Order (Butterworth)")

                with gr.Row():
                    use_unet = gr.Checkbox(value=False, label="Use U-Net (vs Otsu)")
                    show_fft = gr.Checkbox(value=False, label="Show FFT Spectrum")

                run_btn = gr.Button("Process", variant="primary")

            with gr.Column(scale=1):
                # Outputs
                output_image = gr.Image(type="numpy", label="Processed Image")
                output_mask = gr.Image(type="numpy", label="Segmentation Mask")
                output_fft = gr.Image(type="numpy", label="FFT Spectrum (optional)")
                info_text = gr.Textbox(label="Processing Info", interactive=False)

                status = gr.Textbox(label="Status", interactive=False)

        def run_with_status(image, filt, dl, dh, ord_, enh, fft, unet):
            try:
                out_img, out_mask, info, fft_img = process_pipeline(
                    image, filt, dl, dh, ord_, enh, fft, unet, models
                )
                return out_img, out_mask, fft_img, info, "Done"
            except Exception as e:
                return None, None, None, str(e), f"Error: {e}"

        run_btn.click(
            fn=run_with_status,
            inputs=[input_image, filter_type, d_low, d_high, order,
                    enhancement, show_fft, use_unet],
            outputs=[output_image, output_mask, output_fft, info_text, status],
        )

    return demo


def write_requirements():
    """Write requirements.txt for deployment."""
    reqs = """# WS5 Demo Requirements
gradio>=4.0
torch>=2.0
numpy>=1.24
Pillow>=10.0
scikit-image>=0.21
scipy>=1.10
matplotlib>=3.7
PyWavelets>=1.4
"""
    path = Path("requirements_demo.txt")
    path.write_text(reqs)
    print(f"  Written: {path}")
    return path


def write_usage_docs():
    """Write deployment documentation."""
    docs = """# WS5: Real-Time Demo — Deployment Guide

## Quick Start

```bash
source .venv/bin/activate
python src/ws5_gradio.py
```

Open browser at http://localhost:7860

## With GPU Acceleration

The demo automatically uses CUDA if available (for U-Net inference).
Ensure PyTorch CUDA is installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Public Share Link

```bash
python src/ws5_gradio.py --share
```

## Gradio on HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Upload: app.py (this script), requirements_demo.txt, outputs/unet_fold1.pth
4. Rename requirements_demo.txt to requirements.txt
5. Space auto-degradio built-in runner

## API Access

The demo also exposes a Gradio API:
```python
from gradio_client import Client
client = Client("http://localhost:7860/")
result = client.predict(
    image, "DoG", 0.02, 0.3, 2, "None", False, False,
    api_name="/run_with_status"
)
```

## Controls

| Control | Description |
|---------|-------------|
| Enhancement | DeBCR (wavelet+physics), PI-DDPM (iterative diffusion), or None |
| Filter | 12 bandpass types: DoG, Butterworth, Gaussian, Homomorphic, etc. |
| d_low / d_high | Frequency cutoffs (0.001–0.5) |
| Filter Order | Butterworth order (1–8) |
| U-Net | Toggle deep learning segmentation vs Otsu thresholding |
| FFT | Show power spectrum visualization |
"""
    path = Path("DEPLOYMENT_WS5.md")
    path.write_text(docs)
    print(f"  Written: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="WS5: Real-Time Demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--prep", action="store_true", help="Only prepare models and docs, don't launch")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for U-Net inference")
    args = parser.parse_args()

    print("=" * 60)
    print("WS5: Real-Time Demo")
    print("=" * 60)

    # Install gradio if needed
    try:
        import gradio as gr
        print(f"  Gradio {gr.__version__} available")
    except ImportError:
        print("  Installing Gradio...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gradio"], check=True)
        import gradio as gr
        print(f"  Gradio {gr.__version__} installed")

    # Write deployment artifacts
    print("\n5.4: Writing deployment artifacts...")
    write_requirements()
    write_usage_docs()

    # Load models
    print("\n5.2: Loading models...")
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        device = "cpu"
    models = load_models(device=device)

    if args.prep:
        print("\n[Prep complete — models loaded, artifacts written. Use without --prep to launch.]")
        return

    # Build and launch
    print("\n5.1/5.3: Building Gradio interface...")
    demo = build_gradio(models)

    print(f"\n  Launching on port {args.port}...")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
