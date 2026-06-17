#!/usr/bin/env python3
"""
Master runner: executes WS4-7 after WS1-3 complete.
Checks for required input files before running each workstream.
"""
import sys
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Dependencies: which WS need which input files
DEPENDENCIES = {
    "ws4": ["ws1_model_comparison.csv", "ws1_statistics.csv",
            "ws2_bbbc005_quality.csv", "ws3_unet_evaluation.csv"],
    "ws5": ["ws1_model_comparison.csv"],  # Needs models, not just CSVs
    "ws6": ["ws2_bbbc005_quality.csv", "ws2_synthetic_vs_real.csv"],
    "ws7": ["ws1_model_comparison.csv", "ws3_unet_evaluation.csv"],
}

SCRIPTS = {
    "ws4": "src/ws4_manuscript.py",
    "ws5": "src/ws5_gradio.py",
    "ws6": "src/ws6_multimodal.py",
    "ws7": "src/ws7_adaptive.py",  # Already run, but included for completeness
}

def check_dependencies(ws):
    """Check if all required input files exist."""
    missing = []
    for f in DEPENDENCIES.get(ws, []):
        if not (OUTPUT_DIR / f).exists():
            missing.append(f)
    return missing

def run_workstream(ws):
    """Run a workstream script."""
    script = SCRIPTS.get(ws)
    if not script:
        print(f"  No script for {ws}")
        return False

    print(f"\n  Running {ws} ({script})...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=7200
    )

    if result.returncode == 0:
        print(f"  {ws} complete")
        return True
    else:
        print(f"  {ws} failed (exit code {result.returncode})")
        print(f"  STDERR: {result.stderr[-500:]}")
        return False

def main():
    print("=" * 60)
    print("MASTER RUNNER — WS4-7")
    print("=" * 60)

    # Check which WS are ready
    for ws in ["ws4", "ws5", "ws6", "ws7"]:
        missing = check_dependencies(ws)
        if missing:
            print(f"\n  {ws}: WAITING (missing: {missing})")
        else:
            print(f"\n  {ws}: READY")

    # Run ready workstreams
    for ws in ["ws4", "ws5", "ws6"]:
        missing = check_dependencies(ws)
        if missing:
            print(f"\n  Skipping {ws} (missing dependencies)")
            continue

        success = run_workstream(ws)
        if not success:
            print(f"  Warning: {ws} failed, continuing...")

    print("\n" + "=" * 60)
    print("MASTER RUNNER COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
