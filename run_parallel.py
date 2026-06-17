#!/usr/bin/env python3
"""
Parallel execution script for workstreams 4-7 using MPI.
Each workstream runs on a separate MPI rank.
"""
from mpi4py import MPI
import sys
import subprocess
import json
from pathlib import Path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Workstream assignments by rank
WORKSTREAMS = {
    0: "ws4_manuscript",
    1: "ws5_gradio",
    2: "ws6_multimodal",
    3: "ws7_adaptive",
}

def run_workstream(ws_id):
    """Execute a specific workstream."""
    ws_name = WORKSTREAMS.get(ws_id, "unknown")
    print(f"[Rank {rank}] Starting {ws_name}...")

    script_path = Path(__file__).parent / f"src/{ws_name}.py"

    if not script_path.exists():
        print(f"[Rank {rank}] Script not found: {script_path}")
        return {"status": "skipped", "reason": "script_not_found"}

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=3600
        )
        return {
            "status": "done" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    if size < 4:
        if rank == 0:
            print(f"Warning: MPI size={size}, need 4 ranks for parallel execution")
            print("Running all workstreams sequentially on rank 0...")
            for ws_id in range(4):
                result = run_workstream(ws_id)
                print(f"  {WORKSTREAMS[ws_id]}: {result['status']}")
    else:
        ws_id = rank
        result = run_workstream(ws_id)
        print(f"[Rank {rank}] {WORKSTREAMS[ws_id]}: {result['status']}")

    # Gather results
    all_results = comm.gather(result, root=0)

    if rank == 0:
        print("\n" + "=" * 60)
        print("PARALLEL EXECUTION RESULTS")
        print("=" * 60)
        for ws_id, ws_name in WORKSTREAMS.items():
            if ws_id < len(all_results):
                r = all_results[ws_id]
                print(f"  {ws_name}: {r.get('status', 'unknown')}")
        print("=" * 60)
