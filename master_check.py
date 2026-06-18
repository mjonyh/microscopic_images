#!/usr/bin/env python3
"""Master checkpoint tracker for all 7 workstreams."""
import json
import sys
import datetime
from pathlib import Path

CHECKPOINT_FILE = Path(__file__).parent.parent / ".master_checkpoint.json"

DEFAULT = {
    "workstream_1": {"name": "Physics-Informed Models", "status": "pending", "steps": {
        "1.1_care": "pending", "1.2_n2v": "pending", "1.3_stats": "pending", "1.4_report": "pending"
    }},
    "workstream_2": {"name": "BBBC005 Analysis", "status": "done", "steps": {
        "2.1_quality": "done", "2.2_filters": "done", "2.3_compare": "done", "2.4_scale": "done"
    }},
    "workstream_3": {"name": "DL Segmentation", "status": "pending", "steps": {
        "3.1_model": "pending", "3.2_train": "pending", "3.3_eval": "pending", "3.4_e2e": "pending"
    }},
    "workstream_4": {"name": "Manuscript", "status": "pending", "steps": {
        "4.1_stats": "pending", "4.2_figure": "pending", "4.3_methods": "pending", "4.4_results": "pending", "4.5_discussion": "pending"
    }},
    "workstream_5": {"name": "Real-Time Demo", "status": "pending", "steps": {
        "5.1_gradio": "pending", "5.2_models": "pending", "5.3_controls": "pending", "5.4_deploy": "pending"
    }},
    "workstream_6": {"name": "Multi-Modal", "status": "pending", "steps": {
        "6.1_fluor": "pending", "6.2_transfer": "pending", "6.3_bright": "pending", "6.4_guide": "pending"
    }},
    "workstream_7": {"name": "Adaptive Enhancement", "status": "done", "steps": {
        "7.1_train": "done", "7.2_selector": "done", "7.3_pipeline": "done", "7.4_recommend": "done"
    }},
}

def load():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return DEFAULT.copy()

def save(state):
    state["last_updated"] = datetime.datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)

def mark(ws, step):
    state = load()
    state[ws]["steps"][step] = "done"
    if all(v == "done" for v in state[ws]["steps"].values()):
        state[ws]["status"] = "done"
    else:
        state[ws]["status"] = "in_progress"
    save(state)
    print(f"  [CHECKPOINT] {ws}/{step} → done")

def show():
    state = load()
    print("\n" + "=" * 65)
    print("MASTER PROGRESS — 7 Workstreams")
    print("=" * 65)
    for ws, info in state.items():
        if ws == "last_updated":
            continue
        status = info["status"]
        done = sum(1 for v in info["steps"].values() if v == "done")
        total = len(info["steps"])
        icon = {"pending": "○", "in_progress": "◐", "done": "●"}.get(status, "○")
        print(f"  {icon} {ws}: {info['name']} ({done}/{total})")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "show":
            show()
        elif cmd == "mark" and len(sys.argv) >= 4:
            mark(sys.argv[2], sys.argv[3])
        elif cmd == "reset":
            save(DEFAULT)
            print("Checkpoints reset")
    else:
        show()
