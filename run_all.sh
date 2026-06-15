#!/bin/bash
# Run all FFT analysis objectives sequentially
# Usage: bash run_all.sh [1|2|3|4|5|6]
# No arg = run all

set -e
cd "$(dirname "$0")"
source .venv/bin/active 2>/dev/null || source .venv/bin/activate

OBJECTIVES="${1:-all}"

run_obj() {
    local num=$1
    local name=$2
    echo ""
    echo "============================================"
    echo "  Objective $num: $name"
    echo "============================================"
    time python "src/obj${num}_"*.py
}

case "$OBJECTIVES" in
    1) run_obj 1 "Cell Density & Spatial Spectrum" ;;
    2) run_obj 2 "Cell Morphology & Size" ;;
    3) run_obj 3 "Image Quality & Artifacts" ;;
    4) run_obj 4 "Cell Line Classification" ;;
    5) run_obj 5 "Segmentation Preprocessing" ;;
    6) run_obj 6 "Time-Lapse Dynamics" ;;
    all)
        run_obj 1 "Cell Density & Spatial Spectrum"
        run_obj 2 "Cell Morphology & Size"
        run_obj 3 "Image Quality & Artifacts"
        run_obj 4 "Cell Line Classification"
        run_obj 5 "Segmentation Preprocessing"
        run_obj 6 "Time-Lapse Dynamics"
        ;;
    *) echo "Usage: bash run_all.sh [1|2|3|4|5|6|all]" ;;
esac

echo ""
echo "All outputs saved to outputs/"
