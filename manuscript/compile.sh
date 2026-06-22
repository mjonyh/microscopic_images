#!/bin/bash
# Compile the manuscript to PDF
# Usage: bash manuscript/compile.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TEX="ms_manuscript.tex"
BASE="ms_manuscript"
OUT_DIR="$SCRIPT_DIR"

# Check for pdflatex
if ! command -v pdflatex &>/dev/null; then
    echo "ERROR: pdflatex not found. Install TeX Live or MiKTeX first."
    echo "Arch: su - && pacman -S texlive-most texlive-science"
    exit 1
fi

# Check for bibtex
if ! command -v bibtex &>/dev/null; then
    echo "ERROR: bibtex not found. Install TeX Live first."
    exit 1
fi

cd "$OUT_DIR"

echo "=== Compiling manuscript ==="
echo "  Source: $OUT_DIR/$TEX"
echo "  Output: $OUT_DIR/${BASE}.pdf"
echo ""

# Pass 1: generate .aux and .bbl
echo "[1/4] pdflatex pass 1..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" 2>&1 | tail -5

# Run bibtex
echo "[2/4] bibtex..."
bibtex "$BASE" 2>&1 | tail -3

# Pass 2: resolve references
echo "[3/4] pdflatex pass 2..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" 2>&1 | tail -5

# Pass 3: final pass for cross-references
echo "[4/4] pdflatex pass 3 (final)..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" 2>&1 | tail -5

# Check result
PDF="$OUT_DIR/${BASE}.pdf"
if [ -f "$PDF" ]; then
    SIZE=$(du -h "$PDF" | cut -f1)
    echo ""
    echo "=== SUCCESS ==="
    echo "  PDF: $PDF ($SIZE)"
    echo "  Pages: $(pdfinfo "$PDF" 2>/dev/null | grep Pages | awk '{print $2}')"
else
    echo "=== FAILED ==="
    echo "  Check $OUT_DIR/${BASE}.log for errors"
    exit 1
fi
