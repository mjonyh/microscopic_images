#!/bin/bash
# Compile Paper 2 (Medical Image Analysis format)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling Paper 2 (Medical Image Analysis) ==="

echo "[1/4] pdflatex pass 1..."
pdflatex -interaction=nonstopmode -halt-on-error ms_paper2.tex 2>&1 | tail -3

echo "[2/4] bibtex..."
bibtex ms_paper2 2>&1 | tail -2

echo "[3/4] pdflatex pass 2..."
pdflatex -interaction=nonstopmode ms_paper2.tex 2>&1 | tail -3

echo "[4/4] pdflatex pass 3 (final)..."
pdflatex -interaction=nonstopmode ms_paper2.tex 2>&1 | tail -3

PDF="ms_paper2.pdf"
if [ -f "$PDF" ]; then
    SIZE=$(du -h "$PDF" | cut -f1)
    echo ""
    echo "=== SUCCESS ==="
    echo "  PDF: $SCRIPT_DIR/$PDF ($SIZE)"
    echo "  Pages: $(pdfinfo "$PDF" 2>/dev/null | grep Pages | awk '{print $2}')"
else
    echo "=== FAILED ==="
    exit 1
fi
