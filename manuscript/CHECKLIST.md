# MANUSCRIPT CHECKLIST — FFT Microscopy Analysis Paper

## Target Journal: Nature Methods / Medical Image Analysis
## Author: Prof. Dr. Md. Enamul Hoque (SUST, Bangladesh)

---

## SECTION 1: INTRODUCTION
- [x] Motivation: phase-contrast microscopy + FFT underexploited
- [x] LIVECell dataset description (3,727 images, 8 lines, 22 wells)
- [x] 5 research questions (Q1-Q5)
- [x] Scope: 7 workstreams + 3 extensions, 20K+ segmentations

## SECTION 2: MATERIALS AND METHODS
- [x] Dataset (LIVECell + BBBC005)
- [x] Synthetic degradation pipeline (4 types)
- [x] 2D-FFT computation + feature extraction (94-dim)
- [x] Bandpass filter library (12 types)
- [x] Physics-informed models (DeBCR, PI-DDPM, PSF-Learning)
- [x] U-Net segmentation
- [x] Evaluation metrics (IoU, Dice, precision, recall)
- [x] Adaptive filter selection

## SECTION 3: RESULTS

### WS1: Cell Density & Spatial Spectrum
- [x] Dataset overview figure (report_fig1.png) — 311 KB
- [x] Density spectrum figure (report_fig2.png) — 248 KB
- [x] Total FFT power vs cell count: r=0.751, p<0.001
- [x] Radial power spectra per cell line
- [x] Spectral centroid vs cell count: r=0.045

### WS2: Cell Morphology
- [x] Morphology figure (report_fig3.png) — 151 KB
- [x] FFT peak period per cell line (table)
- [x] Spectral centroid period: 3.7±0.3 (Huh7) to 5.2±0.2 (SkBr3)
- [x] Peak period vs sqrt(area): r=-0.016 (weak)

### WS3: Image Quality
- [x] Isotropy ≈ 1.0 for all lines
- [x] Low-freq fraction: 1.6% (SHSY5Y) to 6.6% (BV2)

### WS4: Cell Line Classification
- [x] Classification figure (report_fig4.png) — 49 KB
- [x] SVM-RBF: 81.7% accuracy
- [x] Per-class table (recall, precision, F1)
- [x] Confusion analysis (SKOV3 46.4%, BT474 63.9%)

### WS5: Bandpass Filter Comparison (CORE)
- [x] Filter segmentation summary CSV (96 rows)
- [x] Best filter per cell line table (HQ images)
- [x] Filter performance on LQ images table
- [x] Filter transfer efficiency figure (report_filter_transfer.png) — 140 KB
- [x] Adaptive comparison figure (filter_adaptive_comparison.png) — 51 KB
- [x] IoU heatmap (filter_iou_heatmap.png) — 138 KB
- [x] Filter x cell-line comparison matrix
- [x] Radial profiles figure (filter_radial_profiles.png) — 119 KB
- [x] Improvement boxplot (filter_improvement_boxplot.png) — 139 KB
- [x] Raw vs best (filter_raw_vs_best.png) — 244 KB
- [x] Quality comparison (filter_quality_comparison.png) — 118 KB
- [x] 2D heatmaps (filter_2d_heatmaps.png) — 637 KB
- [x] Application results (filter_application_results.png) — 80 KB
- [x] Best frequency (filter_best_frequency.png) — 38 KB
- [x] Impulse responses (filter_impulse_responses.png) — 77 KB
- [x] Decision tree (report_filter_decision_tree.png) — 171 KB
- [x] Filter performance summary (report_filter_performance.png) — 244 KB

### WS6: Physics-Informed Enhancement
- [x] Physics model comparison CSV (840 rows)
- [x] Statistics CSV (t-test, p-value, Cohen's d)
- [x] Visual comparison summary (visual_comparison_summary.png) — 1.6 MB
- [x] Per-cell-line visual comparisons (BV2, MCF7, SHSY5Y, SkBr3)
- [x] FFT visual comparison (visual_comparison_fft.png) — 2.2 MB
- [x] Physics models report figure (report_physics_models.png) — 215 KB
- [x] Key finding: DeBCR+DoG = +0.057 IoU (2x DoG alone)
- [x] Statistical significance table

### WS7: U-Net Segmentation
- [x] Trained model: unet_fold1.pth (30 MB)
- [x] 5-fold cross-validation results

### WS8: Time-Lapse Dynamics
- [x] Timelapse figure (report_fig6.png) — 162 KB
- [x] Spectral centroid trends
- [x] Mitosis events: 49 total (MCF7=11, SkBr3=11)

### WS9: BBBC005 Blur-Level Analysis
- [x] Blur quality scale (ws2_blur_scale.png) — 40 KB
- [x] BBBC005 quality (ws2_bbbc005_quality.png) — 202 KB
- [x] Synthetic vs real (ws2_synthetic_vs_real.png) — 90 KB

### WS10: Cross-Modality
- [x] Universal guide (ws6_universal_guide.png) — 107 KB
- [x] Cross-modality CSV

### WS11: Adaptive Enhancement
- [x] Adaptive results (ws7_adaptive_results.png) — 145 KB
- [x] Recommendations CSV
- [x] Selector CSV
- [x] Pipeline CSV

### WS12: Composite Summary
- [x] WS4 composite (ws4_composite_summary.png) — 311 KB
- [x] All statistics (ws4_all_statistics.csv)

---

## FIGURES TO GENERATE (TikZ/LaTeX)
- [ ] Fig 1: FFT pipeline schematic (TikZ)
- [ ] Fig 2: Filter taxonomy diagram (TikZ)
- [ ] Fig 3: Image formation model (TikZ)
- [ ] Fig 4: U-Net architecture (TikZ)
- [ ] Fig 5: Adaptive pipeline flow (TikZ)
- [ ] Fig 6: Quality-aware decision tree (TikZ)

## FIGURES TO GENERATE (Matplotlib)
- [ ] Fig 7: Filter x method comparison matrix (publication quality)
- [ ] Fig 8: Physics model comparison bar chart with significance
- [ ] Fig 9: ROC-style classification comparison
- [ ] Fig 10: Enhancement visual comparison grid (4x4)

## TABLES
- [ ] Table 1: FFT feature extraction summary
- [ ] Table 2: Best filter per cell line (HQ)
- [ ] Table 3: Filter performance on LQ images
- [ ] Table 4: Transfer efficiency matrix
- [ ] Table 5: Physics model comparison with statistics
- [ ] Table 6: Classification per-class metrics
- [ ] Table 7: Adaptive vs fixed filtering
- [ ] Table 8: Degradation-specific recommendations

## DATA FILES AVAILABLE
- outputs/filter_segmentation_results.csv (1.9 MB, 96 rows)
- outputs/filter_segmentation_summary.csv (14 KB, 96 rows)
- outputs/filter_adaptive_results.csv (11.6 KB, 152 rows)
- outputs/ws1_model_comparison.csv (50.7 KB, 840 rows)
- outputs/ws1_statistics.csv (1.1 KB, 10 rows)
- outputs/physics_model_comparison.csv (101.6 KB)
- outputs/obj4_classification_report.csv
- outputs/ws7_recommendations.csv
- outputs/ws7_pipeline.csv
- outputs/ws7_selector.csv
- outputs/filter_lq_comparison.csv (720.8 KB)
- outputs/mixed_quality_metrics_subset.csv (431.9 KB)

## EXISTING FIGURES (Ready to include)
- report_fig1.png (311 KB) — Dataset overview
- report_fig2.png (248 KB) — Density spectrum
- report_fig3.png (151 KB) — Morphology
- report_fig4.png (49 KB) — Classification
- report_fig5.png (282 KB) — Segmentation
- report_fig6.png (162 KB) — Timelapse
- filter_iou_heatmap.png (138 KB) — Filter x cell-line heatmap
- filter_adaptive_comparison.png (51 KB) — Adaptive comparison
- filter_radial_profiles.png (119 KB) — Filter radial profiles
- filter_improvement_boxplot.png (139 KB) — Improvement boxplot
- filter_raw_vs_best.png (244 KB) — Raw vs best
- filter_quality_comparison.png (118 KB) — Quality comparison
- filter_2d_heatmaps.png (637 KB) — 2D heatmaps
- filter_application_results.png (80 KB) — Application results
- filter_best_frequency.png (38 KB) — Best frequency
- filter_impulse_responses.png (77 KB) — Impulse responses
- report_filter_decision_tree.png (171 KB) — Decision tree
- report_filter_performance.png (244 KB) — Performance summary
- report_filter_transfer.png (140 KB) — Transfer efficiency
- report_physics_models.png (215 KB) — Physics models
- visual_comparison_summary.png (1.6 MB) — Visual comparison
- visual_comparison_fft.png (2.2 MB) — FFT visual comparison
- visual_comparison_*.png (10 files) — Per-cell-line comparisons
- ws1_comparison.png (216 KB) — WS1 comparison
- ws2_bbbc005_quality.png (202 KB) — BBBC005 quality
- ws2_blur_scale.png (40 KB) — Blur scale
- ws2_synthetic_vs_real.png (90 KB) — Synthetic vs real
- ws4_composite_summary.png (311 KB) — Composite
- ws6_universal_guide.png (107 KB) — Universal guide
- ws7_adaptive_results.png (145 KB) — Adaptive results

## MISSING / NEED TO GENERATE
- [ ] High-quality filter x method comparison matrix (publication-ready)
- [ ] Physics model bar chart with error bars and significance stars
- [ ] Schematic diagrams (TikZ): pipeline, taxonomy, decision tree
- [ ] U-Net architecture diagram (TikZ)
- [ ] Combined figure: enhancement visual comparison grid
- [ ] Supplementary tables (full filter results, all parameters)

## COMPILE & DELIVER
- [ ] Write complete LaTeX manuscript
- [ ] Create figures/ directory with TikZ diagrams
- [ ] Generate publication-quality comparison figures
- [ ] Compile with pdflatex + bibtex
- [ ] Fix all references and cross-references
- [ ] Deliver final PDF
