# Research Article Drafting & Submission Plan

## Project: FFT-Based Analysis and Physics-Informed Enhancement of Phase-Contrast Microscopy

**Author:** Prof. Dr. Md. Enamul Hoque (SUST, Bangladesh)
**Repository:** https://github.com/mjonyh/microscopic_images
**Current manuscript:** `manuscript/ms_manuscript.tex` (979 lines, ~3,546 words body, 19 display items)

---

## 1. TARGET JOURNAL ANALYSIS

### 1.1 Primary Target: Nature Methods

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Body text | ~3,546 words | ~546 over 3,000 limit |
| Display items | 19 (figures + tables) | 13 over 6 limit |
| Abstract | ~250 words | Must be ≤150 words, unstructured |
| Methods placement | In main text | Must move to Online Methods (after references) |
| References | ~10 | Must expand to ~50 |
| Format | LaTeX article class | Must use Nature template |
| Key requirement | Single high-consequence claim | Current manuscript has 10 workstreams — too broad |

**Verdict:** Major restructuring needed. The current manuscript tries to do too much for Nature Methods' tight format.

### 1.2 Secondary Target: Medical Image Analysis (Elsevier)

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Body text | ~3,546 words | Within 4,000 limit |
| Display items | 19 | Within 6 combined limit for some article types |
| Abstract | ~250 words | Must be 150-250 words, structured |
| Format | LaTeX article class | Must use Elsevier template |
| Key requirement | Methodological novelty for medical imaging | FFT analysis is established; physics-informed enhancement is the novel angle |

**Verdict:** Better fit for the full scope, but display items still exceed limits.

### 1.3 Tertiary Target: IEEE Transactions on Medical Imaging

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Page limit | 20 pages | Must fit 10 pages (initial submission) |
| Format | LaTeX article class | Must use IEEE template |
| Key requirement | Methodological innovation in medical imaging | Need stronger ML/technical contribution |

**Verdict:** Too long; requires significant compression.

### 1.4 Recommended Strategy: Two-Paper Approach

**Paper 1 (Methods/Tools):** "SPINDEEP: A Spectral Pipeline for Label-Free Cell Line Classification and Quality-Aware Enhancement of Phase-Contrast Microscopy"
- Target: Nature Methods or Bioinformatics
- Focus: The computational framework + physics-informed enhancement as the core novelty
- Scope: 3,000 words, 6 figures, ~50 references

**Paper 2 (Application/Analysis):** "Systematic Evaluation of Bandpass Filters for FFT-Based Segmentation of Phase-Contrast Microscopy Across 8 Cell Lines and 13 Degradation Types"
- Target: Medical Image Analysis or IEEE TMI
- Focus: The filter comparison + quality-aware selection as the core novelty
- Scope: 4,000 words, 6 figures

---

## 2. PAPER 1 RESTRUCTURE PLAN (Nature Methods Format)

### 2.1 Title Options

1. "Physics-informed spectral enhancement of phase-contrast microscopy for label-free cell segmentation"
2. "SPINDEEP: Spectral pipeline for quality-aware enhancement and classification of phase-contrast microscopy"
3. "Fourier-domain quality-aware enhancement of phase-contrast microscopy images using physics-informed deep learning"

### 2.2 Structure (Nature Methods Article Format)

```
Title (≤20 words)
Author + affiliation
Corresponding email

Abstract (≤150 words, unstructured)
  - One sentence: problem
  - One sentence: what we did
  - 2-3 sentences: key results
  - One sentence: significance/availability

Introduction (≤500 words)
  - Phase-contrast microscopy context (2 sentences)
  - FFT analysis gap (2 sentences)
  - Physics-informed enhancement gap (2 sentences)
  - What this paper does (3 sentences)
  - Key finding preview (2 sentences)

Results (≤1,500 words)
  - Figure 1: Framework overview (TikZ schematic)
  - Figure 2: FFT features → classification (81.7% accuracy)
  - Figure 3: Quality-dependent filter performance (10-100x gap)
  - Figure 4: Physics-informed enhancement closes the gap (DeBCR+DoG 2x)
  - Figure 5: Adaptive pipeline on real LQ images
  - Figure 6: Cross-dataset validation (BBBC005)

Discussion (≤800 words)
  - Summary of key findings (3 sentences)
  - Comparison with related work (5 sentences)
  - Limitations (3 sentences)
  - Future directions (2 sentences)

Conclusion (≤200 words)
  - 3-5 bullet points of key contributions

Online Methods (≤3,000 words, after references)
  - Dataset details
  - FFT computation
  - Filter library
  - Physics-informed models
  - U-Net architecture
  - Evaluation metrics
  - Statistical tests

References (~50)
Data Availability
Code Availability
Acknowledgments
Author Contributions
Competing Interests
```

### 2.3 Figure Plan (6 figures max)

| Figure | Content | Type | Current file |
|--------|---------|------|-------------|
| 1 | Computational framework overview | TikZ schematic | `figures/tikz_pipeline.tex` |
| 2 | FFT feature extraction + classification results | Multi-panel (a-d) | `outputs/report_fig2.png` + `outputs/report_fig4.png` |
| 3 | Filter performance: HQ vs LQ (quality gap) | Multi-panel (a-c) | `outputs/filter_quality_comparison.png` + `outputs/ws2_blur_scale.png` |
| 4 | Physics-informed enhancement results | Multi-panel (a-d) | `outputs/fig2_physics_models.pdf` + `outputs/fig4_transfer_efficiency.pdf` |
| 5 | Adaptive pipeline: quality-aware selection | Multi-panel (a-c) | `outputs/fig5_adaptive_vs_fixed.pdf` + `outputs/ws7_adaptive_results.png` |
| 6 | Cross-dataset validation (BBBC005) | Multi-panel (a-b) | `outputs/ws2_bbbc005_quality.png` + `outputs/ws2_synthetic_vs_real.png` |

### 2.4 What Gets Cut from Current Manuscript

- Time-lapse dynamics (WS8) → move to supplementary or Paper 2
- Cross-modality transfer (WS10) → move to supplementary
- Detailed filter taxonomy (WS5 full) → move to supplementary
- U-Net training details → move to Online Methods only
- BBBC005 full analysis → condensed to 1 figure panel

### 2.5 What Gets Added

- Statistical comparison table (effect sizes, confidence intervals)
- Ablation study: each component's contribution
- Runtime benchmarking table
- Comparison with existing methods (CARE, N2V, etc.)

---

## 3. PAPER 2 RESTRUCTURE PLAN (Medical Image Analysis Format)

### 3.1 Title Options

1. "Systematic evaluation of twelve bandpass filters for FFT-based segmentation of phase-contrast microscopy across eight cell lines and thirteen degradation types"
2. "Quality-aware bandpass filter selection for Fourier-domain preprocessing of phase-contrast cell images"
3. "No universal best filter: cell-line-adaptive bandpass selection for phase-contrast microscopy segmentation"

### 3.2 Structure (MIA Format)

```
Title
Authors + affiliations

Abstract (150-250 words, structured)
  - Background
  - Methods
  - Results
  - Conclusions

Keywords (4-6)

1. Introduction (600 words)
   - Phase-contrast segmentation challenge
   - Bandpass filtering as preprocessing
   - Gap: no systematic evaluation across cell lines/qualities
   - Our contribution: 20K+ evaluations, adaptive selection

2. Related Work (400 words)
   - FFT in microscopy
   - Bandpass filtering for segmentation
   - Cell line classification from texture

3. Methods (800 words)
   - Dataset (LIVECell + BBBC005 + synthetic)
   - 12 filter types (table)
   - Evaluation protocol
   - Adaptive selection algorithm

4. Results (1,200 words)
   - Figure 1: Filter taxonomy + radial profiles
   - Figure 2: Filter x cell-line heatmap
   - Figure 3: HQ vs LQ performance
   - Figure 4: Adaptive vs fixed comparison
   - Figure 5: Degradation-specific recommendations
   - Table 1: Best filter per cell line (HQ)
   - Table 2: Best filter per degradation type (LQ)

5. Discussion (600 words)
   - Key findings
   - Practical recommendations
   - Limitations

6. Conclusion (200 words)

References (~40)
```

### 3.3 Figure Plan (6 figures max)

| Figure | Content | Type | Current file |
|--------|---------|------|-------------|
| 1 | Filter taxonomy + radial profiles | Multi-panel | `outputs/fig1_filter_matrix.pdf` + `outputs/filter_radial_profiles.png` |
| 2 | Filter x cell-line IoU heatmap | Heatmap | `outputs/filter_iou_heatmap.png` |
| 3 | HQ vs LQ performance degradation | Grouped bar | `outputs/filter_quality_comparison.png` |
| 4 | Adaptive vs fixed filtering | Multi-panel | `outputs/fig5_adaptive_vs_fixed.pdf` + `outputs/filter_adaptive_comparison.png` |
| 5 | Degradation-specific results | Multi-panel | `outputs/filter_application_results.png` |
| 6 | BBBC005 blur scale validation | Multi-panel | `outputs/ws2_blur_scale.png` + `outputs/ws2_synthetic_vs_real.png` |

---

## 4. SUPPLEMENTARY MATERIAL PLAN

### 4.1 Supplementary Figures (unlimited for most journals)

| Fig | Content | Current file |
|-----|---------|-------------|
| S1 | Dataset overview (all 8 cell lines) | `outputs/report_fig1.png` |
| S2 | Cell density correlation details | `outputs/report_fig2.png` |
| S3 | Morphology analysis (FFT peak) | `outputs/report_fig3.png` |
| S4 | Classification per-class metrics | `outputs/report_fig4.png` |
| S5 | Time-lapse dynamics | `outputs/report_fig6.png` |
| S6 | Filter impulse responses | `outputs/filter_impulse_responses.png` |
| S7 | Filter 2D heatmaps | `outputs/filter_2d_heatmaps.png` |
| S8 | Visual comparison grid | `outputs/visual_comparison_summary.png` |
| S9 | Cross-modality transfer | `outputs/ws6_universal_guide.png` |
| S10 | U-Net training curves | Generate new |
| S11 | Ablation study results | Generate new |
| S12 | Runtime benchmarking | Generate new |

### 4.2 Supplementary Tables

| Table | Content | Data source |
|-------|---------|------------|
| S1 | Full filter parameters (12 types × configs) | `outputs/filter_segmentation_summary.csv` |
| S2 | Per-cell-line classification metrics | `outputs/obj4_classification_report.csv` |
| S3 | All statistical test results | `outputs/ws1_statistics.csv` |
| S4 | Degradation-specific recommendations | `outputs/ws7_recommendations.csv` |
| S5 | BBBC005 blur level analysis | `outputs/ws2_blur_scale.png` data |

### 4.3 Supplementary Methods (for Nature Methods Online Methods)

- Detailed FFT computation parameters
- Training procedures for all models
- Hyperparameter search ranges
- Hardware specifications
- Software versions

---

## 5. IMPLEMENTATION TIMELINE

### Phase 1: Core Restructuring (Week 1-2)

| Task | Days | Deliverable |
|------|------|-------------|
| Decide target journal (Paper 1) | 1 | Decision document |
| Create new LaTeX template | 1 | `manuscript_paper1/ms_paper1.tex` |
| Rewrite Abstract (150 words) | 1 | New abstract |
| Rewrite Introduction (500 words) | 2 | New intro section |
| Select and arrange 6 main figures | 2 | Figure selection document |
| Write Results (1,500 words) | 3 | New results section |
| Write Discussion (800 words) | 2 | New discussion section |
| Move methods to Online Methods | 2 | New methods section |
| Expand references to ~50 | 2 | Updated .bib file |
| Internal review + revision | 2 | Revised draft |

### Phase 2: Supplementary & Validation (Week 3)

| Task | Days | Deliverable |
|------|------|-------------|
| Generate supplementary figures | 3 | 12 supplementary figures |
| Generate supplementary tables | 2 | 5 supplementary tables |
| Write supplementary methods | 2 | Detailed methods |
| Ablation study | 2 | New figure + analysis |
| Runtime benchmarking | 1 | Benchmark table |
| Cross-check all references | 1 | Verified bibliography |

### Phase 3: Paper 2 Preparation (Week 4-5)

| Task | Days | Deliverable |
|------|------|-------------|
| Create MIA template | 1 | `manuscript_paper2/ms_paper2.tex` |
| Write Paper 2 manuscript | 5 | Full draft |
| Select 6 figures for Paper 2 | 1 | Figure selection |
| Internal review | 2 | Revised draft |

### Phase 4: Submission Prep (Week 6)

| Task | Days | Deliverable |
|------|------|-------------|
| Cover letter for Paper 1 | 1 | Cover letter |
| Cover letter for Paper 2 | 1 | Cover letter |
| Suggest reviewers | 1 | Reviewer list |
| Final proofreading | 2 | Polished manuscripts |
| Submit Paper 1 | 1 | Submission confirmation |
| Submit Paper 2 | 1 | Submission confirmation |

---

## 6. KEY NARRATIVE CHANGES

### 6.1 Current Problem
The current manuscript reads like a technical report with 10 workstreams. It lacks a single, clear narrative arc. High-impact journals need ONE story.

### 6.2 Paper 1 Narrative Arc

**Hook:** Phase-contrast microscopy is ubiquitous but quantitative analysis of its frequency content is underexploited, especially for low-quality images common in automated screening.

**Problem:** Bandpass filtering helps segmentation of high-quality images, but improvements on low-quality images are 10-100x smaller. No existing method bridges this gap.

**Solution:** We present a physics-informed spectral enhancement pipeline that combines FFT-based quality assessment with deep learning enhancement (DeBCR, PI-DDPM) and adaptive bandpass filtering.

**Key result:** The DeBCR+DoG combination achieves 2x improvement over filter-only on degraded images, and the full pipeline achieves 81.7% cell line classification from spectral features alone.

**Significance:** This is the first systematic framework for quality-aware enhancement of phase-contrast microscopy, with implications for high-throughput screening and automated cell analysis.

### 6.3 Paper 2 Narrative Arc

**Hook:** Bandpass filtering is the standard preprocessing step for FFT-based segmentation, but no systematic evaluation exists across cell lines and quality levels.

**Problem:** Different papers use different filters with no justification. The "best" filter is unknown.

**Solution:** We evaluate 12 filter types across 8 cell lines, 4 degradation types, and 25 blur levels (20,000+ segmentations), and provide an adaptive selection algorithm.

**Key result:** No single filter is universally optimal. Adaptive selection improves IoU by +0.130 over fixed filtering. Filter transfer from HQ to LQ is <15%.

**Significance:** First comprehensive filter evaluation for phase-contrast microscopy. Provides practical, evidence-based selection guidelines.

---

## 7. RISK ASSESSMENT

### 7.1 Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Nature Methods desk rejection (scope) | High | High | Ensure single clear contribution; strong cover letter |
| Reviewer requests additional experiments | Medium | Medium | Pre-generate ablation studies; have data ready |
| Reference count too low | Low | Medium | Expand literature review; add 40 more references |
| Figure quality insufficient | Low | High | All figures at 300 DPI minimum; vector for schematics |
| Methods section too long | Medium | Medium | Move details to supplementary; keep Online Methods focused |

### 7.2 Strengths to Emphasize

- Large-scale evaluation (20,000+ segmentations, 16,912 synthetic + 19,200 real LQ images)
- First systematic filter comparison for phase-contrast microscopy
- Physics-informed enhancement is a hot topic (diffusion models, foundation models)
- Open source code and trained models available
- Cross-dataset validation (LIVECell + BBBC005)
- Practical, actionable recommendations (filter selection guide)

---

## 8. IMMEDIATE NEXT STEPS

1. **Decide target journal** for Paper 1 (Nature Methods vs. Bioinformatics vs. MIA)
2. **Create new LaTeX template** for the chosen journal
3. **Write new abstract** (150 words, single narrative)
4. **Select 6 main figures** from existing outputs
5. **Draft new Introduction** (500 words, focused narrative)
6. **Move current Methods** to Online Methods format
7. **Expand references** from ~10 to ~50

---

*Document created: 2024-06-22*
*Last updated: 2024-06-22*
*Status: Planning phase — awaiting journal decision to begin drafting*
