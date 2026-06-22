# Tutorial 7: Evaluation Metrics and Statistical Tests

## Overview

This tutorial documents the evaluation metrics and statistical tests used to quantify segmentation performance, classification accuracy, and the significance of filter improvements in phase-contrast microscopy images.

## Segmentation Metrics

### Intersection-over-Union (IoU)

The primary metric for segmentation quality:

```python
def iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Intersection-over-Union between predicted and ground-truth masks.
    
    IoU = |pred ∩ gt| / |pred ∪ gt|
    
    Range: 0 (no overlap) to 1 (perfect overlap)
    """
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union
```

### Dice Coefficient

```python
def dice_coefficient(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Dice coefficient (F1-score for segmentation).
    
    Dice = 2|pred ∩ gt| / (|pred| + |gt|)
    
    Range: 0 to 1
    """
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    
    if total == 0:
        return 1.0
    
    return 2.0 * intersection / total
```

### Precision and Recall

```python
def precision_recall(pred_mask, gt_mask, threshold=0.5):
    """Compute precision and recall for segmentation."""
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall
```

## Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification(y_true, y_pred, class_names):
    """Comprehensive classification evaluation."""
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'per_class': report,
        'confusion_matrix': cm
    }
```

## Statistical Tests

### Paired t-test (Filter Comparison)

Used to determine if the difference in IoU between two methods is statistically significant:

```python
from scipy import stats

def paired_ttest(iou_method1, iou_method2, alpha=0.05):
    """
    Paired t-test comparing two methods across images.
    
    H0: mean difference = 0
    H1: mean difference ≠ 0
    
    Returns: t_statistic, p_value, significant
    """
    differences = iou_method1 - iou_method2
    
    # Check normality of differences (Shapiro-Wilk test)
    _, normality_p = stats.shapiro(differences)
    
    if normality_p > 0.05:
        # Normal distribution: use paired t-test
        t_stat, p_value = stats.ttest_rel(iou_method1, iou_method2)
    else:
        # Non-normal: use Wilcoxon signed-rank test
        t_stat, p_value = stats.wilcoxon(iou_method1, iou_method2)
    
    significant = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'mean_difference': differences.mean(),
        'std_difference': differences.std()
    }
```

### Bonferroni Correction

When performing multiple comparisons, the significance threshold must be adjusted:

```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Adjusted alpha = alpha / n_comparisons
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    
    results = []
    for p in p_values:
        results.append({
            'original_p': p,
            'adjusted_alpha': adjusted_alpha,
            'significant': p < adjusted_alpha
        })
    
    return results
```

### Cohen's d Effect Size

Measures the practical significance of the difference:

```python
def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size.
    
    d < 0.2: negligible
    0.2 ≤ d < 0.5: small
    0.5 ≤ d < 0.8: medium
    d ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    mean_diff = group1.mean() - group2.mean()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std()**2 + (n2 - 1) * group2.std()**2) / (n1 + n2 - 2)
    )
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std
```

### Complete Statistical Analysis

```python
def statistical_analysis(iou_raw, iou_filtered, iou_enhanced, alpha=0.05):
    """
    Complete statistical analysis comparing three methods.
    """
    comparisons = [
        ('Raw vs Filtered', iou_raw, iou_filtered),
        ('Raw vs Enhanced', iou_raw, iou_enhanced),
        ('Filtered vs Enhanced', iou_filtered, iou_enhanced)
    ]
    
    p_values = []
    results = {}
    
    for name, g1, g2 in comparisons:
        t_stat, p_val = stats.ttest_rel(g1, g2)
        d = cohens_d(g1, g2)
        
        results[name] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'mean_diff': (g1 - g2).mean(),
            'effect_size': 'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small' if abs(d) >= 0.2 else 'negligible'
        }
        p_values.append(p_val)
    
    # Bonferroni correction
    corrected = bonferroni_correction(p_values, alpha)
    
    for i, (name, _) in enumerate(comparisons):
        results[name]['p_value_corrected'] = corrected[i]['adjusted_alpha']
        results[name]['significant'] = corrected[i]['significant']
    
    return results
```

## Usage Example

```python
# Load IoU results for each method
iou_raw = np.array([0.32, 0.45, 0.28, ...])      # Raw images
iou_filtered = np.array([0.39, 0.52, 0.31, ...])  # After DoG filtering
iou_enhanced = np.array([0.42, 0.55, 0.35, ...]) # After DeBCR+DoG

# Run statistical analysis
results = statistical_analysis(iou_raw, iou_filtered, iou_enhanced)

for comparison, stats_dict in results.items():
    print(f"\n{comparison}:")
    print(f"  Mean difference: {stats_dict['mean_diff']:.4f}")
    print(f"  p-value: {stats_dict['p_value']:.6f}")
    print(f"  Cohen's d: {stats_dict['cohens_d']:.3f} ({stats_dict['effect_size']})")
    print(f"  Significant: {stats_dict['significant']}")
```

## Results from Our Evaluation

### Filter Comparison (noise_50 degradation)

| Comparison | Mean Δ | p-value | Cohen's d | Significant |
|------------|--------|---------|-----------|-------------|
| Raw vs DoG | +0.002 | 0.0008 | 0.17 | Yes |
| Raw vs DeBCR | -0.007 | 0.0007 | -0.56 | Yes |
| Raw vs DeBCR+DoG | +0.057 | 0.0000 | 0.44 | Yes |
| DoG vs DeBCR+DoG | +0.055 | 0.0000 | 0.43 | Yes |

### Adaptive vs Fixed Filtering

| Comparison | Mean Δ | p-value | Cohen's d | Significant |
|------------|--------|---------|-----------|-------------|
| Fixed vs Adaptive | +0.130 | <0.0001 | 0.82 | Yes |

## Key Points

1. **IoU is the primary metric**: It directly measures segmentation overlap with ground truth
2. **Paired t-test is appropriate**: Same images evaluated under different conditions
3. **Bonferroni correction is essential**: Multiple comparisons inflate false positive rate
4. **Cohen's d provides practical significance**: Statistical significance ≠ practical importance
5. **Report both**: Always report both p-values and effect sizes

## Source Code

- Metrics: `src/common.py`
- Statistical tests: `src/ws4_manuscript.py`
- Classification evaluation: `src/obj4_classification.py`
- Filter evaluation: `src/obj5_segmentation_filter.py`
