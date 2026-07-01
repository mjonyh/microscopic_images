---
title: Tutorial 7 - Evaluation Metrics and Statistical Tests for Microscopy Analysis
author: Prof. Dr. Md. Enamul Hoque
date: 2026-07-01
version: 1.1
prerequisites: Tutorial 4 (U-Net Segmentation), Tutorial 5 (Adaptive Filtering)
estimated_time: 90 minutes
difficulty: Advanced
---

**Previous:** [Tutorial 6: Synthetic Degradation](06_synthetic_degradation.md) | **Next:** [Tutorial Index](README.md)

# Tutorial 7: Evaluation Metrics and Statistical Tests

## Learning Objectives

By the end of this tutorial, you will be able to:
- [ ] Understand and implement segmentation metrics (IoU, Dice, precision, recall)
- [ ] Understand and implement classification metrics (accuracy, F1, confusion matrix)
- [ ] Implement image quality metrics (PSNR, SSIM, MSE)
- [ ] Perform statistical significance tests (paired t-test, Wilcoxon, ANOVA)
- [ ] Apply multiple comparison corrections (Bonferroni, Holm)
- [ ] Visualize evaluation results with confidence intervals
- [ ] Interpret p-values and effect sizes in microscopy context

## Overview

This tutorial documents the evaluation metrics and statistical tests used to quantify segmentation performance, classification accuracy, and the significance of filter improvements in phase-contrast microscopy images. Rigorous evaluation is crucial for validating the effectiveness of image processing and machine learning methods.

## Segmentation Metrics

### Intersection-over-Union (IoU / Jaccard Index)

The **primary metric** for segmentation quality. IoU measures the overlap between predicted and ground truth masks:

```python
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt

def iou(pred_mask: np.ndarray, gt_mask: np.ndarray, 
        threshold: float = 0.5) -> float:
    """
    Compute Intersection-over-Union between predicted and ground-truth masks.
    
    IoU = |pred ∩ gt| / |pred ∪ gt|
    
    Args:
        pred_mask: Predicted segmentation mask as 2D numpy array (0-1 or 0-255)
        gt_mask: Ground truth segmentation mask as 2D numpy array (0-1 or 0-255)
        threshold: Threshold for binarizing predicted mask (default: 0.5)
        
    Returns:
        IoU score in range [0, 1] where 0 = no overlap, 1 = perfect overlap
        
    Note:
        IoU is equivalent to the Jaccard Index. It penalizes both false positives 
        and false negatives equally.
    """
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Handle edge cases
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)

def batch_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
              threshold: float = 0.5) -> np.ndarray:
    """
    Compute IoU for a batch of predictions and ground truth masks.
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        threshold: Binarization threshold
        
    Returns:
        Array of IoU scores for each pair
    """
    return np.array([iou(p, g, threshold) for p, g in zip(pred_masks, gt_masks)])
```

**Interpretation:**
- **1.0**: Perfect overlap (predicted mask exactly matches ground truth)
- **0.8-1.0**: Excellent segmentation
- **0.6-0.8**: Good segmentation
- **0.4-0.6**: Moderate segmentation
- **0.0-0.4**: Poor segmentation
- **0.0**: No overlap at all

### Dice Coefficient (F1-Score for Segmentation)

The Dice coefficient is related to IoU but gives more weight to true positives:

```python
def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray,
                     threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient (F1-score for segmentation).
    
    Dice = 2|pred ∩ gt| / (|pred| + |gt|)
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        threshold: Threshold for binarizing predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient in range [0, 1]
        
    Note:
        Dice is related to IoU: Dice = 2*IoU / (1 + IoU)
        Dice gives more weight to true positives than IoU.
    """
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    # Compute intersection and total
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    
    # Handle edge cases
    if total == 0:
        return 1.0
    
    return float((2.0 * intersection + smooth) / (total + smooth))

def batch_dice(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
               threshold: float = 0.5) -> np.ndarray:
    """Compute Dice for a batch of predictions."""
    return np.array([dice_coefficient(p, g, threshold) for p, g in zip(pred_masks, gt_masks)])
```

**Relationship between IoU and Dice:**
```
Dice = 2 * IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

### Precision and Recall

Precision and recall measure different aspects of segmentation quality:

```python
def precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray,
                     threshold: float = 0.5) -> Tuple[float, float]:
    """
    Compute precision and recall for segmentation.
    
    Precision = TP / (TP + FP) = true positives / all predicted positives
    Recall = TP / (TP + FN) = true positives / all actual positives
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        threshold: Threshold for binarizing predicted mask
        
    Returns:
        Tuple of (precision, recall)
        
    Note:
        - High precision: Few false positives (conservative segmentation)
        - High recall: Few false negatives (liberal segmentation)
        - Perfect segmentation: precision = recall = 1.0
    """
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    # Compute true positives, false positives, false negatives
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    
    # Compute precision and recall
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def f1_score(precision: float, recall: float) -> float:
    """
    Compute F1-score from precision and recall.
    
    F1 = 2 * precision * recall / (precision + recall)
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1-score (harmonic mean of precision and recall)
    """
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
```

**Interpretation:**
- **High Precision, Low Recall**: Conservative segmentation (misses some cells but rarely includes background)
- **Low Precision, High Recall**: Liberal segmentation (captures most cells but includes background)
- **Balanced**: Both precision and recall are high

### Fβ-Score (Generalized F-Measure)

For applications where precision and recall have different importance:

```python
def fbeta_score(pred_mask: np.ndarray, gt_mask: np.ndarray,
                threshold: float = 0.5, beta: float = 1.0) -> float:
    """
    Compute Fβ-score, a weighted harmonic mean of precision and recall.
    
    Fβ = (1 + β²) * precision * recall / (β² * precision + recall)
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        threshold: Threshold for binarizing predicted mask
        beta: Weight for recall vs precision
            - beta=1.0: F1-score (balanced)
            - beta>1: More weight to recall (e.g., beta=2 for F2-score)
            - beta<1: More weight to precision (e.g., beta=0.5)
        
    Returns:
        Fβ-score in range [0, 1]
    """
    precision, recall = precision_recall(pred_mask, gt_mask, threshold)
    
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)
```

### Hausdorff Distance

Measures the maximum distance between the boundaries of predicted and ground truth masks:

```python
def hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray,
                       threshold: float = 0.5) -> float:
    """
    Compute Hausdorff distance between two binary masks.
    
    HD = max( sup_{p∈pred} inf_{g∈gt} d(p,g), sup_{g∈gt} inf_{p∈pred} d(p,g) )
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        threshold: Threshold for binarizing predicted mask
        
    Returns:
        Hausdorff distance in pixels
        
    Note:
        Hausdorff distance is sensitive to outliers (a single far-away point 
        can dominate the metric). Use 95th percentile Hausdorff distance 
        for more robustness.
    """
    from scipy.ndimage import distance_transform_edt
    
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    # Handle edge case
    if not pred_binary.any() or not gt_binary.any():
        return float('inf')
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_binary)
    gt_dist = distance_transform_edt(~gt_binary)
    
    # Compute Hausdorff distance
    hd1 = pred_dist[gt_binary].max()
    hd2 = gt_dist[pred_binary].max()
    
    return float(max(hd1, hd2))

def hausdorff_95(pred_mask: np.ndarray, gt_mask: np.ndarray,
                  threshold: float = 0.5) -> float:
    """
    Compute 95th percentile Hausdorff distance (more robust to outliers).
    """
    from scipy.ndimage import distance_transform_edt
    
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > threshold).astype(bool)
    gt_binary = (gt_mask > threshold).astype(bool)
    
    # Handle edge case
    if not pred_binary.any() or not gt_binary.any():
        return float('inf')
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_binary)
    gt_dist = distance_transform_edt(~gt_binary)
    
    # Compute 95th percentile Hausdorff distance
    hd1 = np.percentile(pred_dist[gt_binary], 95)
    hd2 = np.percentile(gt_dist[pred_binary], 95)
    
    return float(max(hd1, hd2))
```

### Visualization of Segmentation Metrics

```python
def visualize_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                    image: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize segmentation results with metrics overlay.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        image: Optional original image for background
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    
    # Binarize masks
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    
    pred_binary = (pred_mask > 0.5).astype(np.float64)
    gt_binary = (gt_mask > 0.5).astype(np.float64)
    
    # Compute metrics
    iou_score = iou(pred_mask, gt_mask)
    dice_score = dice_coefficient(pred_mask, gt_mask)
    precision, recall = precision_recall(pred_mask, gt_mask)
    f1 = f1_score(precision, recall)
    
    # Create figure
    if image is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image (if provided)
    if image is not None:
        axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        ax_idx = 1
    else:
        ax_idx = 0
    
    # Ground truth
    axes[ax_idx].imshow(gt_binary, cmap='gray', vmin=0, vmax=1)
    axes[ax_idx].set_title('Ground Truth')
    axes[ax_idx].axis('off')
    
    # Predicted
    axes[ax_idx + 1].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
    axes[ax_idx + 1].set_title('Predicted')
    axes[ax_idx + 1].axis('off')
    
    # Overlay: True Positives (Green), False Positives (Red), False Negatives (Blue)
    tp = np.logical_and(pred_binary, gt_binary)
    fp = np.logical_and(pred_binary, ~gt_binary)
    fn = np.logical_and(~pred_binary, gt_binary)
    
    overlay = np.zeros((*pred_binary.shape, 3))
    overlay[tp, 1] = 1.0  # Green for TP
    overlay[fp, 0] = 1.0  # Red for FP
    overlay[fn, 2] = 1.0  # Blue for FN
    
    axes[ax_idx + 2].imshow(overlay)
    axes[ax_idx + 2].set_title('TP (Green), FP (Red), FN (Blue)')
    axes[ax_idx + 2].axis('off')
    
    # Metrics overlay
    metrics_text = f"IoU: {iou_score:.3f}\n"
    metrics_text += f"Dice: {dice_score:.3f}\n"
    metrics_text += f"Precision: {precision:.3f}\n"
    metrics_text += f"Recall: {recall:.3f}\n"
    metrics_text += f"F1: {f1:.3f}"
    
    if image is not None:
        axes[3].axis('off')
        axes[3].text(0.1, 0.8, metrics_text, fontsize=12, va='top',
                     family='monospace', bbox=dict(facecolor='white', alpha=0.8))
        axes[3].set_title('Metrics')
    else:
        axes[2].text(0.05, 0.05, metrics_text, fontsize=10, va='bottom',
                     ha='left', family='monospace',
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.suptitle('Segmentation Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
```

## Classification Metrics

For cell classification tasks (e.g., cell line classification from FFT features):

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List

def evaluate_classification(y_true: Union[List[int], np.ndarray],
                            y_pred: Union[List[int], np.ndarray],
                            class_names: List[str] = None) -> Dict[str, any]:
    """
    Comprehensive classification evaluation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names for reporting
        
    Returns:
        Dictionary with classification metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_true)]
    
    report = classification_report(y_true, y_pred, 
                                    target_names=class_names, 
                                    output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': accuracy,
        'per_class': report,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy,
        'class_names': class_names
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as 2D numpy array
        class_names: List of class names
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.set_title('Normalized Confusion Matrix')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Proportion')
    
    # Add labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
```

## Image Quality Metrics

For evaluating image enhancement and restoration:

```python
def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.
    
    MSE = (1/MN) * Σ(x,y) [I1(x,y) - I2(x,y)]²
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        MSE value (lower is better)
    """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    return float(np.mean((image1 - image2) ** 2))

def rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(MSE)
    """
    return float(np.sqrt(mse(image1, image2)))

def psnr(image1: np.ndarray, image2: np.ndarray, 
         data_range: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(data_range² / MSE)
    
    Args:
        image1: First image
        image2: Second image
        data_range: Maximum possible pixel value (default: 255)
        
    Returns:
        PSNR in decibels (higher is better)
    """
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10((data_range ** 2) / mse_val))

def ssim(image1: np.ndarray, image2: np.ndarray, 
         data_range: float = 255.0, window_size: int = 11,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute Structural Similarity Index.
    
    SSIM(x,y) = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    
    Args:
        image1: First image
        image2: Second image
        data_range: Maximum possible pixel value
        window_size: Size of sliding window
        k1: Constant for mean comparison
        k2: Constant for variance comparison
        
    Returns:
        SSIM value in range [-1, 1] (higher is better, 1 = identical)
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import convolve2d
    
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    
    # Create Gaussian window
    sigma = 1.5
    size = window_size
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)
    window = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    window = window / window.sum()
    
    # Compute means
    mu1 = convolve2d(image1, window, mode='valid')
    mu2 = convolve2d(image2, window, mode='valid')
    
    # Compute variances
    sigma1_sq = convolve2d(image1**2, window, mode='valid') - mu1**2
    sigma2_sq = convolve2d(image2**2, window, mode='valid') - mu2**2
    
    # Compute covariance
    sigma12 = convolve2d(image1 * image2, window, mode='valid') - mu1 * mu2
    
    # Constants
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    return float(ssim_map.mean())
```

## Statistical Tests

### Paired t-test (Filter Comparison)

Used to determine if the difference in performance between two methods is statistically significant:

```python
from scipy import stats
from typing import Tuple

def paired_ttest(metric1: Union[List[float], np.ndarray],
                 metric2: Union[List[float], np.ndarray],
                 alpha: float = 0.05, 
                 test_type: str = 'auto') -> Dict[str, any]:
    """
    Paired t-test comparing two methods across images.
    
    H0: mean difference = 0 (no difference between methods)
    H1: mean difference ≠ 0 (methods are different)
    
    Args:
        metric1: Performance metrics for method 1 (e.g., IoU scores)
        metric2: Performance metrics for method 2
        alpha: Significance level (default: 0.05)
        test_type: 'auto' (check normality), 'ttest' (force t-test), 'wilcoxon' (force Wilcoxon)
        
    Returns:
        Dictionary with test results including t_statistic, p_value, significant flag
    """
    metric1 = np.array(metric1)
    metric2 = np.array(metric2)
    
    differences = metric1 - metric2
    
    # Check normality of differences (Shapiro-Wilk test)
    if test_type == 'auto':
        _, normality_p = stats.shapiro(differences)
        
        if normality_p > 0.05:
            # Normal distribution: use paired t-test
            t_stat, p_value = stats.ttest_rel(metric1, metric2)
            test_used = 'paired_ttest'
        else:
            # Non-normal: use Wilcoxon signed-rank test
            t_stat, p_value = stats.wilcoxon(metric1, metric2)
            test_used = 'wilcoxon'
    elif test_type == 'ttest':
        t_stat, p_value = stats.ttest_rel(metric1, metric2)
        test_used = 'paired_ttest'
    else:  # wilcoxon
        t_stat, p_value = stats.wilcoxon(metric1, metric2)
        test_used = 'wilcoxon'
    
    significant = p_value < alpha
    
    # Effect size (Cohen's d)
    d = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'mean_difference': float(differences.mean()),
        'std_difference': float(differences.std()),
        'effect_size': float(d),
        'n_samples': len(differences),
        'test_used': test_used,
        'alpha': alpha
    }
```

### Independent t-test (Group Comparison)

For comparing performance across independent groups (e.g., different cell lines):

```python
def independent_ttest(group1: Union[List[float], np.ndarray],
                     group2: Union[List[float], np.ndarray],
                     alpha: float = 0.05) -> Dict[str, any]:
    """
    Independent t-test comparing two independent groups.
    
    H0: mean(group1) = mean(group2)
    H1: mean(group1) ≠ mean(group2)
    
    Args:
        group1: Metrics for group 1
        group2: Metrics for group 2
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Check for equal variance (Levene's test)
    _, lev_p = stats.levene(group1, group2)
    
    if lev_p > 0.05:
        # Equal variance
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        test_used = 'ttest_ind (equal_var)'
    else:
        # Unequal variance (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_used = 'ttest_ind (unequal_var)'
    
    significant = p_value < alpha
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'mean_group1': float(group1.mean()),
        'mean_group2': float(group2.mean()),
        'std_group1': float(group1.std()),
        'std_group2': float(group2.std()),
        'effect_size': float(d),
        'n_group1': len(group1),
        'n_group2': len(group2),
        'test_used': test_used,
        'alpha': alpha
    }
```

### ANOVA (Multiple Group Comparison)

For comparing performance across multiple groups:

```python
def one_way_anova(*groups: Union[List[float], np.ndarray],
                  alpha: float = 0.05) -> Dict[str, any]:
    """
    One-way ANOVA for comparing multiple independent groups.
    
    H0: All group means are equal
    H1: At least one group mean is different
    
    Args:
        *groups: Variable number of groups (arrays of metrics)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    groups = [np.array(g) for g in groups]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    significant = p_value < alpha
    
    # Effect size (eta squared)
    grand_mean = np.mean([g.mean() for g in groups])
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum(np.sum((g - g.mean())**2) for g in groups) + ss_between
    eta_sq = ss_between / (ss_total + 1e-10)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': significant,
        'group_means': [float(g.mean()) for g in groups],
        'group_stds': [float(g.std()) for g in groups],
        'group_sizes': [len(g) for g in groups],
        'effect_size_eta_sq': float(eta_sq),
        'n_groups': len(groups),
        'alpha': alpha
    }

# Post-hoc tests (Tukey HSD)
def tukey_hsd(*groups: Union[List[float], np.ndarray],
              alpha: float = 0.05) -> Dict[str, any]:
    """
    Tukey's Honestly Significant Difference test for post-hoc ANOVA comparisons.
    
    Args:
        *groups: Variable number of groups
        alpha: Significance level
        
    Returns:
        Dictionary with pairwise comparison results
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    groups = [np.array(g) for g in groups]
    
    # Flatten data and create group labels
    all_data = np.concatenate(groups)
    group_labels = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    
    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
    
    # Convert to dictionary
    results = {
        'summary': str(tukey.summary()),
        'group_means': [float(g.mean()) for g in groups],
        'comparisons': []
    }
    
    for i, comparison in enumerate(tukey.summary_data):
        results['comparisons'].append({
            'group1': int(comparison[0]),
            'group2': int(comparison[1]),
            'mean_diff': float(comparison[2]),
            'p_value': float(comparison[4]),
            'conf_int_low': float(comparison[5]),
            'conf_int_high': float(comparison[6]),
            'significant': comparison[4] < alpha
        })
    
    return results
```

### Multiple Comparison Corrections

When performing multiple statistical tests, the significance threshold must be adjusted:

```python
def bonferroni_correction(p_values: Union[List[float], np.ndarray],
                         alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Adjusted alpha = alpha / n_comparisons
    
    Args:
        p_values: Array of p-values
        alpha: Original significance level
        
    Returns:
        Tuple of (adjusted_p_values, adjusted_alpha)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    adjusted_alpha = alpha / n
    adjusted_p_values = p_values * n
    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)
    
    return adjusted_p_values, adjusted_alpha

def holm_bonferroni_correction(p_values: Union[List[float], np.ndarray],
                               alpha: float = 0.05) -> Tuple[np.ndarray, List[bool]]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni).
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted_p_values, list of significant flags)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    adjusted_p = np.zeros(n)
    significant = [False] * n
    
    for i, p in enumerate(sorted_p):
        adjusted_p[sorted_indices[i]] = p * (n - i)
        if adjusted_p[sorted_indices[i]] < alpha:
            significant[sorted_indices[i]] = True
    
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    return adjusted_p, significant
```

### Statistical Test Visualization

```python
def plot_statistical_comparison(metrics: Dict[str, List[float]],
                                alpha: float = 0.05,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot statistical comparison of multiple methods with confidence intervals.
    
    Args:
        metrics: Dictionary of method_name -> list of metric values
        alpha: Significance level
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    
    method_names = list(metrics.keys())
    n_methods = len(method_names)
    
    # Compute statistics
    means = [np.mean(metrics[m]) for m in method_names]
    stds = [np.std(metrics[m], ddof=1) for m in method_names]
    cis = [1.96 * s / np.sqrt(len(metrics[m])) for m, s in zip(method_names, stds)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot means with error bars
    x = np.arange(n_methods)
    ax.errorbar(x, means, yerr=cis, fmt='o', capsize=5, 
               color='black', markersize=8, label='95% CI')
    
    # Add significance markers
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            # Perform paired t-test
            result = paired_ttest(metrics[method_names[i]], metrics[method_names[j]], alpha=alpha)
            
            if result['significant']:
                # Add significance bar
                y = max(means[i], means[j]) + 0.05 * (max(means) - min(means))
                ax.plot([i, j], [y, y], 'k-', linewidth=1)
                ax.plot([i, j], [y + 0.01, y + 0.01], 'k*', markersize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylabel('Metric Value')
    ax.set_title('Method Comparison with 95% Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_boxplot_comparison(metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot boxplot comparison of multiple methods.
    
    Args:
        metrics: Dictionary of method_name -> list of metric values
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    
    method_names = list(metrics.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot data
    data = [metrics[m] for m in method_names]
    
    # Plot boxplot
    bp = ax.boxplot(data, labels=method_names, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add scatter points
    for i, method in enumerate(method_names):
        y = metrics[method]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=20, color='black')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Method Comparison Boxplot')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
```

## Usage Examples

### Example 1: Evaluating Segmentation Performance

```python
# Load predictions and ground truth
import glob
from PIL import Image

pred_paths = sorted(glob.glob('output/predictions/*.png'))
gt_paths = sorted(glob.glob('data/livecell/*Mask*.tif'))

# Ensure matching
assert len(pred_paths) == len(gt_paths)

# Compute metrics
pred_masks = [np.array(Image.open(p).convert('L')) for p in pred_paths]
gt_masks = [np.array(Image.open(g).convert('L')) for g in gt_paths]

ious = batch_iou(pred_masks, gt_masks)
dices = batch_dice(pred_masks, gt_masks)

print(f"Mean IoU: {ious.mean():.4f} ± {ious.std():.4f}")
print(f"Mean Dice: {dices.mean():.4f} ± {dices.std():.4f}")

# Visualize first sample
visualize_segmentation_metrics(pred_masks[0], gt_masks[0],
                                image=np.array(Image.open('output/predictions/0001.png')))
```

### Example 2: Comparing Multiple Methods

```python
# Simulate results for multiple methods
methods = {
    'Raw': np.random.normal(0.70, 0.05, 100),
    'Butterworth': np.random.normal(0.75, 0.04, 100),
    'Adaptive': np.random.normal(0.78, 0.03, 100),
    'U-Net': np.random.normal(0.82, 0.03, 100)
}

# Plot comparison
plot_statistical_comparison(methods, alpha=0.05)
plot_boxplot_comparison(methods)

# Perform pairwise comparisons
for m1 in methods:
    for m2 in methods:
        if m1 < m2:
            result = paired_ttest(methods[m1], methods[m2], alpha=0.05)
            print(f"{m1} vs {m2}: p={result['p_value']:.4f}, "
                  f"significant={result['significant']}, "
                  f"effect_size={result['effect_size']:.3f}")

# Apply Bonferroni correction
p_values = []
comparisons = []
for m1 in methods:
    for m2 in methods:
        if m1 < m2:
            result = paired_ttest(methods[m1], methods[m2], alpha=0.05)
            p_values.append(result['p_value'])
            comparisons.append(f"{m1} vs {m2}")

adjusted_p, adjusted_alpha = bonferroni_correction(p_values, alpha=0.05)
print(f"\nBonferroni corrected alpha: {adjusted_alpha:.6f}")
for comp, p, adj_p in zip(comparisons, p_values, adjusted_p):
    print(f"{comp}: p={p:.6f}, adjusted_p={adj_p:.6f}, "
          f"significant={adj_p < adjusted_alpha}")
```

### Example 3: Evaluating Image Quality

```python
# Evaluate enhancement quality
from tutorials.t06_synthetic_degradation import degrade_image, add_gaussian_noise

# Load HQ image
hq_image = np.array(Image.open('data/livecell/A172_Phase_A7_1_00d04h00m_1.tif').convert('L'))

# Create LQ version
lq_image = add_gaussian_noise(hq_image, sigma=50)

# Apply enhancement (e.g., from Tutorial 2 or 3)
from tutorials.t02_bandpass_filters import butterworth_bandpass
enhanced = apply_filter(lq_image, butterworth_bandpass, r_low=0.02, r_high=0.3, order=2)

# Compute quality metrics
print(f"MSE (LQ vs HQ): {mse(lq_image, hq_image):.2f}")
print(f"PSNR (LQ vs HQ): {psnr(lq_image, hq_image):.2f} dB")
print(f"SSIM (LQ vs HQ): {ssim(lq_image, hq_image):.4f}")

print(f"\nMSE (Enhanced vs HQ): {mse(enhanced, hq_image):.2f}")
print(f"PSNR (Enhanced vs HQ): {psnr(enhanced, hq_image):.2f} dB")
print(f"SSIM (Enhanced vs HQ): {ssim(enhanced, hq_image):.4f}")
```

## Interpretation Guidelines

### Segmentation Metrics

| IoU Range | Quality | Interpretation |
|-----------|---------|----------------|
| 0.90-1.00 | Excellent | Near-perfect segmentation |
| 0.80-0.90 | Good | Minor errors, mostly correct |
| 0.70-0.80 | Fair | Noticeable errors but usable |
| 0.60-0.70 | Poor | Significant errors |
| 0.00-0.60 | Very Poor | Mostly incorrect |

### Image Quality Metrics

| PSNR (dB) | Quality | Interpretation |
|------------|---------|----------------|
| >40 | Excellent | Nearly indistinguishable from original |
| 30-40 | Good | Minor visible differences |
| 20-30 | Fair | Noticeable differences |
| 10-20 | Poor | Significant degradation |
| <10 | Very Poor | Major degradation |

| SSIM Range | Quality | Interpretation |
|-------------|---------|----------------|
| 0.90-1.00 | Excellent | Very similar structure |
| 0.80-0.90 | Good | Some structural differences |
| 0.70-0.80 | Fair | Noticeable structural differences |
| 0.60-0.70 | Poor | Significant structural differences |
| <0.60 | Very Poor | Major structural differences |

### Statistical Significance

| p-value Range | Significance | Interpretation |
|---------------|--------------|----------------|
| p < 0.001 | *** | Very strong evidence against H0 |
| 0.001 ≤ p < 0.01 | ** | Strong evidence against H0 |
| 0.01 ≤ p < 0.05 | * | Moderate evidence against H0 |
| 0.05 ≤ p < 0.10 | . | Weak evidence against H0 |
| p ≥ 0.10 | ns | No significant evidence against H0 |

### Effect Sizes

| Cohen's d | Interpretation |
|------------|----------------|
| <0.2 | Negligible |
| 0.2-0.5 | Small |
| 0.5-0.8 | Medium |
| >0.8 | Large |

## Key Implementation Details

1. **Binarization**: All segmentation metrics require binary masks. The threshold parameter (default: 0.5) determines how continuous predictions are converted to binary.

2. **Edge Cases**: Special handling for cases where both masks are empty (return 1.0 for IoU/Dice) or one mask is empty.

3. **Numerical Stability**: Small smoothing factors (e.g., 1e-10) prevent division by zero in various calculations.

4. **Multiple Testing**: Always apply correction (Bonferroni, Holm) when performing multiple statistical tests to control the family-wise error rate.

5. **Effect Sizes**: Always report effect sizes alongside p-values to assess the practical significance of results.

## Exercises

### Beginner
1. Compute IoU and Dice for a single pair of segmentation masks
2. Perform a paired t-test comparing two sets of IoU scores
3. Visualize segmentation results with overlay

### Intermediate
1. Compare multiple segmentation methods using boxplots
2. Apply Bonferroni correction to a set of p-values
3. Compute PSNR and SSIM for image quality assessment

### Advanced
1. Implement a new segmentation metric (e.g., Average Surface Distance)
2. Perform ANOVA to compare multiple methods across cell lines
3. Create a comprehensive evaluation pipeline with multiple metrics and statistical tests

## Frequently Asked Questions

**Q: Which segmentation metric should I use?**
A: Use **IoU** as your primary metric (it's the standard for segmentation). Report **Dice** as a secondary metric. Use **precision/recall** to understand error types. Use **Hausdorff distance** if boundary accuracy is critical.

**Q: Why report both IoU and Dice?**
A: IoU and Dice measure slightly different aspects of overlap. IoU penalizes false positives and false negatives equally. Dice gives more weight to true positives. Reporting both provides a complete picture.

**Q: When should I use parametric vs non-parametric tests?**
A: Use parametric tests (t-test, ANOVA) when data is normally distributed. Use non-parametric tests (Wilcoxon, Kruskal-Wallis) when data is not normal or sample sizes are small. The paired_ttest function automatically checks normality.

**Q: What is the difference between paired and independent t-tests?**
A: Use **paired t-test** when comparing the same images with different methods (each image is its own control). Use **independent t-test** when comparing different sets of images (e.g., different cell lines).

**Q: How do I interpret a p-value?**
A: A p-value is the probability of observing the data (or more extreme) if the null hypothesis is true. A small p-value (typically <0.05) indicates strong evidence against the null hypothesis.

**Q: Why is effect size important?**
A: A small p-value indicates statistical significance, but not necessarily practical significance. Effect size measures the magnitude of the difference, which is more important for practical applications.

**Q: How do I handle multiple comparisons?**
A: Use **Bonferroni correction** for a simple, conservative approach. Use **Holm-Bonferroni** for a less conservative approach. Consider **False Discovery Rate (FDR)** control for exploratory analyses.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| IoU/Dice is NaN | Both masks are empty | Check threshold, handle edge cases |
| p-value is very small | Large effect or many samples | Check effect size, verify data |
| Statistical test fails | Non-numeric data or wrong shape | Check input types and shapes |
| Confidence intervals too wide | Small sample size | Collect more data |
| Multiple comparison issue | Many tests without correction | Apply Bonferroni or Holm correction |

## References

- Jaccard, P. (1901). Étude comparative de la distribution florale dans une portion des Alpes et des Jura. Bulletin de la Société Vaudoise des Sciences Naturelles.
- Dice, L.R. (1945). Measures of the amount of ecologic association between species. Ecology.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Lawrence Erlbaum Associates.
- Nunnally, J.C. & Bernstein, I.H. (1994). Psychometric Theory, 3rd ed. McGraw-Hill.
- Field, A. (2017). Discovering Statistics Using IBM SPSS Statistics, 5th ed. SAGE.

## How to Cite

If you use these evaluation metrics and statistical tests in your research, please cite:

```bibtex
@article{Hoque2026SPINDEEP,
  author = {Hoque, Md. Enamul},
  title = {SPINDEEP: Spectral Pipeline for Phase-Contrast Microscopy},
  journal = {Nature Methods},
  year = {2026},
  volume = {XX},
  pages = {XXX-XXX}
}
```

## Source Code

The full implementation is in:
- `src/metrics.py` - All evaluation metrics
- `src/statistics.py` - Statistical tests and corrections
- `tutorials/t07_evaluation_metrics.py` - Tutorial code

**Previous:** [Tutorial 6: Synthetic Degradation](06_synthetic_degradation.md) | **Next:** [Tutorial Index](README.md)
