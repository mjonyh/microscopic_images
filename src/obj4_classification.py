#!/usr/bin/env python3
"""
Objective 4: Texture-Based Cell Line Classification
Classify cell lines using FFT features alone.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_image, list_images, compute_fft, radial_profile,
    spectral_features, get_cell_line, OUTPUT_DIR
)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract FFT feature vector from a single image."""
    power, fx, fy = compute_fft(img)
    freqs, radial = radial_profile(power, n_bins=50)
    angles, azimuthal = azimuthal_profile(power, n_bins=36)
    feats = spectral_features(power, fx, fy)

    # Combine into feature vector
    feature_vec = np.concatenate([
        radial,                          # 50 features: radial power profile
        azimuthal,                       # 36 features: angular power profile
        [feats["centroid"], feats["bandwidth"], feats["skewness"],
         feats["kurtosis"], feats["total_power"], feats["low_power"],
         feats["mid_power"], feats["high_power"]],  # 8 scalar features
    ])
    return feature_vec


def main():
    print("Objective 4: Texture-Based Cell Line Classification")
    print("=" * 55)

    images = list_images()
    print(f"  Images: {len(images)}")

    # Build feature matrix
    print("  Extracting FFT features...")
    X = []
    y = []
    filenames = []
    for i, path in enumerate(images):
        if i % 500 == 0:
            print(f"    {i}/{len(images)}...")
        img = load_image(path)
        power, fx, fy = compute_fft(img)
        freqs, radial = radial_profile(power, n_bins=50)
        angles, azimuthal = azimuthal_profile(power, n_bins=36)
        feats = spectral_features(power, fx, fy)
        vec = np.concatenate([
            radial, azimuthal,
            [feats["centroid"], feats["bandwidth"], feats["skewness"],
             feats["kurtosis"], feats["total_power"], feats["low_power"],
             feats["mid_power"], feats["high_power"]],
        ])
        X.append(vec)
        y.append(get_cell_line(path.stem))
        filenames.append(path.stem)

    X = np.array(X)
    y = np.array(y)
    print(f"  Feature matrix: {X.shape}")

    # Classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", random_state=42))]),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n  Cross-validation results (5-fold):")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = scores
        print(f"    {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Best classifier confusion matrix
    best_name = max(results, key=lambda k: results[k].mean())
    best_clf = classifiers[best_name]
    print(f"\n  Best classifier: {best_name}")

    # Refit on full data for confusion matrix
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(best_clf, X, y, cv=cv, n_jobs=-1)

    # Save results
    cell_lines = sorted(set(y))
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    csv_path = OUTPUT_DIR / "obj4_classification_report.csv"
    report_df.to_csv(csv_path)
    print(f"  Classification report saved: {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Objective 4: Cell Line Classification from FFT Features", fontsize=14, fontweight="bold")

    # (a) Classifier comparison
    ax = axes[0]
    names = list(results.keys())
    means = [results[n].mean() for n in names]
    stds = [results[n].std() for n in names]
    bars = ax.barh(names, means, xerr=stds, color=["#89b4fa", "#a6e3a1", "#f9e2af"])
    ax.set_xlabel("Accuracy")
    ax.set_title("5-Fold CV Accuracy")
    ax.set_xlim(0, 1)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=10)

    # (b) Confusion matrix
    ax = axes[1]
    cm = confusion_matrix(y, y_pred, labels=cell_lines)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cell_lines)))
    ax.set_yticks(range(len(cell_lines)))
    ax.set_xticklabels(cell_lines, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cell_lines, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({best_name})")
    # Add text annotations
    for i in range(len(cell_lines)):
        for j in range(len(cell_lines)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj4_classification.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for cl in cell_lines:
        mask = y == cl
        acc = (y_pred[mask] == cl).mean()
        print(f"    {cl:10s}: {acc:.3f} ({mask.sum()} images)")

    print("  Done.")


if __name__ == "__main__":
    main()
