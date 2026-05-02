"""
evaluate.py
-----------
Metric computation for Model A and Model B.
Produces accuracy, F1, Exact Match, Precision, Recall, R², Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    classification_report,
)


# ── Model A Metrics ───────────────────────────────────────────────────────────
def exact_match(y_true, y_pred) -> float:
    """Strict character-level match — fraction of exact label matches."""
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def model_a_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Compute all Model A evaluation metrics.
    y_true / y_pred: integer-encoded labels (0=A, 1=B, 2=C, 3=D)
    """
    label_names = label_names or ["A", "B", "C", "D"]
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    em   = exact_match(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names)

    return {
        "accuracy":         round(acc,  4),
        "macro_f1":         round(f1,   4),
        "exact_match":      round(em,   4),
        "macro_precision":  round(prec, 4),
        "macro_recall":     round(rec,  4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ── Model B Metrics ───────────────────────────────────────────────────────────
def model_b_distractor_metrics(y_true, y_pred) -> dict:
    """
    Metrics for distractor ranker (binary: 1=good distractor, 0=not).
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }


def model_b_hint_metrics(y_true_scores, y_pred_scores) -> dict:
    """
    Regression metrics for hint scorer (predicted vs true relevance scores).
    """
    r2   = r2_score(y_true_scores, y_pred_scores)
    rmse = float(np.sqrt(mean_squared_error(y_true_scores, y_pred_scores)))
    mae  = mean_absolute_error(y_true_scores, y_pred_scores)

    return {
        "r2_score": round(r2,   4),
        "rmse":     round(rmse, 4),
        "mae":      round(mae,  4),
    }


# ── Clustering Metrics ────────────────────────────────────────────────────────
def clustering_purity(y_true, cluster_labels) -> float:
    """Compute clustering purity score."""
    from collections import Counter
    total = len(y_true)
    purity = 0.0
    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        most_common = Counter(y_true[mask]).most_common(1)[0][1]
        purity += most_common
    return round(purity / total, 4)


def silhouette(X, labels) -> float:
    from sklearn.metrics import silhouette_score
    try:
        return round(float(silhouette_score(X, labels, sample_size=2000)), 4)
    except Exception:
        return 0.0


def print_summary(metrics: dict, title: str = "Metrics"):
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "classification_report"):
            print(f"  {k:<28} {v}")
    if "classification_report" in metrics:
        print(metrics["classification_report"])
    if "confusion_matrix" in metrics:
        print("  Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print("   ", row)
