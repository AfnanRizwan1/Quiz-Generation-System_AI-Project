"""
evaluate.py
-----------
Metric computation for Model A and Model B.

Model A: MCQ accuracy, Macro-F1, Exact Match, Precision, Recall,
         Confusion Matrix, clustering metrics
Model B: Binary classification metrics (distractor ranker),
         Binary classification metrics (hint scorer),
         NDCG@K for ranking quality
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ndcg_score,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model A — MCQ Metrics
# ══════════════════════════════════════════════════════════════════════════════
def exact_match(y_true, y_pred) -> float:
    """Fraction of predictions that exactly match the true label."""
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def model_a_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Full Model A evaluation suite.
    y_true / y_pred: integer-encoded MCQ labels (0=A, 1=B, 2=C, 3=D).
    """
    label_names = label_names or ["A", "B", "C", "D"]
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    em     = exact_match(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0)

    return {
        "accuracy":              round(acc,  4),
        "macro_f1":              round(f1,   4),
        "exact_match":           round(em,   4),
        "macro_precision":       round(prec, 4),
        "macro_recall":          round(rec,  4),
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


def mcq_accuracy_from_binary(y_bin_true, y_bin_pred_proba,
                              n_options: int = 4) -> float:
    """
    Compute MCQ accuracy from binary classifier outputs.

    y_bin_true       : (N*4,) binary labels — 1 for correct option
    y_bin_pred_proba : (N*4,) predicted probabilities of being correct
    n_options        : number of options per question (default 4)

    For each question, picks the option with the highest predicted
    probability and checks if it matches the true correct option.
    """
    n_questions = len(y_bin_true) // n_options
    assert len(y_bin_true) == n_questions * n_options

    proba   = np.array(y_bin_pred_proba).reshape(n_questions, n_options)
    labels  = np.array(y_bin_true).reshape(n_questions, n_options)

    predicted = np.argmax(proba,  axis=1)   # predicted correct option index
    true_ans  = np.argmax(labels, axis=1)   # actual correct option index

    return float((predicted == true_ans).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Model B — Distractor Ranker Metrics
# ══════════════════════════════════════════════════════════════════════════════
def model_b_distractor_metrics(y_true, y_pred, y_proba=None,
                                group_size: int = 6) -> dict:
    """
    Metrics for the distractor ranker (binary classifier).

    y_true      : (N,) binary labels  — 1=good distractor, 0=hard negative
    y_pred      : (N,) predicted labels
    y_proba     : (N,) predicted probabilities (optional, for NDCG)
    group_size  : samples per question group for NDCG (3 pos + 3 neg = 6)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }

    # NDCG@3 — ranking quality within each question group
    if y_proba is not None:
        ndcg_scores = []
        n_groups = len(y_true) // group_size
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            yt = np.array(y_true[sl], dtype=float).reshape(1, -1)
            yp = np.array(y_proba[sl]).reshape(1, -1)
            try:
                ndcg_scores.append(ndcg_score(yt, yp, k=3))
            except Exception:
                pass
        result["ndcg_at_3"] = round(float(np.mean(ndcg_scores)), 4) \
            if ndcg_scores else 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Model B — Hint Scorer Metrics
# ══════════════════════════════════════════════════════════════════════════════
def model_b_hint_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Metrics for the hint scorer (binary classifier).

    y_true  : (N,) binary labels  — 1=relevant sentence, 0=not
    y_pred  : (N,) predicted labels
    y_proba : (N,) predicted probabilities (optional, for precision@K)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }

    # Precision@3: of the top-3 ranked sentences, how many are truly relevant?
    if y_proba is not None:
        top3_idx = np.argsort(y_proba)[-3:]
        p_at_3   = float(np.array(y_true)[top3_idx].mean())
        result["precision_at_3"] = round(p_at_3, 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Clustering Metrics  (for K-Means / LabelSpreading evaluation)
# ══════════════════════════════════════════════════════════════════════════════
def clustering_purity(y_true, cluster_labels) -> float:
    """
    Clustering purity: fraction of samples in the majority class per cluster.
    Uses np.unique for reliable operation on numpy arrays.
    """
    y_true        = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    total  = len(y_true)
    purity = 0.0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        most_common = int(np.bincount(y_true[mask]).max())
        purity += most_common
    return round(purity / total, 4)


def silhouette(X, labels, sample_size: int = 2000) -> float:
    from sklearn.metrics import silhouette_score
    try:
        return round(float(
            silhouette_score(X, labels, sample_size=sample_size,
                             random_state=42)), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Pretty Printer
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(metrics: dict, title: str = "Metrics"):
    skip = {"confusion_matrix", "classification_report"}
    print(f"\n{'='*44}")
    print(f"  {title}")
    print(f"{'='*44}")
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<30} {v}")
    if "classification_report" in metrics:
        print(metrics["classification_report"])
    if "confusion_matrix" in metrics:
        print("  Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print("   ", row)
