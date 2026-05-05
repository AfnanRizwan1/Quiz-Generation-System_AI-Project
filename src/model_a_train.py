"""
model_a_train.py  (v2 — binary formulation)
--------------------------------------------
Model A — Answer Verifier / Question Generator

Core change from v1:
  Each MCQ → 4 binary samples (article+question+option, label=1/0).
  A binary classifier scores all 4 options; the highest-scoring one is
  selected as the predicted answer.  This is the correct ML framing for
  multiple-choice reading comprehension.

Pipeline:
  Features : TF-IDF (unigrams+bigrams) + cosine-sim + lexical  → hstack
  Models   : Logistic Regression, Linear SVM, Naive Bayes,
             Random Forest, (XGBoost if available)
  Ensemble : Soft-vote (LR + RF)
  Unsup    : K-Means + Label Propagation (evaluated with silhouette + F1)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, issparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, silhouette_score,
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = os.path.join("models", "model_a", "traditional")
os.makedirs(MODEL_DIR, exist_ok=True)

# Try importing XGBoost — optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found — skipping XGB model.")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_processed():
    def lp(f):
        with open(os.path.join(PROCESSED_DIR, f), "rb") as fh:
            return pickle.load(fh)

    X_bin    = lp("binary_features.pkl")   # (X_train, X_val, X_test) — combined
    y_bin    = lp("binary_labels.pkl")     # (y_train, y_val, y_test) — 0/1
    bin_dfs  = lp("binary_dfs.pkl")        # (train_bin, val_bin, test_bin)
    ohe      = lp("ohe_matrices.pkl")      # (X_tr_ohe, X_va_ohe, X_te_ohe)
    labels4  = lp("labels.pkl")            # (y4_train, y4_val, y4_test, le)
    return X_bin, y_bin, bin_dfs, ohe, labels4


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MCQ Accuracy  (the real metric we care about)
# ══════════════════════════════════════════════════════════════════════════════
def mcq_accuracy(model, X_bin, y4_true, n_options=4, use_proba=True):
    """
    Evaluate MCQ accuracy:
      - Score all 4 options per question with the binary classifier
      - Pick the option with the highest positive-class probability
      - Compare to the true answer label (0-3)

    Returns MCQ accuracy (float).
    """
    n_questions = X_bin.shape[0] // n_options
    assert X_bin.shape[0] == n_questions * n_options, \
        "X_bin rows must be a multiple of n_options"

    if use_proba and hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_bin)[:, 1]   # P(correct)
    else:
        # LinearSVC: use decision_function
        scores = model.decision_function(X_bin)
        if scores.ndim > 1:
            scores = scores[:, 1]

    scores = scores.reshape(n_questions, n_options)
    predicted = np.argmax(scores, axis=1)            # 0-3
    correct   = (predicted == y4_true).mean()
    return float(correct)


def evaluate_binary(model, X, y_bin, split="val"):
    """Binary classification metrics (used during training)."""
    if hasattr(model, "predict_proba"):
        preds = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    else:
        preds = model.predict(X)
    acc = accuracy_score(y_bin, preds)
    f1  = f1_score(y_bin, preds, average="macro")
    print(f"  [{split}] Binary Acc={acc:.4f}  Macro-F1={f1:.4f}")
    return acc, f1


def evaluate_mcq(model, X_bin, y4, split="val", use_proba=True):
    """MCQ-level accuracy (the headline number)."""
    acc = mcq_accuracy(model, X_bin, y4, use_proba=use_proba)
    print(f"  [{split}] MCQ Accuracy = {acc:.4f}  ({acc*100:.1f}%)")
    return acc


def print_results_table(results: dict):
    print("\n" + "="*62)
    print(f"{'Model':<28} {'Val MCQ Acc':>11} {'Test MCQ Acc':>12}")
    print("="*62)
    for name, r in results.items():
        val_s  = f"{r.get('val_mcq',  0):.4f}"
        test_s = f"{r.get('test_mcq', 0):.4f}" if "test_mcq" in r else "  —"
        print(f"  {name:<26} {val_s:>11} {test_s:>12}")
    print("="*62)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Supervised Models  (binary classifiers)
# ══════════════════════════════════════════════════════════════════════════════
def train_logistic_regression(X_train, y_train, C=0.5):
    print(f"\n--- Logistic Regression (C={C}) ---")
    model = LogisticRegression(
        C=C, max_iter=1000, solver="liblinear",
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, C=0.5):
    """
    LinearSVC wrapped in CalibratedClassifierCV so it supports predict_proba,
    which is needed for MCQ scoring.
    """
    print(f"\n--- Linear SVM (C={C}, calibrated) ---")
    base = LinearSVC(C=C, max_iter=2000, dual=False, class_weight="balanced")
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_tfidf_train, y_train):
    """
    ComplementNB works better than MultinomialNB for imbalanced / text data.
    Requires non-negative features — use raw TF-IDF block only.
    """
    print("\n--- Complement Naive Bayes ---")
    model = ComplementNB(alpha=0.3)
    model.fit(X_tfidf_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200):
    print(f"\n--- Random Forest (n={n_estimators}) ---")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    print("\n--- XGBoost ---")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


# ── Manual C sweep for LR ─────────────────────────────────────────────────────
def tune_logistic_regression(X_train, y_train, X_val, y_val_bin, y4_val):
    """Try a few C values and return the best model by MCQ val accuracy."""
    print("\n--- LR Hyperparameter Sweep ---")
    best_acc, best_model, best_C = 0.0, None, None
    for C in [0.1, 0.5, 1.0, 2.0]:
        m = LogisticRegression(C=C, max_iter=1000, solver="liblinear",
                               class_weight="balanced")
        m.fit(X_train, y_train)
        acc = mcq_accuracy(m, X_val, y4_val)
        print(f"  C={C:<4}  MCQ val acc = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_C = acc, m, C
    print(f"  Best C={best_C}  MCQ val acc={best_acc:.4f}")
    return best_model


# ── Manual C sweep for SVM ────────────────────────────────────────────────────
def tune_svm(X_train, y_train, X_val, y_val_bin, y4_val):
    """Try a few C values for LinearSVC and return the best by MCQ val accuracy."""
    print("\n--- SVM Hyperparameter Sweep ---")
    best_acc, best_model, best_C = 0.0, None, None
    for C in [0.1, 0.5, 1.0]:
        base = LinearSVC(C=C, max_iter=2000, dual=False, class_weight="balanced")
        m = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        m.fit(X_train, y_train)
        acc = mcq_accuracy(m, X_val, y4_val)
        print(f"  C={C:<4}  MCQ val acc = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_C = acc, m, C
    print(f"  Best C={best_C}  MCQ val acc={best_acc:.4f}")
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ensemble
# ══════════════════════════════════════════════════════════════════════════════
    """Try a few C values for LinearSVC and return the best by MCQ val accuracy."""
    print("\n--- SVM Hyperparameter Sweep ---")
    best_acc, best_model, best_C = 0.0, None, None
    for C in [0.1, 0.5, 1.0]:
        base = LinearSVC(C=C, max_iter=2000, dual=False, class_weight="balanced")
        m = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        m.fit(X_train, y_train)
        acc = mcq_accuracy(m, X_val, y4_val)
        print(f"  C={C:<4}  MCQ val acc = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_C = acc, m, C
    print(f"  Best C={best_C}  MCQ val acc={best_acc:.4f}")
    return best_model


def train_soft_vote_ensemble(X_train, y_train, lr_model, rf_model):
    """
    Soft-voting ensemble (LR + RF).
    Clones estimator configs from already-tuned models and re-fits on full train.
    """
    print("\n--- Soft-Vote Ensemble (LR + RF) ---")
    lr_clone = LogisticRegression(
        C=lr_model.C, max_iter=lr_model.max_iter,
        solver=lr_model.solver, class_weight=lr_model.class_weight,
    )
    rf_clone = RandomForestClassifier(
        n_estimators=rf_model.n_estimators,
        max_depth=rf_model.max_depth,
        min_samples_leaf=rf_model.min_samples_leaf,
        class_weight=rf_model.class_weight,
        n_jobs=-1, random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("lr", lr_clone), ("rf", rf_clone)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# 3. Unsupervised / Semi-Supervised  (on 4-class OHE features)
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate_kmeans(X_ohe_train, y4_train, X_ohe_val, y4_val,
                               n_clusters=4):
    """
    MiniBatchKMeans (fast on large sparse data).
    Evaluates with:
      - Silhouette score (on a 5k subsample)
      - Cluster-label mapping accuracy
    """
    print("\n--- K-Means Clustering ---")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                         n_init=10, batch_size=4096)
    km.fit(X_ohe_train)

    # ── Silhouette on subsample ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n = X_ohe_val.shape[0]
    idx = rng.choice(n, size=min(5000, n), replace=False)
    X_sub = X_ohe_val[idx]
    labels_sub = km.predict(X_sub)
    try:
        sil = silhouette_score(X_sub, labels_sub, sample_size=2000,
                               random_state=42)
    except Exception:
        sil = float("nan")
    print(f"  Silhouette score (val subsample) = {sil:.4f}")

    # ── Cluster-label mapping accuracy ───────────────────────────────────────
    cluster_labels = km.predict(X_ohe_val)
    # Map each cluster to its majority true label
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            mapping[c] = 0
            continue
        mapping[c] = int(np.bincount(y4_val[mask]).argmax())

    mapped_preds = np.array([mapping[c] for c in cluster_labels])
    purity = accuracy_score(y4_val, mapped_preds)
    print(f"  Cluster-label mapping accuracy   = {purity:.4f}")

    return km, {"silhouette": sil, "purity": purity}


def train_and_evaluate_label_spreading(X_ohe_train, y4_train,
                                        X_ohe_val, y4_val,
                                        max_samples=4000,
                                        labeled_fraction=0.15):
    """
    LabelSpreading (more robust than LabelPropagation for noisy data).
    Subsampled to max_samples to avoid OOM.
    Evaluates with MCQ-level F1 vs supervised baseline.
    """
    print("\n--- Label Spreading (Semi-Supervised) ---")
    n = X_ohe_train.shape[0]
    rng = np.random.default_rng(42)

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X_ohe_train[idx]
        y_sub = y4_train[idx].copy()
        print(f"  Subsampled {n} → {max_samples} rows")
    else:
        X_sub = X_ohe_train
        y_sub = y4_train.copy()

    # Densify (safe: 4000 × 5000 × 8 ≈ 160 MB)
    X_dense = X_sub.toarray() if issparse(X_sub) else np.asarray(X_sub)

    # Mask unlabeled
    n_unlabeled = int(len(y_sub) * (1 - labeled_fraction))
    unlabeled_idx = rng.choice(len(y_sub), size=n_unlabeled, replace=False)
    y_semi = y_sub.copy()
    y_semi[unlabeled_idx] = -1

    ls = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=100, alpha=0.2)
    ls.fit(X_dense, y_semi)

    # Evaluate on val subsample (dense)
    n_val = X_ohe_val.shape[0]
    val_idx = rng.choice(n_val, size=min(2000, n_val), replace=False)
    X_val_dense = (X_ohe_val[val_idx].toarray()
                   if issparse(X_ohe_val) else X_ohe_val[val_idx])
    preds = ls.predict(X_val_dense)
    f1 = f1_score(y4_val[val_idx], preds, average="macro")
    acc = accuracy_score(y4_val[val_idx], preds)
    print(f"  Semi-supervised  Acc={acc:.4f}  Macro-F1={f1:.4f}")
    print(f"  (evaluated on {len(val_idx)} val samples)")

    return ls, {"acc": acc, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Template-Based Question Generation  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════
WH_TEMPLATES  = ["Who", "What", "Where", "When", "Why", "How"]
COMMON_VERBS  = {"is", "was", "are", "were", "did", "does", "do", "has", "have", "had"}


def extract_candidate_sentences(article: str, answer: str, ohe_vec, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sentences = [s.strip() for s in article.split(".") if len(s.strip()) > 10]
    if not sentences:
        return []
    ans_vec   = ohe_vec.transform([answer])
    sent_vecs = ohe_vec.transform(sentences)
    sims = cos_sim(ans_vec, sent_vecs)[0]
    return [s for _, s in sorted(zip(sims, sentences), reverse=True)[:top_k]]


def apply_wh_template(sentence: str) -> list:
    words = sentence.split()
    verb  = next((w for w in words if w in COMMON_VERBS), "is")
    rest  = " ".join(w for w in words if w != verb)[:60]
    return [f"{wh} {verb} {rest}?" for wh in WH_TEMPLATES]


def generate_questions(article, answer, ohe_vec, top_k=3):
    candidates = extract_candidate_sentences(article, answer, ohe_vec, top_k=5)
    questions  = []
    for sent in candidates:
        questions.extend(apply_wh_template(sent))
    if not questions:
        return ["What is the main idea of the passage?"]
    return sorted(set(questions), key=lambda q: -len(q))[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_training():
    print("Loading processed data …")
    X_bin, y_bin, bin_dfs, ohe, labels4 = load_processed()

    X_train, X_val, X_test = X_bin
    y_train, y_val, y_test = y_bin
    train_bin, val_bin, test_bin = bin_dfs
    X_tr_ohe, X_va_ohe, X_te_ohe = ohe
    y4_train, y4_val, y4_test, le = labels4

    # TF-IDF-only block for Naive Bayes (no dense features, must be non-negative)
    with open(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vec = pickle.load(f)

    def tfidf_only(bin_df):
        def concat(r):
            return r["article"] + " " + r["question"] + " " + r["option"]
        return tfidf_vec.transform(bin_df.apply(concat, axis=1))

    X_tr_tfidf = tfidf_only(train_bin)
    X_va_tfidf = tfidf_only(val_bin)
    X_te_tfidf = tfidf_only(test_bin)

    results = {}

    # ── Logistic Regression (tuned) ───────────────────────────────────────────
    lr = tune_logistic_regression(X_train, y_train, X_val, y_val, y4_val)
    evaluate_binary(lr, X_val, y_val, "val")
    val_mcq = evaluate_mcq(lr, X_val, y4_val, "val")
    results["LogisticRegression"] = {"val_mcq": val_mcq}
    save_model(lr, "logistic_regression")

    # ── Linear SVM (calibrated, tuned) ───────────────────────────────────────
    svm = tune_svm(X_train, y_train, X_val, y_val, y4_val)
    evaluate_binary(svm, X_val, y_val, "val")
    val_mcq = evaluate_mcq(svm, X_val, y4_val, "val")
    results["LinearSVM"] = {"val_mcq": val_mcq}
    save_model(svm, "linear_svm")

    # ── Complement Naive Bayes (TF-IDF only) ─────────────────────────────────
    nb = train_naive_bayes(X_tr_tfidf, y_train)
    evaluate_binary(nb, X_va_tfidf, y_val, "val")
    val_mcq = evaluate_mcq(nb, X_va_tfidf, y4_val, "val")
    results["ComplementNB"] = {"val_mcq": val_mcq}
    save_model(nb, "naive_bayes")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = train_random_forest(X_train, y_train, n_estimators=200)
    evaluate_binary(rf, X_val, y_val, "val")
    val_mcq = evaluate_mcq(rf, X_val, y4_val, "val")
    results["RandomForest"] = {"val_mcq": val_mcq}
    save_model(rf, "random_forest")

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    if HAS_XGB:
        xgb = train_xgboost(X_train, y_train)
        evaluate_binary(xgb, X_val, y_val, "val")
        val_mcq = evaluate_mcq(xgb, X_val, y4_val, "val")
        results["XGBoost"] = {"val_mcq": val_mcq}
        save_model(xgb, "xgboost")

    # ── Soft-Vote Ensemble ────────────────────────────────────────────────────
    ensemble = train_soft_vote_ensemble(X_train, y_train, lr, rf)
    evaluate_binary(ensemble, X_val, y_val, "val")
    val_mcq = evaluate_mcq(ensemble, X_val, y4_val, "val")
    results["SoftVoteEnsemble"] = {"val_mcq": val_mcq}
    save_model(ensemble, "soft_vote_ensemble")

    # ── Unsupervised / Semi-Supervised ────────────────────────────────────────
    km, km_metrics = train_and_evaluate_kmeans(
        X_tr_ohe, y4_train, X_va_ohe, y4_val)
    save_model(km, "kmeans")

    ls, ls_metrics = train_and_evaluate_label_spreading(
        X_tr_ohe, y4_train, X_va_ohe, y4_val)
    save_model(ls, "label_spreading")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n========== TEST SET RESULTS ==========")
    best_model = None
    best_val   = 0.0

    for name, model, X_t, use_p in [
        ("LogisticRegression", lr,       X_test,    True),
        ("LinearSVM",          svm,      X_test,    True),
        ("ComplementNB",       nb,       X_te_tfidf, True),
        ("RandomForest",       rf,       X_test,    True),
        ("SoftVoteEnsemble",   ensemble, X_test,    True),
    ]:
        test_mcq = evaluate_mcq(model, X_t, y4_test, f"test/{name}", use_proba=use_p)
        results[name]["test_mcq"] = test_mcq
        if results[name]["val_mcq"] > best_val:
            best_val   = results[name]["val_mcq"]
            best_model = (name, model)

    if HAS_XGB:
        test_mcq = evaluate_mcq(xgb, X_test, y4_test, "test/XGBoost")
        results["XGBoost"]["test_mcq"] = test_mcq

    # ── Confusion matrix for best model ──────────────────────────────────────
    if best_model:
        name, model = best_model
        print(f"\nConfusion matrix — {name} (test, MCQ predictions):")
        n_q = X_test.shape[0] // 4
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1].reshape(n_q, 4)
        else:
            scores = model.decision_function(X_test).reshape(n_q, 4)
        preds = np.argmax(scores, axis=1)
        cm = confusion_matrix(y4_test, preds)
        print(cm)
        print(classification_report(y4_test, preds,
                                    target_names=["A", "B", "C", "D"]))

    # ── Summary table ─────────────────────────────────────────────────────────
    print_results_table(results)

    print("\nUnsupervised summary:")
    print(f"  K-Means  silhouette={km_metrics['silhouette']:.4f}"
          f"  purity={km_metrics['purity']:.4f}")
    print(f"  LabelSpreading  acc={ls_metrics['acc']:.4f}"
          f"  f1={ls_metrics['f1']:.4f}")

    print("\nModel A training complete.")
    return results


if __name__ == "__main__":
    run_training()
