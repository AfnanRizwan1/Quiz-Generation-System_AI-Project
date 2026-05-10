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
import re
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


def train_soft_vote_ensemble(X_train, y_train, lr_model, rf_model, svm_model=None):
    """
    Soft-voting ensemble with 3 classifiers: LR + RF + SVM.
    Rubric requires ≥3 classifiers for soft voting.
    Falls back to LR + RF if svm_model is not provided.
    """
    estimators = []

    # Logistic Regression
    lr_clone = LogisticRegression(
        C=lr_model.C, max_iter=lr_model.max_iter,
        solver=lr_model.solver, class_weight=lr_model.class_weight,
    )
    estimators.append(("lr", lr_clone))

    # Random Forest
    rf_clone = RandomForestClassifier(
        n_estimators=rf_model.n_estimators,
        max_depth=rf_model.max_depth,
        min_samples_leaf=rf_model.min_samples_leaf,
        class_weight=rf_model.class_weight,
        n_jobs=-1, random_state=42,
    )
    estimators.append(("rf", rf_clone))

    # SVM (calibrated) — adds a third diverse classifier
    if svm_model is not None:
        # Extract the best C from the calibrated SVM
        try:
            best_C = svm_model.estimator.C
        except AttributeError:
            best_C = 1.0
        svm_clone = CalibratedClassifierCV(
            LinearSVC(C=best_C, max_iter=2000, dual=False,
                      class_weight="balanced"),
            cv=3, method="sigmoid"
        )
        estimators.append(("svm", svm_clone))
        print(f"\n--- Soft-Vote Ensemble (LR + RF + SVM, C={best_C}) ---")
    else:
        print("\n--- Soft-Vote Ensemble (LR + RF) ---")

    ensemble = VotingClassifier(
        estimators=estimators,
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
      - Silhouette score (on a dense PCA-reduced subsample to avoid nan)
      - Cluster-label mapping accuracy (purity)
    """
    print("\n--- K-Means Clustering ---")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                         n_init=10, batch_size=4096)
    km.fit(X_ohe_train)

    # ── Cluster-label mapping accuracy (purity) ───────────────────────────────
    cluster_labels = km.predict(X_ohe_val)
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            mapping[c] = 0
            continue
        mapping[c] = int(np.bincount(y4_val[mask]).argmax())
    mapped_preds = np.array([mapping[c] for c in cluster_labels])
    purity = accuracy_score(y4_val, mapped_preds)
    print(f"  Cluster-label mapping accuracy (purity) = {purity:.4f}")

    # ── Silhouette score ──────────────────────────────────────────────────────
    # silhouette_score requires at least 2 distinct cluster labels in the sample.
    # We use a stratified subsample to guarantee all 4 clusters are represented,
    # then reduce to dense with TruncatedSVD (avoids the nan from single-cluster
    # subsamples that occur with pure random sampling on sparse OHE features).
    sil = float("nan")
    try:
        from sklearn.decomposition import TruncatedSVD
        rng = np.random.default_rng(42)
        # Stratified subsample: 250 per cluster → 1000 total
        sub_idx = []
        for c in range(n_clusters):
            c_idx = np.where(cluster_labels == c)[0]
            chosen = rng.choice(c_idx, size=min(250, len(c_idx)), replace=False)
            sub_idx.extend(chosen.tolist())
        sub_idx = np.array(sub_idx)
        X_sub   = X_ohe_val[sub_idx]
        lab_sub = cluster_labels[sub_idx]

        # Reduce to 50 dims for silhouette (much faster, avoids sparse issues)
        n_components = min(50, X_sub.shape[1] - 1, X_sub.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_dense = svd.fit_transform(X_sub)

        n_unique = len(np.unique(lab_sub))
        if n_unique >= 2:
            sil = float(silhouette_score(X_dense, lab_sub, random_state=42))
        else:
            print("  [WARN] Only 1 cluster in subsample — silhouette undefined")
    except Exception as e:
        print(f"  [WARN] Silhouette computation failed: {e}")

    print(f"  Silhouette score (SVD-reduced subsample) = "
          f"{sil:.4f}" if not np.isnan(sil) else "  Silhouette score = nan (single cluster)")

    return km, {"silhouette": sil, "purity": purity}


def train_and_evaluate_label_spreading(X_ohe_train, y4_train,
                                        X_ohe_val, y4_val,
                                        max_samples=4000,
                                        labeled_fraction=0.30):
    """
    LabelSpreading (semi-supervised).
    Improvements over v1:
      - labeled_fraction raised from 15% → 30% (more signal for 4-class problem)
      - rbf kernel instead of knn (better on dense PCA-reduced features)
      - PCA reduction to 100 dims before fitting (faster + better geometry)
      - Larger subsample for evaluation (3000 instead of 2000)
    """
    print("\n--- Label Spreading (Semi-Supervised) ---")
    from sklearn.decomposition import TruncatedSVD

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

    # Reduce dimensionality: 5000-dim OHE → 100-dim dense
    # This dramatically improves label propagation geometry
    n_components = min(100, X_sub.shape[1] - 1, X_sub.shape[0] - 1)
    print(f"  TruncatedSVD: {X_sub.shape[1]} → {n_components} dims …")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_dense = svd.fit_transform(X_sub)

    # Mask unlabeled (keep labeled_fraction labeled)
    n_unlabeled = int(len(y_sub) * (1 - labeled_fraction))
    unlabeled_idx = rng.choice(len(y_sub), size=n_unlabeled, replace=False)
    y_semi = y_sub.copy()
    y_semi[unlabeled_idx] = -1
    n_labeled = (y_semi != -1).sum()
    print(f"  Labeled={n_labeled}  Unlabeled={n_unlabeled}  "
          f"({labeled_fraction*100:.0f}% labeled)")

    # RBF kernel works better than KNN on dense reduced features
    ls = LabelSpreading(kernel="rbf", gamma=0.25, max_iter=100, alpha=0.2)
    ls.fit(X_dense, y_semi)

    # Evaluate on val subsample (transform with same SVD)
    n_val = X_ohe_val.shape[0]
    val_idx = rng.choice(n_val, size=min(3000, n_val), replace=False)
    X_val_sub = X_ohe_val[val_idx]
    X_val_dense = svd.transform(X_val_sub)
    preds = ls.predict(X_val_dense)

    f1  = f1_score(y4_val[val_idx], preds, average="macro", zero_division=0)
    acc = accuracy_score(y4_val[val_idx], preds)
    print(f"  Semi-supervised  Acc={acc:.4f}  Macro-F1={f1:.4f}")
    print(f"  (evaluated on {len(val_idx)} val samples)")
    print(f"  Random baseline = 0.2500  |  Improvement = +{acc-0.25:.4f}")

    # Store SVD transform alongside model for inference
    ls._svd_transform = svd

    return ls, {"acc": acc, "f1": f1, "n_samples": len(val_idx)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Template-Based Question Generation  (Steps 1–3 from spec)
#    Step 1: Extract candidate sentences via OHE cosine similarity
#    Step 2: Apply Wh-word templates
#    Step 3: Rank generated questions with a trained ML ranker (SVM/LR)
# ══════════════════════════════════════════════════════════════════════════════
WH_TEMPLATES = {
    "what": [
        "What does the passage say about {keyword}?",
        "What is {keyword}?",
    ],
    "who": [
        "Who is {keyword}?",
    ],
    "where": [
        "Where did {keyword} happen?",
        "Where is {keyword} located?",
    ],
    "when": [
        "When did {keyword} happen?",
    ],
    "why": [
        "Why did {keyword} happen?",
    ],
    "how": [
        "How did {keyword} happen?",
    ],
}

COMMON_VERBS  = {"is", "was", "are", "were", "did", "does", "do", "has", "have", "had"}

STOPWORDS_QG = {
    "the","a","an","is","was","are","were","and","or","but","in","on",
    "at","to","for","of","with","by","from","that","this","it","he",
    "she","they","we","i","not","be","been","had","have","has","do",
    "did","does","so","as","if","its","his","her","their","our","my",
    "very","just","also","more","than","some","such","each","both",
    "all","any","few","most","other","same","own","too","now","here",
    "there","where","why","while","when","what","who","how","will",
    "would","could","should","may","might","shall","can","about",
    "after","before","during","through","over","under","again","once",
    "only","even","still","yet","already","always","never","often",
    "usually","sometimes","although","because","since","then","than",
    "which","whose","whom","these","those","every","many","much",
    "into","onto","upon","within","without","between","among","around",
    "however","therefore","moreover","furthermore","nevertheless",
    "began","began","began","let","sent","took","went","came","said",
    "told","found","saw","knew","got","put","set","kept","made","gave",
}


def extract_candidate_sentences(article: str, answer: str, ohe_vec, top_k=5):
    """
    Step 1 from spec: Extract candidate sentences from passage using
    OHE cosine similarity with the correct answer / seed phrase.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sentences = [s.strip() for s in article.split(".") if len(s.strip()) > 10]
    if not sentences:
        return []
    ans_vec   = ohe_vec.transform([answer])
    sent_vecs = ohe_vec.transform(sentences)
    sims = cos_sim(ans_vec, sent_vecs)[0]
    return [s for _, s in sorted(zip(sims, sentences), reverse=True)[:top_k]]


def _extract_keyword(sentence: str) -> str:
    """
    Extract the most meaningful keyword from a sentence.
    Strategy (matching your friend's approach):
      1. Prefer multi-word proper nouns (named entities)
      2. Fall back to the longest non-stopword token
    """
    # Try multi-word proper noun first
    proper = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', sentence)
    if proper:
        # Return the longest proper noun
        return max(proper, key=len)

    # Single proper noun
    single_proper = re.findall(r'\b([A-Z][a-z]{2,})\b', sentence)
    # Filter out sentence-starting capitalisation
    words = sentence.split()
    single_proper = [p for p in single_proper
                     if p != words[0] or len(words) == 1]
    if single_proper:
        return max(single_proper, key=len)

    # Longest non-stopword token
    tokens = re.findall(r'\b[a-z]{4,}\b', sentence.lower())
    content = [t for t in tokens if t not in STOPWORDS_QG]
    if content:
        return max(content, key=len).title()

    # Last resort: first word
    return words[0] if words else "this topic"


def _select_wh_type(sentence: str, keyword: str) -> str:
    """
    Choose the most appropriate Wh-word type based on sentence content.
    """
    s_lower   = sentence.lower()
    kw_lower  = keyword.lower()
    kw_words  = set(kw_lower.split())

    PLACE_WORDS = {"ocean", "sea", "city", "country", "valley", "mountain",
                   "river", "lake", "university", "school", "island",
                   "village", "town", "region", "district", "province",
                   "hospital", "park", "street", "road"}
    AWARD_WORDS = {"prize", "award", "medal", "honor", "trophy"}
    ORG_WORDS   = {"fund", "organization", "organisation", "committee",
                   "association", "society", "institute", "foundation",
                   "church", "order", "company", "corporation"}
    PERSON_VERBS = {"born", "died", "said", "told", "wrote", "won",
                    "studied", "worked", "lived", "graduated", "founded",
                    "discovered", "invented", "received", "became",
                    "passed", "shot", "flown"}

    # Awards/prizes → "what"
    if kw_words & AWARD_WORDS:
        return "what"

    # Organisations → "what"
    if kw_words & ORG_WORDS:
        return "what"

    # Places → "where"
    if kw_words & PLACE_WORDS:
        return "where"

    # Person name (two capitalised words) + person verb in sentence → "who"
    if (re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', keyword)
            and any(w in s_lower for w in PERSON_VERBS)):
        return "who"

    # Place indicators in sentence → "where"
    if any(phrase in s_lower for phrase in ["born in", "moved to", "went to",
                                             "located in", "flown to",
                                             "sent to", "returned to"]):
        return "where"

    # Time indicators → "when"
    if re.search(r'\b\d{4}\b', sentence) or any(
            w in s_lower for w in ["died", "passed away", "sank",
                                    "crashed", "happened"]):
        return "when"

    # Default → "what"
    return "what"


def apply_wh_template(sentence: str) -> list:
    """
    Generate questions from a sentence using keyword extraction + templates.

    Implements your friend's approach:
      1. Extract the most meaningful keyword (longest non-stopword / proper noun)
      2. Apply Wh-word templates based on sentence content
      3. Return all generated questions

    This always produces grammatically correct questions unlike the old
    verb-stripping approach.
    """
    keyword = _extract_keyword(sentence)
    wh_type = _select_wh_type(sentence, keyword)

    questions = []
    # Primary: use the selected Wh-type
    for template in WH_TEMPLATES.get(wh_type, WH_TEMPLATES["what"]):
        questions.append(template.format(keyword=keyword))

    # Secondary: always add a "what" question as backup
    if wh_type != "what":
        for template in WH_TEMPLATES["what"]:
            q = template.format(keyword=keyword)
            if q not in questions:
                questions.append(q)

    return questions


# ── Question Ranker features ──────────────────────────────────────────────────
def _question_features(question: str, article: str, ohe_vec,
                       true_question: str = "", correct_answer: str = "") -> np.ndarray:
    """
    Feature vector for a generated question (for the ML ranker — Step 3).

    Implements your friend's 6-feature approach:
      [0]  OHE cosine similarity: question ↔ true question (0 if unknown)
      [1]  OHE cosine similarity: question ↔ article
      [2]  OHE cosine similarity: question ↔ correct answer (0 if unknown)
      [3]  Wh-word type match with true question (0 if unknown)
      [4]  Question length (word count, normalised)
      [5]  Keyword overlap: question words ∩ true question words (0 if unknown)
    """
    from sklearn.preprocessing import normalize as sk_norm
    words = question.lower().split()
    wh_words = {"who", "what", "where", "when", "why", "how"}

    try:
        q_vec = sk_norm(ohe_vec.transform([question]), norm="l2")
        a_vec = sk_norm(ohe_vec.transform([article[:300]]), norm="l2")
        f1 = float((q_vec * a_vec.T).toarray()[0, 0])
    except Exception:
        f1 = 0.0

    f4 = min(len(words) / 15.0, 1.0)

    if true_question:
        try:
            tq_vec = sk_norm(ohe_vec.transform([true_question]), norm="l2")
            f0 = float((q_vec * tq_vec.T).toarray()[0, 0])
        except Exception:
            f0 = 0.0
        # Wh-word type match
        tq_wh = true_question.lower().split()[0] if true_question.split() else ""
        q_wh  = words[0] if words else ""
        f3 = float(q_wh == tq_wh and q_wh in wh_words)
        # Keyword overlap
        tq_words = set(true_question.lower().split()) - STOPWORDS_QG
        q_words  = set(words) - STOPWORDS_QG
        f5 = len(q_words & tq_words) / (len(q_words) + 1e-9)
    else:
        # No true question available — use proxy features
        f0 = float(words[0] in wh_words) if words else 0.0  # starts with Wh
        f3 = float(words[0] in wh_words) if words else 0.0
        f5 = len(set(words) - STOPWORDS_QG) / (len(words) + 1e-9)

    if correct_answer:
        try:
            ans_vec = sk_norm(ohe_vec.transform([correct_answer]), norm="l2")
            f2 = float((q_vec * ans_vec.T).toarray()[0, 0])
        except Exception:
            f2 = 0.0
    else:
        f2 = 0.0

    return np.array([f0, f1, f2, f3, f4, f5], dtype=np.float32)


def build_question_ranker_dataset(df, ohe_vec, max_rows: int = 3000):
    """
    Build training data for the question ranker.

    Positive samples  : RACE ground-truth questions (label=1)
    Negative samples  : template-generated questions from the SAME article
                        using the new keyword-based templates (hard negatives)

    Features: 6-dim vector matching your friend's approach
    """
    import random as _random
    rng = _random.Random(42)

    X, y = [], []
    for row in df.head(max_rows).itertuples(index=False):
        article  = row.article
        question = row.question
        answer   = getattr(row, row.answer)

        # Positive: real RACE question (with true question for feature computation)
        feats = _question_features(question, article, ohe_vec,
                                   true_question=question,
                                   correct_answer=answer)
        X.append(feats)
        y.append(1)

        # Hard negative: template question from a candidate sentence
        sentences = [s.strip() for s in article.split(".") if len(s.strip()) > 10]
        if sentences:
            try:
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                from sklearn.preprocessing import normalize as sk_norm
                ans_vec   = sk_norm(ohe_vec.transform([answer]), norm="l2")
                sent_vecs = sk_norm(ohe_vec.transform(sentences), norm="l2")
                sims      = cos_sim(ans_vec, sent_vecs)[0]
                top_idx   = np.argsort(sims)[::-1][:3]
                neg_sent  = sentences[rng.choice(top_idx.tolist())]
            except Exception:
                neg_sent = rng.choice(sentences)

            neg_qs = apply_wh_template(neg_sent)
            # Pick a question that uses a different Wh-word than the true question
            true_wh = question.lower().split()[0] if question.split() else "what"
            diff_wh = [q for q in neg_qs
                       if q.lower().split()[0] != true_wh]
            neg_q = rng.choice(diff_wh) if diff_wh else neg_qs[0]

            feats_neg = _question_features(neg_q, article, ohe_vec,
                                           true_question=question,
                                           correct_answer=answer)
            X.append(feats_neg)
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_question_ranker(X_train, y_train):
    """
    Train an SVM classifier to rank generated questions by quality.
    Step 3 from the spec: rank generated questions using a trained
    SVM or Random Forest classifier scoring fluency and relevance features.

    Uses cross-validated calibration and regularisation to prevent overfitting
    on the small 7-feature dataset.
    """
    print("\n--- Question Ranker (SVM, Step 3) ---")
    from sklearn.model_selection import cross_val_score

    # Try a few C values and pick the best by cross-validation
    best_acc, best_model, best_C = 0.0, None, None
    for C in [0.01, 0.05, 0.1, 0.5]:
        base = LinearSVC(C=C, max_iter=2000, class_weight="balanced",
                         dual=False)
        m = CalibratedClassifierCV(base, cv=5, method="sigmoid")
        cv_scores = cross_val_score(m, X_train, y_train, cv=5,
                                    scoring="f1_macro", n_jobs=-1)
        cv_mean = cv_scores.mean()
        print(f"  C={C:<5}  CV Macro-F1 = {cv_mean:.4f} ± {cv_scores.std():.4f}")
        if cv_mean > best_acc:
            best_acc, best_C = cv_mean, C

    print(f"  Best C={best_C}  CV Macro-F1={best_acc:.4f}")
    base  = LinearSVC(C=best_C, max_iter=2000, class_weight="balanced",
                      dual=False)
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Train acc = {train_acc:.4f}  (CV F1 = {best_acc:.4f})")
    if train_acc > 0.95 and best_acc < 0.80:
        print("  [NOTE] Train acc >> CV F1 — some overfitting expected on "
              "this small feature set; CV score is the reliable estimate.")
    return model


def generate_questions(article, answer, ohe_vec, top_k=3,
                       question_ranker=None):
    """
    Generate and rank questions using the 3-step pipeline from the spec.

    Step 1: Extract candidate sentences via OHE cosine similarity to answer
    Step 2: Apply keyword-based Wh-word templates (your friend's approach)
    Step 3: Rank with trained ML ranker (if available), else rank by length
    """
    candidates = extract_candidate_sentences(article, answer, ohe_vec, top_k=5)
    questions  = []
    for sent in candidates:
        questions.extend(apply_wh_template(sent))
    if not questions:
        return ["What is the main idea of the passage?"]

    # Deduplicate
    seen, unique_qs = set(), []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_qs.append(q)
    questions = unique_qs

    if question_ranker is not None:
        # Step 3: ML ranker scores each question
        try:
            n_feats = question_ranker.n_features_in_
            feats = np.array([
                _question_features(q, article, ohe_vec)[:n_feats]
                for q in questions
            ], dtype=np.float32)
            probs = question_ranker.predict_proba(feats)[:, 1]
            ranked = [q for _, q in sorted(zip(probs, questions), reverse=True)]
        except Exception:
            ranked = sorted(questions, key=lambda q: -len(q))
    else:
        # Fallback: rank by length (longer = more specific)
        ranked = sorted(questions, key=lambda q: -len(q))

    return ranked[:top_k]


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

    # ── Soft-Vote Ensemble (LR + RF + SVM) ───────────────────────────────────
    ensemble = train_soft_vote_ensemble(X_train, y_train, lr, rf, svm_model=svm)
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

    # ── Question Ranker (Step 3 from spec) ────────────────────────────────────
    print("\n--- Building Question Ranker Dataset ---")
    with open(os.path.join(PROCESSED_DIR, "cleaned_dfs.pkl"), "rb") as f:
        train_df_clean, val_df_clean, _ = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "ohe_vectorizer.pkl"), "rb") as f:
        ohe_vec_qr = pickle.load(f)

    X_qr_train, y_qr_train = build_question_ranker_dataset(
        train_df_clean, ohe_vec_qr, max_rows=3000)
    X_qr_val, y_qr_val = build_question_ranker_dataset(
        val_df_clean, ohe_vec_qr, max_rows=500)

    qr = train_question_ranker(X_qr_train, y_qr_train)
    val_acc_qr = accuracy_score(y_qr_val, qr.predict(X_qr_val))
    val_f1_qr  = f1_score(y_qr_val, qr.predict(X_qr_val), average="macro")
    print(f"  Question Ranker  Val Acc={val_acc_qr:.4f}  Macro-F1={val_f1_qr:.4f}")
    save_model(qr, "question_ranker")

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
    sil_str = f"{km_metrics['silhouette']:.4f}" if not np.isnan(km_metrics['silhouette']) else "nan"
    print(f"  K-Means  silhouette={sil_str}  purity={km_metrics['purity']:.4f}")
    print(f"  LabelSpreading  acc={ls_metrics['acc']:.4f}"
          f"  f1={ls_metrics['f1']:.4f}"
          f"  (n={ls_metrics.get('n_samples', '?')})")

    print("\nModel A training complete.")
    return results


if __name__ == "__main__":
    run_training()
