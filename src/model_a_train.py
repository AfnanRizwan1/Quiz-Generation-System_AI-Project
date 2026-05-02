"""
model_a_train.py
----------------
Model A — Question & Answer Generator / Verifier

Tasks:
  1. Answer Verification  — classify which option (A/B/C/D) is correct
  2. Question Generation  — template-based Wh-word generation + ML ranker
  3. Unsupervised/Semi-Supervised — K-Means, GMM, Label Propagation
  4. Ensemble             — soft-vote / stacking over base classifiers
"""

import os
import pickle
import numpy as np
from scipy.sparse import hstack

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models", "model_a", "traditional")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_processed():
    def lp(f):
        with open(os.path.join(PROCESSED_DIR, f), "rb") as fh:
            return pickle.load(fh)

    X_ohe   = lp("ohe_matrices.pkl")          # (X_train, X_val, X_test)
    lex     = lp("lexical_features.pkl")       # (lex_train, lex_val, lex_test)
    labels  = lp("labels.pkl")                # (y_train, y_val, y_test, le)
    return X_ohe, lex, labels


def combine_features(X_ohe, lex):
    """Stack sparse OHE matrix with dense lexical features."""
    from scipy.sparse import csr_matrix
    return hstack([X_ohe, csr_matrix(lex)])


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Saved model → {path}")


def evaluate(model, X, y, split_name="val"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro")
    print(f"\n[{split_name}]  Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    print(classification_report(y, preds, target_names=["A", "B", "C", "D"]))
    return acc, f1


# ── 1. Supervised Models ──────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    print("\n--- Logistic Regression ---")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               multi_class="multinomial", n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    print("\n--- Linear SVM ---")
    model = LinearSVC(max_iter=2000, C=1.0)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_ohe_train, y_train):
    """Naive Bayes requires non-negative features — use raw OHE (no lexical)."""
    print("\n--- Naive Bayes ---")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_ohe_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("\n--- Random Forest ---")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── 2. Unsupervised / Semi-Supervised ─────────────────────────────────────────
def train_kmeans(X_train, n_clusters=4):
    """K-Means clustering — groups question-answer pairs by OHE similarity."""
    print("\n--- K-Means Clustering ---")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_train)
    return km


def train_gmm(X_train, n_components=4):
    """Gaussian Mixture Model — soft cluster membership."""
    print("\n--- Gaussian Mixture Model ---")
    # GMM needs dense input
    X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_dense)
    return gmm


def train_label_propagation(X_train, y_train, labeled_fraction=0.1):
    """
    Label Propagation (semi-supervised).
    Masks (1 - labeled_fraction) of training labels as -1 (unlabeled).
    """
    print("\n--- Label Propagation (Semi-Supervised) ---")
    rng = np.random.default_rng(42)
    y_semi = y_train.copy()
    n_unlabeled = int(len(y_train) * (1 - labeled_fraction))
    unlabeled_idx = rng.choice(len(y_train), size=n_unlabeled, replace=False)
    y_semi[unlabeled_idx] = -1

    X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
    lp.fit(X_dense, y_semi)
    return lp


# ── 3. Ensemble ───────────────────────────────────────────────────────────────
def train_soft_vote_ensemble(X_train, y_train):
    """Soft-voting ensemble: LR + SVM (with probability) + RF."""
    print("\n--- Soft-Vote Ensemble ---")
    lr  = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="multinomial", n_jobs=-1)
    svm = SVC(kernel="linear", probability=True, C=1.0)
    rf  = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def train_stacking_ensemble(X_train, y_train):
    """Stacking: LR + NB as base; RF as meta-classifier."""
    print("\n--- Stacking Ensemble ---")
    base = [
        ("lr",  LogisticRegression(max_iter=1000, solver="lbfgs",
                                   multi_class="multinomial", n_jobs=-1)),
        ("svm", LinearSVC(max_iter=2000)),
    ]
    meta = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    stack = StackingClassifier(estimators=base, final_estimator=meta,
                               cv=3, n_jobs=-1)
    stack.fit(X_train, y_train)
    return stack


# ── 4. Template-Based Question Generation ────────────────────────────────────
WH_TEMPLATES = {
    "who":   "Who {verb} {rest}?",
    "what":  "What {verb} {rest}?",
    "where": "Where {verb} {rest}?",
    "when":  "When {verb} {rest}?",
    "why":   "Why {verb} {rest}?",
}

COMMON_VERBS = {"is", "was", "are", "were", "did", "does", "do", "has", "have", "had"}


def extract_candidate_sentences(article: str, answer: str, ohe_vec, top_k: int = 5):
    """
    Extract top-k candidate sentences from article by OHE cosine similarity
    to the correct answer string.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = [s.strip() for s in article.split(".") if len(s.strip()) > 10]
    if not sentences:
        return []

    ans_vec  = ohe_vec.transform([answer])
    sent_vecs = ohe_vec.transform(sentences)
    sims = cosine_similarity(ans_vec, sent_vecs)[0]
    ranked = sorted(zip(sims, sentences), reverse=True)
    return [s for _, s in ranked[:top_k]]


def apply_wh_template(sentence: str) -> list:
    """Apply Wh-word templates to a candidate sentence → list of question strings."""
    words = sentence.split()
    questions = []
    for wh, template in WH_TEMPLATES.items():
        # Find first verb-like word
        verb = next((w for w in words if w in COMMON_VERBS), "is")
        rest_words = [w for w in words if w != verb]
        rest = " ".join(rest_words[:8])  # truncate for readability
        q = f"{wh.capitalize()} {verb} {rest}?"
        questions.append(q)
    return questions


def generate_questions(article: str, answer: str, ohe_vec, ranker_model=None, top_k: int = 3):
    """
    Full generation pipeline:
      1. Extract candidate sentences
      2. Apply Wh-word templates
      3. Rank with ML ranker (if provided) or return top-k by order
    """
    candidates = extract_candidate_sentences(article, answer, ohe_vec, top_k=5)
    all_questions = []
    for sent in candidates:
        all_questions.extend(apply_wh_template(sent))

    if not all_questions:
        return ["What is the main idea of the passage?"]

    # Simple heuristic ranking: prefer longer, more specific questions
    all_questions = sorted(set(all_questions), key=lambda q: -len(q))
    return all_questions[:top_k]


# ── Main ──────────────────────────────────────────────────────────────────────
def run_training():
    print("Loading processed data...")
    X_ohe, lex, labels = load_processed()
    X_train_ohe, X_val_ohe, X_test_ohe = X_ohe
    lex_train, lex_val, lex_test = lex
    y_train, y_val, y_test, le = labels

    # Combined features (OHE + lexical)
    X_train = combine_features(X_train_ohe, lex_train)
    X_val   = combine_features(X_val_ohe,   lex_val)
    X_test  = combine_features(X_test_ohe,  lex_test)

    results = {}

    # ── Supervised ────────────────────────────────────────────────────────────
    lr = train_logistic_regression(X_train, y_train)
    acc, f1 = evaluate(lr, X_val, y_val, "val")
    results["LogisticRegression"] = {"acc": acc, "f1": f1}
    save_model(lr, "logistic_regression")

    svm = train_svm(X_train, y_train)
    acc, f1 = evaluate(svm, X_val, y_val, "val")
    results["LinearSVM"] = {"acc": acc, "f1": f1}
    save_model(svm, "linear_svm")

    nb = train_naive_bayes(X_train_ohe, y_train)
    acc, f1 = evaluate(nb, X_val_ohe, y_val, "val")
    results["NaiveBayes"] = {"acc": acc, "f1": f1}
    save_model(nb, "naive_bayes")

    rf = train_random_forest(X_train, y_train)
    acc, f1 = evaluate(rf, X_val, y_val, "val")
    results["RandomForest"] = {"acc": acc, "f1": f1}
    save_model(rf, "random_forest")

    # ── Unsupervised / Semi-Supervised ────────────────────────────────────────
    km = train_kmeans(X_train_ohe)
    save_model(km, "kmeans")

    lp_model = train_label_propagation(X_train_ohe, y_train)
    save_model(lp_model, "label_propagation")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    stack = train_stacking_ensemble(X_train, y_train)
    acc, f1 = evaluate(stack, X_val, y_val, "val")
    results["StackingEnsemble"] = {"acc": acc, "f1": f1}
    save_model(stack, "stacking_ensemble")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n========== TEST SET RESULTS ==========")
    for name, model, X_t in [
        ("LogisticRegression", lr,    X_test),
        ("LinearSVM",          svm,   X_test),
        ("NaiveBayes",         nb,    X_test_ohe),
        ("RandomForest",       rf,    X_test),
        ("StackingEnsemble",   stack, X_test),
    ]:
        acc, f1 = evaluate(model, X_t, y_test, f"test/{name}")
        results[name]["test_acc"] = acc
        results[name]["test_f1"]  = f1

    print("\nModel A training complete.")
    return results


if __name__ == "__main__":
    run_training()
