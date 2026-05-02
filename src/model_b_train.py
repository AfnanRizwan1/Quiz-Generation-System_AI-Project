"""
model_b_train.py
----------------
Model B — Distractor & Hint Generator

Tasks:
  1. Distractor Generation  — extract + rank plausible wrong options
  2. Hint Generation        — extractive sentence scoring
"""

import os
import re
import pickle
import numpy as np
import joblib
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models", "model_b", "traditional")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_processed():
    def lp(f):
        with open(os.path.join(PROCESSED_DIR, f), "rb") as fh:
            return pickle.load(fh)
    ohe_vec   = lp("ohe_vectorizer.pkl")
    dfs       = lp("cleaned_dfs.pkl")       # (train_df, val_df, test_df)
    return ohe_vec, dfs


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Saved model → {path}")


# ── 1. Candidate Extraction ───────────────────────────────────────────────────
def extract_candidate_phrases(article: str, top_n: int = 20) -> list:
    """
    Extract candidate distractor phrases from the article using:
      - Simple word tokenisation
      - Frequency-based selection (high-frequency content words)
    No NLP tools required — pure string operations.
    """
    words = re.findall(r"\b[a-z]{3,}\b", article.lower())
    stopwords = {
        "the", "and", "that", "this", "with", "for", "are", "was",
        "were", "has", "have", "had", "not", "but", "from", "they",
        "their", "there", "been", "also", "more", "than", "its",
        "into", "about", "which", "when", "what", "who", "how",
    }
    content_words = [w for w in words if w not in stopwords]
    freq = Counter(content_words)
    return [word for word, _ in freq.most_common(top_n)]


# ── 2. Feature Engineering for Distractor Ranking ────────────────────────────
def distractor_features(candidate: str, correct_answer: str,
                        article: str, ohe_vec) -> np.ndarray:
    """
    Per-candidate feature vector:
      [0] OHE cosine similarity to correct answer
      [1] character-level match score (Jaccard on chars)
      [2] passage frequency (normalised)
      [3] candidate length (chars)
    """
    # OHE cosine similarity
    cand_vec = ohe_vec.transform([candidate])
    ans_vec  = ohe_vec.transform([correct_answer])
    ohe_sim  = cosine_similarity(cand_vec, ans_vec)[0, 0]

    # Character-level Jaccard
    c_chars = set(candidate)
    a_chars = set(correct_answer)
    char_jaccard = len(c_chars & a_chars) / (len(c_chars | a_chars) + 1e-9)

    # Passage frequency (normalised by article length)
    art_words = article.lower().split()
    freq = art_words.count(candidate) / (len(art_words) + 1e-9)

    # Length
    length = len(candidate) / 20.0  # normalise

    return np.array([ohe_sim, char_jaccard, freq, length], dtype=np.float32)


def build_distractor_dataset(df, ohe_vec, max_rows: int = 5000):
    """
    Build a labelled dataset for distractor ranking:
      - Positive (label=1): the three actual wrong options from RACE
      - Negative (label=0): random candidate phrases that are NOT any option
    """
    X, y = [], []
    for _, row in df.head(max_rows).iterrows():
        correct = row[row["answer"]]          # text of correct option
        wrong_opts = [row[o] for o in ["A", "B", "C", "D"] if o != row["answer"]]

        # Positives — actual distractors
        for opt in wrong_opts:
            feats = distractor_features(opt, correct, row["article"], ohe_vec)
            X.append(feats)
            y.append(1)

        # Negatives — random candidates from article
        candidates = extract_candidate_phrases(row["article"], top_n=10)
        all_opts = [row[o] for o in ["A", "B", "C", "D"]]
        negatives = [c for c in candidates if c not in " ".join(all_opts)][:3]
        for neg in negatives:
            feats = distractor_features(neg, correct, row["article"], ohe_vec)
            X.append(feats)
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── 3. Distractor Ranker ──────────────────────────────────────────────────────
def train_distractor_ranker(X_train, y_train):
    print("\n--- Distractor Ranker (Logistic Regression) ---")
    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(X_train, y_train)
    return model


def generate_distractors(article: str, question: str, correct_answer: str,
                         ohe_vec, ranker_model, top_k: int = 3) -> list:
    """
    Full distractor generation pipeline:
      1. Extract candidate phrases from article
      2. Compute features for each candidate
      3. Score with ranker; filter out correct answer
      4. Apply diversity penalty (remove near-duplicates)
      5. Return top-k distractors
    """
    candidates = extract_candidate_phrases(article, top_n=30)
    # Remove candidates that match the correct answer
    candidates = [c for c in candidates if c.lower() != correct_answer.lower()]

    if not candidates:
        return ["Option X", "Option Y", "Option Z"][:top_k]

    scored = []
    for cand in candidates:
        feats = distractor_features(cand, correct_answer, article, ohe_vec)
        score = ranker_model.predict_proba(feats.reshape(1, -1))[0][1]
        scored.append((score, cand))

    scored.sort(reverse=True)

    # Diversity penalty: skip candidates too similar to already-selected ones
    selected = []
    for score, cand in scored:
        if len(selected) >= top_k:
            break
        # Simple diversity check: no prefix overlap > 3 chars
        if all(cand[:3] != s[:3] for s in selected):
            selected.append(cand)

    # Pad if needed
    while len(selected) < top_k:
        selected.append(f"none of the above ({len(selected)})")

    return selected[:top_k]


# ── 4. Hint Generation ────────────────────────────────────────────────────────
def score_sentences_for_hints(article: str, question: str, ohe_vec) -> list:
    """
    Score each sentence in the article by OHE cosine similarity to the question.
    Returns list of (score, sentence) sorted descending.
    """
    sentences = [s.strip() for s in article.split(".") if len(s.strip()) > 15]
    if not sentences:
        return []

    q_vec    = ohe_vec.transform([question])
    sent_vecs = ohe_vec.transform(sentences)
    sims = cosine_similarity(q_vec, sent_vecs)[0]

    ranked = sorted(zip(sims, sentences), reverse=True)
    return ranked


def build_hint_dataset(df, ohe_vec, max_rows: int = 3000):
    """
    Build dataset for ML-scored hint extractor.
    Features per sentence: [keyword_overlap, position_norm, length_norm, ohe_sim]
    Label: 1 if sentence contains the correct answer text, else 0
    """
    X, y = [], []
    for _, row in df.head(max_rows).iterrows():
        sentences = [s.strip() for s in row["article"].split(".") if len(s.strip()) > 15]
        correct_text = row[row["answer"]].lower()
        q_words = set(row["question"].split())

        for i, sent in enumerate(sentences):
            sent_words = set(sent.split())
            kw_overlap = len(q_words & sent_words) / (len(q_words) + 1e-9)
            pos_norm   = i / (len(sentences) + 1e-9)
            len_norm   = len(sent.split()) / 50.0

            s_vec  = ohe_vec.transform([sent])
            q_vec  = ohe_vec.transform([row["question"]])
            ohe_sim = cosine_similarity(q_vec, s_vec)[0, 0]

            feats = [kw_overlap, pos_norm, len_norm, ohe_sim]
            label = 1 if correct_text in sent.lower() else 0
            X.append(feats)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_hint_scorer(X_train, y_train):
    print("\n--- Hint Scorer (Logistic Regression) ---")
    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(X_train, y_train)
    return model


def generate_hints(article: str, question: str, ohe_vec,
                   hint_scorer=None, n_hints: int = 3) -> list:
    """
    Generate graduated hints:
      Hint 1 — most general (lowest similarity)
      Hint 2 — more specific
      Hint 3 — near-explicit (highest similarity)
    """
    ranked = score_sentences_for_hints(article, question, ohe_vec)
    if not ranked:
        return ["Re-read the passage carefully.",
                "Focus on the key events described.",
                "Look for the sentence that directly answers the question."]

    # Take top-n and reverse so hint 1 is least specific
    top = [s for _, s in ranked[:n_hints * 2]]
    hints = top[:n_hints]
    hints.reverse()   # hint[0] = most general, hint[-1] = most specific
    return hints


# ── Main ──────────────────────────────────────────────────────────────────────
def run_training():
    print("Loading processed data...")
    ohe_vec, (train_df, val_df, test_df) = load_processed()

    # ── Distractor Ranker ─────────────────────────────────────────────────────
    print("\nBuilding distractor dataset (train)...")
    X_dist_train, y_dist_train = build_distractor_dataset(train_df, ohe_vec, max_rows=3000)
    X_dist_val,   y_dist_val   = build_distractor_dataset(val_df,   ohe_vec, max_rows=500)

    dist_ranker = train_distractor_ranker(X_dist_train, y_dist_train)
    preds = dist_ranker.predict(X_dist_val)
    print(f"Distractor Ranker  Accuracy={accuracy_score(y_dist_val, preds):.4f}"
          f"  F1={f1_score(y_dist_val, preds):.4f}")
    save_model(dist_ranker, "distractor_ranker")

    # ── Hint Scorer ───────────────────────────────────────────────────────────
    print("\nBuilding hint dataset (train)...")
    X_hint_train, y_hint_train = build_hint_dataset(train_df, ohe_vec, max_rows=2000)
    X_hint_val,   y_hint_val   = build_hint_dataset(val_df,   ohe_vec, max_rows=500)

    hint_scorer = train_hint_scorer(X_hint_train, y_hint_train)
    preds_h = hint_scorer.predict(X_hint_val)
    print(f"Hint Scorer  Accuracy={accuracy_score(y_hint_val, preds_h):.4f}"
          f"  F1={f1_score(y_hint_val, preds_h):.4f}")
    save_model(hint_scorer, "hint_scorer")

    print("\nModel B training complete.")


if __name__ == "__main__":
    run_training()
