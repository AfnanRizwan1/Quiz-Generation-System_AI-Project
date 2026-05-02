"""
inference.py
------------
Unified inference API — loads trained models once and exposes
clean functions for the UI layer.
"""

import os
import pickle
import time
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

from src.preprocessing import clean_text, lexical_features
from src.model_a_train import generate_questions, extract_candidate_sentences
from src.model_b_train import generate_distractors, generate_hints

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_A_DIR   = os.path.join("models", "model_a", "traditional")
MODEL_B_DIR   = os.path.join("models", "model_b", "traditional")

# ── Singleton model cache ─────────────────────────────────────────────────────
_cache = {}


def _load(path):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def _load_pkl(filename):
    key = filename
    if key not in _cache:
        with open(os.path.join(PROCESSED_DIR, filename), "rb") as f:
            _cache[key] = pickle.load(f)
    return _cache[key]


def get_ohe_vec():
    return _load_pkl("ohe_vectorizer.pkl")


def get_label_encoder():
    _, _, _, le = _load_pkl("labels.pkl")
    return le


def get_verifier():
    """Best single verifier — stacking ensemble (falls back to LR)."""
    path = os.path.join(MODEL_A_DIR, "stacking_ensemble.pkl")
    if os.path.exists(path):
        return _load(path), "stacking_ensemble"
    path = os.path.join(MODEL_A_DIR, "logistic_regression.pkl")
    return _load(path), "logistic_regression"


def get_distractor_ranker():
    return _load(os.path.join(MODEL_B_DIR, "distractor_ranker.pkl"))


def get_hint_scorer():
    return _load(os.path.join(MODEL_B_DIR, "hint_scorer.pkl"))


# ── Feature helpers ───────────────────────────────────────────────────────────
def _row_to_features(article: str, question: str, option: str, ohe_vec):
    """Build combined OHE + lexical feature vector for a single (article, q, option) triple."""
    import pandas as pd
    row = {
        "article": clean_text(article),
        "question": clean_text(question),
        "A": clean_text(option),
        "B": "", "C": "", "D": "",
    }
    df_row = pd.DataFrame([row])
    text = " ".join([row["article"], row["question"], row["A"]])
    X_ohe = ohe_vec.transform([text])
    lex   = lexical_features(df_row)
    return hstack([X_ohe, csr_matrix(lex)])


# ── Public API ────────────────────────────────────────────────────────────────
def run_full_pipeline(article: str, existing_question: str = None):
    """
    Given a reading passage:
      1. Generate a question (or use existing_question)
      2. Identify the correct answer span
      3. Generate 3 distractors
      4. Generate 3 graduated hints
      5. Return structured result dict + latency
    """
    t0 = time.time()

    ohe_vec = get_ohe_vec()
    art_clean = clean_text(article)

    # Step 1 — Question generation
    if existing_question:
        question = existing_question
        # Use first sentence as a proxy correct answer for generation
        correct_answer = article.split(".")[0].strip()[:80]
    else:
        candidates = extract_candidate_sentences(art_clean, art_clean[:50], ohe_vec, top_k=3)
        correct_answer = candidates[0] if candidates else article[:80]
        questions = generate_questions(art_clean, correct_answer, ohe_vec)
        question = questions[0] if questions else "What is the main idea of the passage?"

    # Step 2 — Distractors
    dist_ranker = get_distractor_ranker()
    distractors = generate_distractors(art_clean, question, correct_answer,
                                       ohe_vec, dist_ranker, top_k=3)

    # Step 3 — Hints
    hint_scorer = get_hint_scorer()
    hints = generate_hints(art_clean, question, ohe_vec, hint_scorer, n_hints=3)

    latency = round(time.time() - t0, 3)

    return {
        "article":        article,
        "question":       question,
        "correct_answer": correct_answer,
        "distractors":    distractors,
        "hints":          hints,
        "latency_s":      latency,
    }


def verify_answer(article: str, question: str, chosen_option: str) -> dict:
    """
    Given an article, question, and the user's chosen option text,
    predict whether the chosen option is correct.
    Returns {"prediction": "correct"/"incorrect", "confidence": float, "latency_s": float}
    """
    t0 = time.time()
    ohe_vec = get_ohe_vec()
    verifier, model_name = get_verifier()

    X = _row_to_features(article, question, chosen_option, ohe_vec)

    # LinearSVC has no predict_proba
    if hasattr(verifier, "predict_proba"):
        proba = verifier.predict_proba(X)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 1.0

    pred_label = verifier.predict(X)[0]
    le = get_label_encoder()
    pred_letter = le.inverse_transform([pred_label])[0]

    latency = round(time.time() - t0, 3)
    return {
        "predicted_label": pred_letter,
        "confidence":      confidence,
        "model_used":      model_name,
        "latency_s":       latency,
    }
