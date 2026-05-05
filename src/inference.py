"""
inference.py
------------
Unified inference API — loads trained models once and exposes
clean functions for the UI layer.

Two operating modes:
  RACE mode   : article + question + real A/B/C/D options are all known.
                Model A verifies which option is correct.
                Model B generates hints only (distractors already exist).
  Custom mode : only article (+ optional question) provided.
                Model A generates a question.
                Model B generates short-phrase distractors from the article.
                Correct answer is the highest-scoring option from the article.
"""

import os
import re
import pickle
import time
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

from src.preprocessing import clean_text
from src.model_a_train import generate_questions, extract_candidate_sentences
from src.model_b_train import (
    generate_distractors, generate_hints,
    split_sentences, extract_candidate_phrases,
)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_A_DIR   = os.path.join("models", "model_a", "traditional")
MODEL_B_DIR   = os.path.join("models", "model_b", "traditional")

_cache = {}


# ══════════════════════════════════════════════════════════════════════════════
# Artifact loaders (cached singletons)
# ══════════════════════════════════════════════════════════════════════════════
def _load_model(path: str):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def _load_pkl(filename: str):
    if filename not in _cache:
        with open(os.path.join(PROCESSED_DIR, filename), "rb") as f:
            _cache[filename] = pickle.load(f)
    return _cache[filename]


def get_tfidf_vec():
    return _load_pkl("tfidf_vectorizer.pkl")

def get_dense_scaler():
    return _load_pkl("dense_scaler.pkl")

def get_ohe_vec():
    return _load_pkl("ohe_vectorizer.pkl")

def get_label_encoder():
    _, _, _, le = _load_pkl("labels.pkl")
    return le

def get_verifier():
    for name in ["soft_vote_ensemble", "logistic_regression"]:
        path = os.path.join(MODEL_A_DIR, f"{name}.pkl")
        if os.path.exists(path):
            return _load_model(path), name
    raise FileNotFoundError(
        "No Model A verifier found. Run python src/model_a_train.py first.")

def get_distractor_ranker():
    path = os.path.join(MODEL_B_DIR, "distractor_ranker.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "distractor_ranker.pkl not found. Run python src/model_b_train.py first.")
    return _load_model(path)

def get_hint_scorer():
    path = os.path.join(MODEL_B_DIR, "hint_scorer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "hint_scorer.pkl not found. Run python src/model_b_train.py first.")
    return _load_model(path)


# ══════════════════════════════════════════════════════════════════════════════
# Feature builder for Model A inference
# ══════════════════════════════════════════════════════════════════════════════
def _build_binary_features(article: str, question: str, option: str,
                            tfidf_vec, scaler):
    """
    Build TF-IDF + dense feature vector for one (article, question, option).
    Matches the training pipeline exactly.
    Returns a (1, n_features) sparse matrix.
    """
    import pandas as pd
    from src.preprocessing import combine_features, build_dense_features

    bin_row = pd.DataFrame([{
        "article":  clean_text(article),
        "question": clean_text(question),
        "option":   clean_text(option),
        "label":    0,
    }])

    def concat(r):
        return r["article"][:200] + " " + r["question"] + " " + r["option"]

    X_tfidf = tfidf_vec.transform(bin_row.apply(concat, axis=1))
    dense   = build_dense_features(bin_row, tfidf_vec, desc="inference")
    X, _    = combine_features(X_tfidf, dense, scaler=scaler, fit_scaler=False)
    return X


def _score_option(article: str, question: str, option: str,
                  verifier, tfidf_vec, scaler) -> float:
    """Return P(correct) for a single option."""
    X = _build_binary_features(article, question, option, tfidf_vec, scaler)
    if hasattr(verifier, "predict_proba"):
        return float(verifier.predict_proba(X)[0][1])
    return float(verifier.decision_function(X)[0])


def score_options(article: str, question: str, options: dict,
                  verifier, tfidf_vec, scaler) -> dict:
    """
    Score all options with the binary verifier.
    options: {"A": text, "B": text, "C": text, "D": text}
    Returns: {"A": prob, "B": prob, ...}
    """
    return {
        key: _score_option(article, question, text, verifier, tfidf_vec, scaler)
        for key, text in options.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# Answer candidate extraction for custom articles
# ══════════════════════════════════════════════════════════════════════════════
def _extract_answer_candidates(article_clean: str, question_clean: str,
                                tfidf_vec, top_n: int = 20) -> list:
    """
    Extract short, meaningful answer candidates from a custom article.

    Strategy:
      1. Extract phrases (unigrams + bigrams) from the article
      2. Score each by TF-IDF cosine similarity to the question
      3. Return top_n ranked by relevance
      4. Filter to reasonable answer length (2–8 words)

    This produces options that look like real MCQ answers rather than
    random words or full sentences.
    """
    candidates = extract_candidate_phrases(article_clean, top_n=80)

    # Keep only phrases of 1–5 words (realistic MCQ answer length)
    candidates = [c for c in candidates if 1 <= len(c.split()) <= 5]

    if not candidates:
        # Fallback: split article into chunks and use first few words of each
        chunks = split_sentences(article_clean)
        candidates = [" ".join(c.split()[:4]) for c in chunks if c.strip()][:20]

    if not candidates:
        return ["option one", "option two", "option three", "option four"]

    # Rank by cosine similarity to question
    q_vec_n = normalize(tfidf_vec.transform([question_clean]), norm="l2")
    cand_vecs_n = normalize(tfidf_vec.transform(candidates), norm="l2")
    sims = (q_vec_n @ cand_vecs_n.T).toarray().ravel()

    ranked = [candidates[i] for i in np.argsort(sims)[::-1]]
    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# RACE mode pipeline — uses real options
# ══════════════════════════════════════════════════════════════════════════════
def run_race_pipeline(article: str, question: str,
                      options: dict, correct_key: str) -> dict:
    """
    RACE mode: article + question + real options are all known.

    1. Clean inputs
    2. Score all 4 options with Model A verifier
    3. Generate hints with Model B (using the real correct answer text)
    4. Return structured result

    options     : {"A": text, "B": text, "C": text, "D": text}
    correct_key : "A" / "B" / "C" / "D"
    """
    t0 = time.time()

    tfidf_vec = get_tfidf_vec()
    scaler    = get_dense_scaler()
    verifier, model_name = get_verifier()
    hint_scorer = get_hint_scorer()

    art_clean = clean_text(article)
    q_clean   = clean_text(question)
    opts_clean = {k: clean_text(v) for k, v in options.items()}
    correct_text = opts_clean[correct_key]

    # Score all options
    scores = score_options(art_clean, q_clean, opts_clean,
                           verifier, tfidf_vec, scaler)
    predicted_key = max(scores, key=scores.get)

    # Hints using real correct answer
    hints = generate_hints(
        art_clean, q_clean, correct_text,
        tfidf_vec, hint_scorer, n_hints=3)

    latency = round(time.time() - t0, 3)
    return {
        "article":        article,
        "question":       question,
        "options":        options,          # original (uncleaned) for display
        "correct_key":    correct_key,      # ground-truth
        "predicted_key":  predicted_key,    # model prediction
        "scores":         scores,
        "hints":          hints,
        "model_used":     model_name,
        "latency_s":      latency,
        "mode":           "race",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Custom mode pipeline — generates options from scratch
# ══════════════════════════════════════════════════════════════════════════════
def run_full_pipeline(article: str, existing_question: str = None) -> dict:
    """
    Custom mode: only article (+ optional question) provided.

    1. Clean article
    2. Generate or use provided question
    3. Extract short answer candidates ranked by question relevance
    4. Score candidates with Model A — pick highest as correct answer
    5. Generate 3 distractors from remaining candidates (Model B)
    6. Generate hints (Model B)
    7. Return structured result

    The correct answer and distractors are all short phrases (1–5 words),
    matching the style of real MCQ options.
    """
    t0 = time.time()

    tfidf_vec   = get_tfidf_vec()
    scaler      = get_dense_scaler()
    ohe_vec     = get_ohe_vec()
    verifier, model_name = get_verifier()
    dist_ranker = get_distractor_ranker()
    hint_scorer = get_hint_scorer()

    art_clean = clean_text(article)

    # ── Step 1: Question ──────────────────────────────────────────────────────
    if existing_question and existing_question.strip():
        question = existing_question.strip()
        q_clean  = clean_text(question)
    else:
        # Generate from article using OHE-based template generator
        sent_candidates = extract_candidate_sentences(
            art_clean, art_clean[:50], ohe_vec, top_k=3)
        seed = sent_candidates[0] if sent_candidates else art_clean[:50]
        questions = generate_questions(art_clean, seed, ohe_vec)
        question  = questions[0] if questions else "What is the main idea of the passage?"
        q_clean   = clean_text(question)

    # ── Step 2: Answer candidates (short phrases) ─────────────────────────────
    candidates = _extract_answer_candidates(art_clean, q_clean, tfidf_vec,
                                            top_n=20)

    if len(candidates) < 4:
        # Pad with generic fallbacks
        candidates += [f"none of the above {i}" for i in range(4 - len(candidates))]

    # ── Step 3: Score candidates — pick best as correct answer ────────────────
    scored = []
    for cand in candidates[:12]:   # score top-12 for speed
        prob = _score_option(art_clean, q_clean, cand,
                             verifier, tfidf_vec, scaler)
        scored.append((prob, cand))
    scored.sort(reverse=True)

    correct_answer = scored[0][1]

    # ── Step 4: Distractors — from remaining candidates via Model B ───────────
    # Pass remaining candidates (excluding correct) to distractor generator
    remaining = [c for _, c in scored[1:]]
    distractors = generate_distractors(
        art_clean, q_clean, correct_answer,
        tfidf_vec, dist_ranker, top_k=3)

    # If distractors overlap with correct answer, replace from remaining
    clean_distractors = []
    for d in distractors:
        if d.lower() != correct_answer.lower():
            clean_distractors.append(d)
    # Pad from remaining candidates if needed
    for cand in remaining:
        if len(clean_distractors) >= 3:
            break
        if cand.lower() != correct_answer.lower() and cand not in clean_distractors:
            clean_distractors.append(cand)
    distractors = clean_distractors[:3]

    # ── Step 5: Hints ─────────────────────────────────────────────────────────
    hints = generate_hints(
        art_clean, q_clean, correct_answer,
        tfidf_vec, hint_scorer, n_hints=3)

    latency = round(time.time() - t0, 3)
    return {
        "article":        article,
        "question":       question,
        "correct_answer": correct_answer,
        "distractors":    distractors,
        "hints":          hints,
        "model_used":     model_name,
        "latency_s":      latency,
        "mode":           "custom",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Answer verification (used by UI Check Answer button)
# ══════════════════════════════════════════════════════════════════════════════
def verify_answer(article: str, question: str, chosen_option: str,
                  all_options: dict = None) -> dict:
    """
    Verify whether chosen_option is the correct answer.

    With all_options: score all 4, predict best, compare to chosen.
    Without all_options: score chosen option alone (fallback).
    """
    t0 = time.time()

    tfidf_vec = get_tfidf_vec()
    scaler    = get_dense_scaler()
    verifier, model_name = get_verifier()

    if all_options:
        scores = score_options(article, question, all_options,
                               verifier, tfidf_vec, scaler)
        predicted_key = max(scores, key=scores.get)
        chosen_key = next(
            (k for k, v in all_options.items()
             if v.strip().lower() == chosen_option.strip().lower()),
            None
        )
        is_correct = (chosen_key == predicted_key) if chosen_key else None
        confidence = scores.get(chosen_key, 0.0) if chosen_key else 0.0
    else:
        confidence = _score_option(article, question, chosen_option,
                                   verifier, tfidf_vec, scaler)
        predicted_key = None
        is_correct    = confidence >= 0.5

    latency = round(time.time() - t0, 3)
    return {
        "predicted_label": predicted_key,
        "is_correct":      is_correct,
        "confidence":      round(confidence, 4),
        "model_used":      model_name,
        "latency_s":       latency,
    }
