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
                Uses rule-based QA extraction: finds a sentence with a
                clear subject-verb-object structure, generates a question
                about the object, and uses the object as the correct answer.
                Distractors are other entities of the same type from the article.
"""

import os
import re
import pickle
import time
import numpy as np
import joblib
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

from src.preprocessing import clean_text
from src.model_a_train import (
    generate_questions, extract_candidate_sentences, apply_wh_template,
)
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

def get_question_ranker():
    path = os.path.join(MODEL_A_DIR, "question_ranker.pkl")
    if os.path.exists(path):
        return _load_model(path)
    return None

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
    dense   = build_dense_features(bin_row, tfidf_vec, desc="inference", verbose=False)
    X, _    = combine_features(X_tfidf, dense, scaler=scaler, fit_scaler=False)
    return X


def _score_option(article: str, question: str, option: str,
                  verifier, tfidf_vec, scaler) -> float:
    X = _build_binary_features(article, question, option, tfidf_vec, scaler)
    if hasattr(verifier, "predict_proba"):
        return float(verifier.predict_proba(X)[0][1])
    return float(verifier.decision_function(X)[0])


def score_options(article: str, question: str, options: dict,
                  verifier, tfidf_vec, scaler) -> dict:
    return {
        key: _score_option(article, question, text, verifier, tfidf_vec, scaler)
        for key, text in options.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# Custom-mode: Rule-Based QA Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences_original(article: str) -> list:
    """Split original article into sentences using punctuation."""
    raw = re.split(r'(?<=[.!?])\s+', article.strip())
    sents = [s.strip() for s in raw if len(s.strip()) > 15]
    if len(sents) >= 2:
        return sents
    lines = [l.strip() for l in article.splitlines() if len(l.strip()) > 15]
    if len(lines) >= 2:
        return lines
    words = article.split()
    return [" ".join(words[i:i+25]) for i in range(0, len(words), 25)
            if len(words[i:i+25]) >= 4]


# ── QA pattern definitions ────────────────────────────────────────────────────
# Each entry: (compiled_regex, question_fn, answer_group_name, answer_type)
# answer_type: "place" | "person" | "thing" | "time"
#
# Design principle: question and answer come from the SAME sentence.
# The answer is always the grammatical object of the verb in the question.

_QA_PATTERNS = [
    # "X was born on DATE in PLACE" or "X was born in PLACE"
    # Handles: "Mother Teresa, whose name was ..., was born on August 26th, 1910 in Skopje"
    (
        re.compile(
            r'(?:^|,\s*)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            r'(?:,\s*[^,]+,)?\s+was born\s+(?:on\s+[^,]+,?\s+)?in\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[,\s]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where was {} born?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X received/was awarded the Y Prize/Award"
    (
        re.compile(
            r'(?:^|,\s*)(?:she|he|they|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'received\s+(?P<ans>the\s+\d{4}\s+[A-Za-z\s]+?(?:Prize|Award)[^,\.]*)',
            re.I
        ),
        lambda m: "What award did she receive?",
        "ans", "thing"
    ),
    # "X discovered/invented/developed Y"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?P<verb>discovered|invented|developed|created|founded|'
            r'established|designed|wrote|produced|built)\s+'
            r'(?:two\s+)?(?:new\s+)?(?:a|an|the)?\s*'
            r'(?P<ans>[a-z][a-z\s,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} {}?".format(m.group("subj"), m.group("verb")),
        "ans", "thing"
    ),
    # "X won the Nobel Prize / won Y"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+won\s+'
            r'(?P<ans>the\s+Nobel\s+[A-Za-z\s]+?Prize[^,\.]*)',
            re.I
        ),
        lambda m: "What did {} win?".format(m.group("subj")),
        "ans", "thing"
    ),
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+won\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} win?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X set up/founded/established Y in PLACE"
    (
        re.compile(
            r'(?:^|,\s*)(?:she|he|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:set up|founded|established|opened)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*(?:in|at)\s|[,;.]|$)',
            re.I
        ),
        lambda m: "What did she set up?",
        "ans", "thing"
    ),
    # "X moved/went/returned to PLACE"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:moved|travelled|went|came|returned)\s+to\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where did {} go?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was a/the Y" -> "What was X?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+was\s+'
            r'(?:a|an|the)\s+(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {}?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X struck/hit Y" -> "What did X strike?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:struck|hit|collided with)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did the {} strike?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X sank/crashed in PLACE/TIME"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:sank|crashed|exploded|collapsed)\s+(?:in|on)\s+'
            r'(?P<ans>[A-Za-z][A-Za-z\s,0-9]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "When did the {} sink?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X died in/on DATE" — also handles "passed away on DATE"
    (
        re.compile(
            r'(?:^|,\s*)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:died|passed away)\s+(?:in|on)\s+'
            r'(?P<ans>[A-Za-z0-9][A-Za-z0-9\s,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "When did {} die?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X studied/worked at/in PLACE"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:studied|worked|lived|graduated)\s+(?:at|in)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where did {} study?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X accepted/took/joined Y" -> "What did X accept?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?P<verb>accepted|joined|took|received|earned)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} {}?".format(m.group("subj"), m.group("verb")),
        "ans", "thing"
    ),
    # "X founded/established Y" -> "What did X found?"
    (
        re.compile(
            r'(?:^|,\s*)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:founded|established)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[A-Z][a-z]+(?:\s+[A-Za-z]+){0,4})',
            re.I
        ),
        lambda m: "What did {} found?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X became Y" -> "What did X become?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+became\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} become?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X began teaching/working in PLACE" -> "Where did X begin teaching?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'began\s+(?:teaching|working|studying)\s+in\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where did {} begin teaching?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X published/released/launched Y" -> "What did X publish?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?P<verb>published|released|launched|announced|introduced)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s,]+?)(?:\s*(?:in|on)\s+\d{4}|[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} {}?".format(m.group("subj"), m.group("verb")),
        "ans", "thing"
    ),
    # "X married Y" -> "Who did X marry?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+married\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who did {} marry?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X is located in/near PLACE" -> "Where is X located?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is located\s+(?:in|near|on)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where is {} located?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X is the capital of COUNTRY" -> "What is the capital of COUNTRY?"
    (
        re.compile(
            r'^(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is the capital\s+of\s+'
            r'(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            re.I
        ),
        lambda m: "What is the capital of {}?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was elected/appointed/named Y" -> "What was X elected?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:was elected|was appointed|was named|was chosen|was selected)\s+'
            r'(?:as\s+)?(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} elected as?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X graduated from PLACE" -> "Where did X graduate from?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'graduated from\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where did {} graduate from?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was raised/grew up in PLACE" -> "Where did X grow up?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:was raised|grew up|was brought up)\s+in\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where did {} grow up?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X spent Y years in PLACE" -> "Where did X spend time?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'spent\s+(?:[a-z\s\d]+)\s+(?:years?|months?|time)\s+in\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where did {} spend time?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X is known for Y" -> "What is X known for?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is known for\s+(?:its\s+|his\s+|her\s+)?(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What is {} known for?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was built/constructed in YEAR" -> "When was X built?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:built|constructed|completed|finished|erected)\s+in\s+'
            r'(?P<ans>\d{4})',
            re.I
        ),
        lambda m: "When was the {} built?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X was built/constructed by Y" -> "Who built X?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:built|constructed|designed|created|founded)\s+by\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who built the {}?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X covers/spans/stretches Y miles/km" -> "How long is X?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:covers|spans|stretches|extends)\s+(?:over\s+|about\s+)?'
            r'(?P<ans>[\d,]+\s+(?:miles?|kilometres?|km|meters?|feet|acres?))',
            re.I
        ),
        lambda m: "How long is the {}?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X has a population of Y" -> "What is the population of X?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'has a population of\s+(?:about\s+|over\s+|around\s+)?'
            r'(?P<ans>[\d,]+(?:\s+(?:million|billion|thousand))?)',
            re.I
        ),
        lambda m: "What is the population of {}?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was founded in YEAR" -> "When was X founded?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was founded\s+in\s+(?P<ans>\d{4})',
            re.I
        ),
        lambda m: "When was {} founded?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X is the largest/smallest/tallest Y in PLACE" -> "What is the largest Y in PLACE?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is the (?:largest|smallest|tallest|longest|oldest|biggest|highest)\s+'
            r'(?P<ans>[a-z][a-z\s]+?)\s+in\s+[A-Z]',
            re.I
        ),
        lambda m: "What is {} known as?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X joined/entered Y in YEAR" -> "What did X join?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:joined|entered|enrolled in|signed up for)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "What did {} join?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X served as Y" -> "What did X serve as?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'served as\s+(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What did {} serve as?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was sentenced to Y" -> "What was X sentenced to?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was sentenced to\s+(?P<ans>[a-z][a-z\s\d]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} sentenced to?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was arrested/captured/imprisoned in PLACE/YEAR" -> "When/Where was X arrested?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:arrested|captured|imprisoned|jailed)\s+in\s+'
            r'(?P<ans>[A-Za-z0-9][A-Za-z0-9\s,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "When was {} arrested?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X defeated/beat/overcame Y" -> "Who did X defeat?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:defeated|beat|overcame|conquered|overthrew)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who did {} defeat?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X was destroyed/damaged/burned in YEAR/EVENT" -> "When was X destroyed?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:destroyed|damaged|burned|demolished|torn down)\s+(?:in|during)\s+'
            r'(?P<ans>[A-Za-z0-9][A-Za-z0-9\s,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "When was the {} destroyed?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X takes place in/at PLACE" -> "Where does X take place?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'takes place\s+(?:in|at|near)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where does {} take place?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X is celebrated/observed on DATE" -> "When is X celebrated?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is (?:celebrated|observed|held|commemorated)\s+(?:on|every|each)\s+'
            r'(?P<ans>[A-Za-z][A-Za-z\s\d,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "When is {} celebrated?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X consists of/is made of Y" -> "What is X made of?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:consists of|is made of|is composed of|is made from)\s+'
            r'(?P<ans>[a-z][a-z\s,]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What is {} made of?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was inspired by Y" -> "What inspired X?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was inspired by\s+(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What inspired {}?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X is named after Y" -> "Who/What is X named after?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is named after\s+(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who is {} named after?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X replaced/succeeded Y" -> "Who did X replace?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:replaced|succeeded|took over from)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who did {} replace?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X was promoted to Y" -> "What was X promoted to?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was promoted to\s+(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} promoted to?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was born to Y" -> "Who were X's parents?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was born to\s+(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who were {}'s parents?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X is the son/daughter/child of Y" -> "Who is X the child of?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is the (?:son|daughter|child|nephew|niece|grandson|granddaughter)\s+of\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who is {} related to?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X flows through/into PLACE" -> "Where does X flow?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:flows|runs|passes)\s+(?:through|into|across)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where does the {} flow?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X borders/is bordered by Y" -> "What borders X?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:borders|is bordered by|is surrounded by)\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "What borders {}?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was signed/ratified in YEAR" -> "When was X signed?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:signed|ratified|passed|enacted|approved)\s+in\s+'
            r'(?P<ans>\d{4})',
            re.I
        ),
        lambda m: "When was the {} signed?".format(m.group("subj")),
        "ans", "time"
    ),
    # "X was awarded/given Y" -> "What was X awarded?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:awarded|given|presented with|granted)\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} awarded?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X trained/practised/competed in PLACE" -> "Where did X train?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(?:trained|practised|practiced|competed|performed)\s+in\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where did {} train?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was the first person to Y" -> "What was X the first to do?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was the first (?:person|woman|man|human|athlete|scientist|leader)?\s*to\s+'
            r'(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} the first to do?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X is divided into Y parts/sections/regions" -> "How is X divided?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is divided into\s+(?P<ans>[a-z\d][a-z\d\s]+?)\s+(?:parts?|sections?|regions?|zones?|areas?)',
            re.I
        ),
        lambda m: "How is {} divided?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was exiled/banished to PLACE" -> "Where was X exiled?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:exiled|banished|deported|expelled)\s+to\s+'
            r'(?P<ans>[A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
            re.I
        ),
        lambda m: "Where was {} exiled?".format(m.group("subj")),
        "ans", "place"
    ),
    # "X was inaugurated/sworn in as Y" -> "What was X inaugurated as?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:inaugurated|sworn in|installed)\s+as\s+'
            r'(?:a|an|the)?\s*(?P<ans>[a-z][a-z\s]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What was {} inaugurated as?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was raised/educated by Y" -> "Who raised X?"
    (
        re.compile(
            r'^(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:raised|educated|taught|trained|mentored)\s+by\s+'
            r'(?P<ans>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            re.I
        ),
        lambda m: "Who raised {}?".format(m.group("subj")),
        "ans", "person"
    ),
    # "X is home to Y" -> "What is X home to?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'is home to\s+(?:the\s+)?(?P<ans>[a-z][a-z\s,\d]+?)(?:\s*[,;.]|$)',
            re.I
        ),
        lambda m: "What is {} home to?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was translated into Y languages" -> "How many languages was X translated into?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was translated into\s+(?P<ans>[a-z\d][a-z\d\s]+?)\s+languages?',
            re.I
        ),
        lambda m: "How many languages was {} translated into?".format(m.group("subj")),
        "ans", "thing"
    ),
    # "X was performed/staged/shown at PLACE" -> "Where was X performed?"
    (
        re.compile(
            r'^(?:the\s+)?(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'was (?:performed|staged|shown|screened|exhibited|displayed)\s+at\s+'
            r'(?:the\s+)?(?P<ans>[A-Z][a-z]+(?:[\s]+[A-Za-z]+){0,3})',
            re.I
        ),
        lambda m: "Where was {} performed?".format(m.group("subj")),
        "ans", "place"
    ),
]


def _clean_answer(text: str) -> str:
    """Strip trailing punctuation and prepositions from extracted answer."""
    text = text.strip().rstrip(".,;:")
    TRAILING = {"in", "at", "on", "to", "for", "of", "with", "by",
                "from", "the", "a", "an", "and", "or", "but", "which",
                "where", "when", "who", "that"}
    words = text.split()
    while words and words[-1].lower() in TRAILING:
        words.pop()
    text = " ".join(words).strip()
    if text and text == text.lower():
        text = text.title()
    return text


def _extract_entities_by_type(article: str, answer_type: str,
                               exclude: str = "") -> list:
    """
    Extract entities of a specific semantic type from the article.
    Used to build plausible same-type distractors.
    """
    sents = _split_sentences_original(article)
    entities = []
    excl_clean = clean_text(exclude)

    for sent in sents:
        if answer_type == "place":
            places = re.findall(
                r'\b(?:in|at|to|from|near)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){0,2})',
                sent
            )
            places += re.findall(
                r'\b(?:born|moved|travelled|went|came|returned)\s+(?:in|to)\s+'
                r'([A-Z][a-z]+(?:[\s,]+[A-Za-z]+){0,2})',
                sent
            )
            for p in places:
                p = _clean_answer(p)
                if len(p) > 3 and clean_text(p) != excl_clean and p not in entities:
                    entities.append(p)

        elif answer_type == "person":
            names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', sent)
            for n in names:
                if clean_text(n) != excl_clean and n not in entities:
                    entities.append(n)

        elif answer_type == "thing":
            STOP = {"the","a","an","is","was","are","were","and","or","but",
                    "in","on","at","to","for","of","with","by","from","that",
                    "this","it","he","she","they","we","i","not","be","been"}
            # Simple: find "the/a/an WORD WORD" patterns — no backtracking risk
            words_in_sent = sent.split()
            for j, w in enumerate(words_in_sent):
                if w.lower() in {"the", "a", "an"} and j + 1 < len(words_in_sent):
                    # Take 2-3 words after the article
                    phrase_words = []
                    for k in range(j + 1, min(j + 4, len(words_in_sent))):
                        pw = words_in_sent[k].rstrip(".,;:")
                        if pw.lower() in STOP or not pw.isalpha():
                            break
                        phrase_words.append(pw.lower())
                    if len(phrase_words) >= 2:
                        phrase = " ".join(phrase_words)
                        if (len(phrase) > 5
                                and clean_text(phrase) != excl_clean):
                            t_titled = phrase.title()
                            if t_titled not in entities:
                                entities.append(t_titled)
            # Also multi-word proper nouns (organisations, awards)
            named = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', sent)
            for n in named:
                if clean_text(n) != excl_clean and n not in entities:
                    entities.append(n)

        elif answer_type == "time":
            years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', sent)
            for y in years:
                if y != excl_clean and y not in entities:
                    entities.append(y)
            dates = re.findall(
                r'\b(?:in|on)\s+((?:January|February|March|April|May|June|'
                r'July|August|September|October|November|December)'
                r'(?:\s+\d{1,2})?,?\s*\d{4})',
                sent
            )
            for d in dates:
                d = d.strip()
                if clean_text(d) != excl_clean and d not in entities:
                    entities.append(d)

    # Deduplicate, remove substrings
    seen, result = set(), []
    for e in entities:
        key = clean_text(e)
        if key and key not in seen and len(e) > 2:
            if not any(key in s or s in key for s in seen):
                seen.add(key)
                result.append(e)

    return result


def _extract_qa_from_article(article: str):
    """
    Scan article sentences for QA patterns.
    Returns (question, correct_answer, answer_type, best_sentence) or None.

    Two passes:
      Pass 1: sentences starting with a proper noun (most reliable)
      Pass 2: all sentences (catches "she received...", "In 1950, she set up...")
    """
    sents = _split_sentences_original(article)

    # Words that make poor question subjects when at sentence start
    SKIP_FW = {
        "although","when","while","because","since","after","before",
        "if","as","so","but","and","or","there","every","many",
        "building","constructing","today","however","during",
        "more","less","most","least","several","some","all","both",
    }

    def _try_patterns(sent):
        for pattern, q_fn, ans_group, ans_type in _QA_PATTERNS:
            m = pattern.search(sent)  # use search not match for flexibility
            if m:
                try:
                    question = q_fn(m)
                    answer   = _clean_answer(m.group(ans_group))
                    # Validate: answer must be non-trivial
                    if (len(answer) > 2
                            and answer.lower() not in {"he", "she", "it",
                                                        "they", "this", "that",
                                                        "her", "his"}):
                        # Validate: question must not use a pronoun as subject
                        PRONOUNS = {"it", "she", "he", "they", "we", "i",
                                    "her", "his", "its", "their"}
                        q_words = question.lower().split()
                        # Check if the subject word in the question is a pronoun
                        # e.g. "What was It?" or "What did She founded?"
                        subject_in_q = None
                        for qw in q_words:
                            qw_clean = qw.rstrip("?")
                            if qw_clean not in {"what", "where", "when", "why",
                                                "how", "who", "did", "was",
                                                "were", "the", "a", "an"}:
                                subject_in_q = qw_clean
                                break
                        if subject_in_q and subject_in_q in PRONOUNS:
                            continue  # skip pronoun subjects
                        return question, answer, ans_type, sent
                except Exception:
                    continue
        return None

    # Pass 1: sentences starting with a proper noun (highest quality)
    for sent in sents:
        fw = sent.split()[0].lower() if sent.split() else ""
        if fw in SKIP_FW:
            continue
        result = _try_patterns(sent)
        if result:
            return result

    # Pass 2: all sentences (catches "she received...", "In 1950, she set up...")
    for sent in sents:
        result = _try_patterns(sent)
        if result:
            return result

    return None


def _build_mcq_options(correct_answer: str, answer_type: str,
                       article: str, n_distractors: int = 3) -> list:
    """
    Build plausible distractors of the same semantic type as the correct answer.

    Strategy:
    1. Extract entities of the same type from the article
    2. Filter out entities that are substrings/superstrings of the correct answer
    3. Prefer entities that are structurally similar (same length, same type)
    4. Fall back to named entities if not enough same-type distractors
    """
    excl_clean = clean_text(correct_answer)
    distractors = []
    seen = {excl_clean}

    # ── Special handling for award/prize answers ──────────────────────────────
    # If the answer contains "Prize" or "Award", look for other prizes/awards
    # in the article first, then generate year-variant alternatives
    if any(w in correct_answer for w in ["Prize", "Award", "Medal", "Honor"]):
        sents = _split_sentences_original(article)
        for sent in sents:
            prizes = re.findall(
                r'\b(?:the\s+)?(?:\d{4}\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
                r'(?:Prize|Award|Medal|Honor)[^,\.]*',
                sent
            )
            for p in prizes:
                p = p.strip().rstrip(".,;")
                key = clean_text(p)
                if (key and key not in seen and len(p) > 5
                        and not any(key in s or s in key for s in seen)):
                    distractors.append(p)
                    seen.add(key)
                if len(distractors) >= n_distractors:
                    break
            if len(distractors) >= n_distractors:
                break

        # Generate year-variant distractors if article only has one prize
        # e.g. "the 1979 Nobel Peace Prize" → "the 1985 Nobel Peace Prize"
        if len(distractors) < n_distractors:
            year_match = re.search(r'\b(\d{4})\b', correct_answer)
            if year_match:
                base_year = int(year_match.group(1))
                base_prize = re.sub(r'\b\d{4}\b', '{}', correct_answer)
                for offset in [6, -6, 12, -12, 3, -3]:
                    if len(distractors) >= n_distractors:
                        break
                    alt_year = base_year + offset
                    if 1900 <= alt_year <= 2024:
                        alt = base_prize.format(alt_year)
                        key = clean_text(alt)
                        if key not in seen:
                            distractors.append(alt)
                            seen.add(key)

    # ── Special handling for place answers ───────────────────────────────────
    if answer_type == "place" and len(distractors) < n_distractors:
        entities = _extract_entities_by_type(article, "place", exclude=correct_answer)
        for e in entities:
            if len(distractors) >= n_distractors:
                break
            key = clean_text(e)
            if key and key not in seen and len(e) > 2:
                if not any(key in s or s in key for s in seen):
                    distractors.append(e)
                    seen.add(key)

    # ── Special handling for time answers ────────────────────────────────────
    if answer_type == "time" and len(distractors) < n_distractors:
        entities = _extract_entities_by_type(article, "time", exclude=correct_answer)
        for e in entities:
            if len(distractors) >= n_distractors:
                break
            key = clean_text(e)
            if key and key not in seen and len(e) > 2:
                if not any(key in s or s in key for s in seen):
                    distractors.append(e)
                    seen.add(key)

    # ── General thing/person extraction ──────────────────────────────────────
    if len(distractors) < n_distractors:
        entities = _extract_entities_by_type(article, answer_type,
                                              exclude=correct_answer)
        for e in entities:
            if len(distractors) >= n_distractors:
                break
            key = clean_text(e)
            if key and key not in seen and len(e) > 2:
                if not any(key in s or s in key for s in seen):
                    distractors.append(e)
                    seen.add(key)

    # ── Fallback: noun phrases from article (2+ words, not people names) ─────
    if len(distractors) < n_distractors:
        sents = _split_sentences_original(article)
        STOP = {"the","a","an","is","was","are","were","and","or","but",
                "in","on","at","to","for","of","with","by","from","that",
                "this","it","he","she","they","we","i","not","be","been",
                "her","his","their","its","our","my","very","just","also"}
        for sent in sents:
            if len(distractors) >= n_distractors:
                break
            words_in_sent = sent.split()
            for j, w in enumerate(words_in_sent):
                if len(distractors) >= n_distractors:
                    break
                if w.lower() in {"the", "a", "an"} and j + 1 < len(words_in_sent):
                    phrase_words = []
                    for k in range(j + 1, min(j + 4, len(words_in_sent))):
                        pw = words_in_sent[k].rstrip(".,;:")
                        if pw.lower() in STOP or not pw.isalpha():
                            break
                        phrase_words.append(pw.lower())
                    if len(phrase_words) >= 2:
                        phrase = " ".join(phrase_words)
                        key = clean_text(phrase)
                        if (len(phrase) > 6 and key not in seen
                                and not any(key in s or s in key for s in seen)):
                            distractors.append(phrase.title())
                            seen.add(key)

    return distractors[:n_distractors]


# ══════════════════════════════════════════════════════════════════════════════
# RACE mode pipeline — uses real options
# ══════════════════════════════════════════════════════════════════════════════
def run_race_pipeline(article: str, question: str,
                      options: dict, correct_key: str) -> dict:
    t0 = time.time()

    tfidf_vec = get_tfidf_vec()
    scaler    = get_dense_scaler()
    verifier, model_name = get_verifier()
    hint_scorer = get_hint_scorer()

    art_clean = clean_text(article)
    q_clean   = clean_text(question)
    opts_clean = {k: clean_text(v) for k, v in options.items()}
    correct_text = opts_clean[correct_key]

    scores = score_options(art_clean, q_clean, opts_clean,
                           verifier, tfidf_vec, scaler)
    predicted_key = max(scores, key=scores.get)

    hints = generate_hints(
        art_clean, q_clean, correct_text,
        tfidf_vec, hint_scorer, n_hints=3)

    latency = round(time.time() - t0, 3)
    return {
        "article":        article,
        "question":       question,
        "options":        options,
        "correct_key":    correct_key,
        "predicted_key":  predicted_key,
        "scores":         scores,
        "hints":          hints,
        "model_used":     model_name,
        "latency_s":      latency,
        "mode":           "race",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Custom mode pipeline — rule-based QA extraction
# ══════════════════════════════════════════════════════════════════════════════
def run_full_pipeline(article: str, existing_question: str = None) -> dict:
    """
    Custom mode: only article (+ optional question) provided.

    Rule-Based QA Extraction approach:
      1. Scan article sentences for clear subject-verb-object patterns
      2. Generate question AND extract correct answer from the SAME sentence
      3. Build distractors of the same semantic type (places->places, etc.)
      4. Generate hints with Model B

    This avoids TF-IDF vocabulary mismatch entirely for answer selection.
    """
    t0 = time.time()

    tfidf_vec   = get_tfidf_vec()
    scaler      = get_dense_scaler()
    verifier, model_name = get_verifier()
    hint_scorer = get_hint_scorer()

    art_clean = clean_text(article)

    # ── Step 1: Extract QA pair from article ──────────────────────────────────
    if existing_question and existing_question.strip():
        question    = existing_question.strip()
        q_clean     = clean_text(question)
        answer_type = "thing"

        # Find the sentence most relevant to the question
        sents = _split_sentences_original(article)
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_clean))
        best_score, best_sent = 0, sents[0] if sents else ""
        for sent in sents:
            sent_words = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
            overlap = len(q_words & sent_words)
            if overlap > best_score:
                best_score, best_sent = overlap, sent

        # Try QA patterns on the best sentence
        correct_answer = None
        for pattern, q_fn, ans_group, ans_type in _QA_PATTERNS:
            m = pattern.match(best_sent)
            if m:
                try:
                    ans = _clean_answer(m.group(ans_group))
                    if len(ans) > 2:
                        correct_answer = ans
                        answer_type    = ans_type
                        break
                except Exception:
                    pass

        # Fallback: first named entity from best sentence
        if not correct_answer and best_sent:
            named = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
                               best_sent)
            if named:
                correct_answer = named[0]

        if not correct_answer:
            correct_answer = "See the passage"

    else:
        # Auto-generate question + answer from article
        qa_result = _extract_qa_from_article(article)

        if qa_result:
            question, correct_answer, answer_type, _ = qa_result
            q_clean = clean_text(question)
        else:
            # Fallback
            sents = _split_sentences_original(article)
            question    = "What is the main idea of the passage?"
            q_clean     = clean_text(question)
            answer_type = "thing"
            correct_answer = None
            for sent in sents:
                named = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
                                   sent)
                if named:
                    correct_answer = named[0]
                    break
            if not correct_answer:
                correct_answer = "See the passage"

    # ── Step 2: Build same-type distractors ───────────────────────────────────
    distractors = _build_mcq_options(correct_answer, answer_type,
                                     article, n_distractors=3)
    while len(distractors) < 3:
        distractors.append("None of the above")

    # ── Step 3: Hints (Model B) ───────────────────────────────────────────────
    hints = generate_hints(
        art_clean, q_clean, clean_text(correct_answer),
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
