"""
test_inference.py
-----------------
Unit tests for preprocessing utilities and inference helpers.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.preprocessing import clean_text, lexical_features
from src.model_a_train import apply_wh_template, extract_candidate_sentences
from src.model_b_train import extract_candidate_phrases, distractor_features
from src.evaluate import exact_match, model_a_metrics, clustering_purity


# ── preprocessing ─────────────────────────────────────────────────────────────
class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert "," not in clean_text("Hello, world!")

    def test_collapses_whitespace(self):
        result = clean_text("too   many   spaces")
        assert "  " not in result

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""


class TestLexicalFeatures:
    def test_output_shape(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article":  "the cat sat on the mat",
            "question": "where did the cat sit",
            "A": "on the mat", "B": "in the box", "C": "under the table", "D": "near the door",
        }])
        feats = lexical_features(df)
        assert feats.shape == (1, 11)   # 6 lengths + 1 q-article overlap + 4 option overlaps

    def test_non_negative(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article":  "dogs are loyal animals",
            "question": "what are dogs",
            "A": "loyal", "B": "cats", "C": "birds", "D": "fish",
        }])
        feats = lexical_features(df)
        assert np.all(feats >= 0)


# ── model_a ───────────────────────────────────────────────────────────────────
class TestWhTemplate:
    def test_returns_list(self):
        result = apply_wh_template("the dog was running in the park")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_questions_end_with_mark(self):
        result = apply_wh_template("she was born in london")
        for q in result:
            assert q.endswith("?")


# ── model_b ───────────────────────────────────────────────────────────────────
class TestCandidateExtraction:
    def test_returns_list(self):
        article = "the quick brown fox jumps over the lazy dog near the river bank"
        result = extract_candidate_phrases(article, top_n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_no_stopwords(self):
        article = "the cat and the dog are friends"
        result = extract_candidate_phrases(article, top_n=10)
        stopwords = {"the", "and", "are"}
        for word in result:
            assert word not in stopwords


# ── evaluate ──────────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_exact_match_perfect(self):
        assert exact_match([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_exact_match_none(self):
        assert exact_match([0, 0, 0, 0], [1, 1, 1, 1]) == 0.0

    def test_model_a_metrics_keys(self):
        y_true = [0, 1, 2, 3, 0, 1]
        y_pred = [0, 1, 2, 2, 0, 0]
        metrics = model_a_metrics(y_true, y_pred)
        for key in ["accuracy", "macro_f1", "exact_match", "confusion_matrix"]:
            assert key in metrics

    def test_clustering_purity(self):
        y_true    = np.array([0, 0, 1, 1, 2, 2])
        clusters  = np.array([0, 0, 1, 1, 2, 2])
        assert clustering_purity(y_true, clusters) == 1.0

        mixed = np.array([0, 1, 0, 1, 0, 1])
        purity = clustering_purity(y_true, mixed)
        assert 0.0 <= purity <= 1.0
