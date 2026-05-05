"""
test_inference.py
-----------------
Unit tests for preprocessing utilities, model helpers, and evaluate functions.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.preprocessing import clean_text, lexical_features, expand_to_binary
from src.model_a_train import apply_wh_template
from src.model_b_train import (
    extract_candidate_phrases, split_sentences,
    hard_negatives, distractor_features,
)
from src.evaluate import (
    exact_match, model_a_metrics, clustering_purity,
    mcq_accuracy_from_binary, model_b_distractor_metrics,
    model_b_hint_metrics,
)


# ══════════════════════════════════════════════════════════════════════════════
# preprocessing — clean_text
# ══════════════════════════════════════════════════════════════════════════════
class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert "," not in clean_text("Hello, world!")

    def test_collapses_whitespace(self):
        assert "  " not in clean_text("too   many   spaces")

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123)  == ""

    def test_numbers_preserved(self):
        # digits should survive cleaning
        assert "42" in clean_text("answer is 42")


# ══════════════════════════════════════════════════════════════════════════════
# preprocessing — lexical_features
# ══════════════════════════════════════════════════════════════════════════════
class TestLexicalFeatures:
    def _make_df(self):
        import pandas as pd
        return pd.DataFrame([{
            "article":  "the cat sat on the mat",
            "question": "where did the cat sit",
            "A": "on the mat", "B": "in the box",
            "C": "under the table", "D": "near the door",
        }])

    def test_output_shape(self):
        feats = lexical_features(self._make_df())
        # 6 lengths + 1 q-article overlap + 4 option overlaps = 11
        assert feats.shape == (1, 11)

    def test_non_negative(self):
        feats = lexical_features(self._make_df())
        assert np.all(feats >= 0)


# ══════════════════════════════════════════════════════════════════════════════
# preprocessing — expand_to_binary
# ══════════════════════════════════════════════════════════════════════════════
class TestExpandToBinary:
    def test_four_rows_per_mcq(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article": "some text", "question": "q?",
            "A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d",
            "answer": "B",
        }])
        result = expand_to_binary(df)
        assert len(result) == 4

    def test_exactly_one_positive(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article": "some text", "question": "q?",
            "A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d",
            "answer": "C",
        }])
        result = expand_to_binary(df)
        assert result["label"].sum() == 1

    def test_correct_option_is_positive(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article": "some text", "question": "q?",
            "A": "wrong", "B": "wrong", "C": "right", "D": "wrong",
            "answer": "C",
        }])
        result = expand_to_binary(df)
        positive_row = result[result["label"] == 1].iloc[0]
        assert positive_row["option"] == "right"

    def test_columns_present(self):
        import pandas as pd
        df = pd.DataFrame([{
            "article": "text", "question": "q?",
            "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A",
        }])
        result = expand_to_binary(df)
        for col in ["article", "question", "option", "label"]:
            assert col in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# model_a — Wh-word template generation
# ══════════════════════════════════════════════════════════════════════════════
class TestWhTemplate:
    def test_returns_list(self):
        result = apply_wh_template("the dog was running in the park")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_questions_end_with_mark(self):
        for q in apply_wh_template("she was born in london"):
            assert q.endswith("?")

    def test_wh_words_present(self):
        result = apply_wh_template("the cat is on the mat")
        wh_words = {"Who", "What", "Where", "When", "Why", "How"}
        starts = {q.split()[0] for q in result}
        assert starts & wh_words  # at least one Wh-word


# ══════════════════════════════════════════════════════════════════════════════
# model_b — split_sentences
# ══════════════════════════════════════════════════════════════════════════════
class TestSplitSentences:
    def test_short_text_returns_chunks(self):
        text = " ".join(["word"] * 60)   # 60 words → ~2-3 chunks of 25
        result = split_sentences(text, chunk_words=25)
        assert len(result) >= 2

    def test_double_space_split(self):
        # Each part must be >= min_len (15) chars to survive the filter
        text = "this is the first part  this is the second part  this is the third part"
        result = split_sentences(text)
        assert len(result) >= 2

    def test_min_length_filter(self):
        text = "hi  " + " ".join(["word"] * 30)
        result = split_sentences(text, min_len=15)
        for s in result:
            assert len(s) >= 15

    def test_empty_string(self):
        result = split_sentences("")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# model_b — candidate extraction
# ══════════════════════════════════════════════════════════════════════════════
class TestCandidateExtraction:
    def test_returns_list(self):
        article = "the quick brown fox jumps over the lazy dog near the river bank"
        result = extract_candidate_phrases(article, top_n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_no_stopwords(self):
        article = "the cat and the dog are friends near the river"
        result = extract_candidate_phrases(article, top_n=20)
        # Import the actual stopwords set used by the function
        from src.model_b_train import STOPWORDS
        for phrase in result:
            first_word = phrase.split()[0]
            assert first_word not in STOPWORDS

    def test_min_length_filter(self):
        # All tokens < 4 chars should be excluded
        article = "the big cat sat"
        result = extract_candidate_phrases(article, top_n=10)
        for phrase in result:
            for token in phrase.split():
                assert len(token) >= 4

    def test_no_pure_digits(self):
        article = "there were 1234 students in 2023 at the school"
        result = extract_candidate_phrases(article, top_n=20)
        for phrase in result:
            assert not phrase.isdigit()


# ══════════════════════════════════════════════════════════════════════════════
# model_b — hard_negatives (vectorized)
# ══════════════════════════════════════════════════════════════════════════════
class TestHardNegatives:
    def _get_tfidf(self):
        """Load the real fitted TF-IDF vectorizer."""
        import pickle
        path = os.path.join("data", "processed", "tfidf_vectorizer.pkl")
        if not os.path.exists(path):
            pytest.skip("tfidf_vectorizer.pkl not found — run preprocessing first")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_returns_at_most_top_k(self):
        tfidf_vec = self._get_tfidf()
        candidates = ["student", "teacher", "school", "learning", "education"]
        result = hard_negatives(candidates, "student", ["pupil", "learner"],
                                tfidf_vec, top_k=3)
        assert len(result) <= 3

    def test_empty_candidates(self):
        tfidf_vec = self._get_tfidf()
        result = hard_negatives([], "answer", ["distractor"], tfidf_vec)
        assert result == []

    def test_returns_subset_of_candidates(self):
        tfidf_vec = self._get_tfidf()
        candidates = ["student", "teacher", "school"]
        result = hard_negatives(candidates, "student", ["pupil"],
                                tfidf_vec, top_k=2)
        for r in result:
            assert r in candidates


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — exact_match
# ══════════════════════════════════════════════════════════════════════════════
class TestExactMatch:
    def test_perfect(self):
        assert exact_match([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_none(self):
        assert exact_match([0, 0, 0, 0], [1, 1, 1, 1]) == 0.0

    def test_partial(self):
        score = exact_match([0, 1, 2, 3], [0, 1, 0, 0])
        assert score == 0.5


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — model_a_metrics
# ══════════════════════════════════════════════════════════════════════════════
class TestModelAMetrics:
    def test_required_keys(self):
        y_true = [0, 1, 2, 3, 0, 1]
        y_pred = [0, 1, 2, 2, 0, 0]
        m = model_a_metrics(y_true, y_pred)
        for key in ["accuracy", "macro_f1", "exact_match",
                    "macro_precision", "macro_recall", "confusion_matrix"]:
            assert key in m

    def test_perfect_score(self):
        y = [0, 1, 2, 3]
        m = model_a_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["exact_match"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — mcq_accuracy_from_binary
# ══════════════════════════════════════════════════════════════════════════════
class TestMCQAccuracyFromBinary:
    def test_perfect_prediction(self):
        # 2 questions, 4 options each
        # Q1: correct=option 2 (index 2), Q2: correct=option 0 (index 0)
        y_true  = [0, 0, 1, 0,   1, 0, 0, 0]
        y_proba = [0.1, 0.2, 0.9, 0.1,   0.8, 0.1, 0.1, 0.1]
        acc = mcq_accuracy_from_binary(y_true, y_proba)
        assert acc == 1.0

    def test_wrong_prediction(self):
        y_true  = [1, 0, 0, 0]
        y_proba = [0.1, 0.9, 0.1, 0.1]   # predicts option 1, true is option 0
        acc = mcq_accuracy_from_binary(y_true, y_proba)
        assert acc == 0.0

    def test_output_range(self):
        y_true  = [0, 1, 0, 0,   0, 0, 1, 0]
        y_proba = [0.3, 0.4, 0.2, 0.1,   0.1, 0.2, 0.6, 0.1]
        acc = mcq_accuracy_from_binary(y_true, y_proba)
        assert 0.0 <= acc <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — clustering_purity
# ══════════════════════════════════════════════════════════════════════════════
class TestClusteringPurity:
    def test_perfect_clustering(self):
        y_true   = np.array([0, 0, 1, 1, 2, 2])
        clusters = np.array([0, 0, 1, 1, 2, 2])
        assert clustering_purity(y_true, clusters) == 1.0

    def test_worst_clustering(self):
        # Each cluster has equal mix — purity = 0.5
        y_true   = np.array([0, 1, 0, 1])
        clusters = np.array([0, 0, 1, 1])
        purity = clustering_purity(y_true, clusters)
        assert 0.0 <= purity <= 1.0

    def test_numpy_array_input(self):
        # Must work with numpy arrays (not just lists)
        y_true   = np.array([0, 0, 1, 1])
        clusters = np.array([0, 0, 1, 1])
        assert clustering_purity(y_true, clusters) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — model_b_distractor_metrics
# ══════════════════════════════════════════════════════════════════════════════
class TestModelBDistractorMetrics:
    def test_required_keys(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 1, 0]
        m = model_b_distractor_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
            assert key in m

    def test_ndcg_with_proba(self):
        y_true  = [1, 1, 1, 0, 0, 0]
        y_pred  = [1, 1, 0, 0, 1, 0]
        y_proba = [0.9, 0.8, 0.4, 0.2, 0.6, 0.1]
        m = model_b_distractor_metrics(y_true, y_pred, y_proba=y_proba)
        assert "ndcg_at_3" in m
        assert 0.0 <= m["ndcg_at_3"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# evaluate — model_b_hint_metrics
# ══════════════════════════════════════════════════════════════════════════════
class TestModelBHintMetrics:
    def test_required_keys(self):
        y_true = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        m = model_b_hint_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in m

    def test_precision_at_3_with_proba(self):
        # 2 positives at indices 0 and 3
        y_true  = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_proba = [0.9, 0.1, 0.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]
        m = model_b_hint_metrics(y_true, [1,0,0,1,0,0,0,0,0,0], y_proba=y_proba)
        assert "precision_at_3" in m
        # top-3 by proba are indices 0, 3, 4 → 2 out of 3 are positive
        assert m["precision_at_3"] == pytest.approx(2/3, abs=0.01)

    def test_is_classification_not_regression(self):
        # Confirm no R2/RMSE keys (old wrong metric type)
        y_true = [1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 0]
        m = model_b_hint_metrics(y_true, y_pred)
        assert "r2_score" not in m
        assert "rmse"     not in m
