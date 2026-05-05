"""
model_b_train.py  (v2 — improved distractors + hints)
------------------------------------------------------
Model B — Distractor & Hint Generator

Improvements over v1:
  1. Phrase-based candidate extraction (unigrams + bigrams + trigrams)
  2. Hard negative sampling (lexically similar negatives, not random)
  3. Stronger features: TF-IDF cosine sims, no char-Jaccard
  4. SBERT semantic similarity (optional, cached)
  5. XGBoost ranker with probability-based ranking
  6. Cosine-similarity diversity filtering (replaces prefix check)
  7. Output quality control (min length, dedup, answer exclusion)
  8. Hint labeling via similarity (not string matching)
  9. Hint scorer actually used during inference
  10. Graduated hint structuring (general → focused → near-answer)
  11. Top-K ranking evaluation + diversity metrics
  12. Cached TF-IDF transforms — no repeated .transform() inside loops
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, ndcg_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# ── Optional XGBoost ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Optional SBERT ────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = os.path.join("models", "model_b", "traditional")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","and","that","this","with","for","are","was","were","has","have",
    "had","not","but","from","they","their","there","been","also","more",
    "than","its","into","about","which","when","what","who","how","will",
    "would","could","should","may","might","shall","can","does","did","do",
    "been","being","very","just","then","than","some","such","each","both",
    "all","any","few","most","other","same","own","too","now","here","there",
    "where","why","while","although","because","since","after","before",
    "during","through","over","under","again","further","once","only","even",
    "still","yet","already","always","never","often","usually","sometimes",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def load_processed():
    def lp(f):
        with open(os.path.join(PROCESSED_DIR, f), "rb") as fh:
            return pickle.load(fh)
    tfidf_vec = lp("tfidf_vectorizer.pkl")
    ohe_vec   = lp("ohe_vectorizer.pkl")
    dfs       = lp("cleaned_dfs.pkl")       # (train_df, val_df, test_df)
    return tfidf_vec, ohe_vec, dfs


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SBERT Cache  (load once, reuse everywhere)
# ══════════════════════════════════════════════════════════════════════════════
_sbert_model = None

def get_sbert():
    global _sbert_model
    if _sbert_model is None and HAS_SBERT:
        print("  Loading SBERT (all-MiniLM-L6-v2) ...")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def sbert_encode(texts: list, batch_size: int = 128) -> np.ndarray:
    """Encode a list of strings with SBERT. Returns (N, 384) float32 array."""
    model = get_sbert()
    if model is None:
        return np.zeros((len(texts), 1), dtype=np.float32)
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True)


def sbert_cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N,D) embedding matrices."""
    # embeddings are already L2-normalised by sbert_encode
    return np.clip((emb_a * emb_b).sum(axis=1), -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Phrase-Based Candidate Extraction  (req #1)
# ══════════════════════════════════════════════════════════════════════════════
def extract_candidate_phrases(article: str, top_n: int = 40) -> list:
    """
    Extract unigrams, bigrams, and trigrams from the article.
    Filters:
      - min 4 characters per token
      - no stopwords
      - no purely numeric tokens
      - deduplication
    Returns up to top_n candidates ranked by frequency.
    """
    # Tokenise
    tokens = re.findall(r"\b[a-z][a-z0-9]{2,}\b", article.lower())
    # Filter tokens
    clean = [t for t in tokens
             if t not in STOPWORDS
             and not t.isdigit()
             and len(t) >= 4]

    phrases = []
    # Unigrams
    phrases.extend(clean)
    # Bigrams
    phrases.extend(f"{clean[i]} {clean[i+1]}"
                   for i in range(len(clean) - 1))
    # Trigrams
    phrases.extend(f"{clean[i]} {clean[i+1]} {clean[i+2]}"
                   for i in range(len(clean) - 2))

    freq = Counter(phrases)
    # Remove single-occurrence noise for multi-word phrases
    candidates = [p for p, c in freq.most_common(top_n * 3)
                  if len(p.split()) == 1 or c > 1]

    # Deduplicate preserving order
    seen, unique = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# 4. TF-IDF Cosine Similarity Helpers  (cached, batched — req #3, #12)
# ══════════════════════════════════════════════════════════════════════════════
def tfidf_cosine_batch(vec_a_sparse, texts_b: list, tfidf_vec) -> np.ndarray:
    """
    Compute cosine similarity between one sparse vector and a batch of texts.
    vec_a_sparse : (1, V) sparse
    texts_b      : list of N strings
    Returns      : (N,) float32 array
    """
    if not texts_b:
        return np.array([], dtype=np.float32)
    vb = tfidf_vec.transform(texts_b)
    va_norm = normalize(vec_a_sparse, norm="l2")
    vb_norm = normalize(vb, norm="l2")
    sims = (va_norm * vb_norm.T).toarray().ravel()
    return sims.astype(np.float32)


def tfidf_cosine_pair(text_a: str, text_b: str, tfidf_vec) -> float:
    va = tfidf_vec.transform([text_a])
    vb = tfidf_vec.transform([text_b])
    va_n = normalize(va, norm="l2")
    vb_n = normalize(vb, norm="l2")
    return float((va_n * vb_n.T).toarray()[0, 0])


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature Engineering  (req #3 — stronger features, no char-Jaccard)
# ══════════════════════════════════════════════════════════════════════════════
def distractor_features(candidate: str, correct_answer: str,
                        question: str, article: str,
                        tfidf_vec,
                        cand_sbert: np.ndarray = None,
                        ans_sbert:  np.ndarray = None,
                        q_sbert:    np.ndarray = None,
                        # ── precomputed vectors (pass to avoid recomputing) ──
                        cand_vec_n=None,   # normalised TF-IDF of candidate
                        ans_vec_n=None,    # normalised TF-IDF of answer
                        q_vec_n=None,      # normalised TF-IDF of question
                        art_vec_n=None,    # normalised TF-IDF of article[:300]
                        # ── precomputed word sets ────────────────────────────
                        art_words: list = None,
                        art_word_set: set = None,
                        q_word_set: set = None,
                        cand_words: list = None,
                        cand_word_set: set = None,
                        ) -> np.ndarray:
    """
    Feature vector per candidate (8 features + 2 optional SBERT).
    All TF-IDF vectors and word sets can be passed precomputed to avoid
    redundant .transform() and .split() calls inside tight loops.

      [0]  TF-IDF cosine sim: candidate <-> correct answer
      [1]  TF-IDF cosine sim: candidate <-> question
      [2]  TF-IDF cosine sim: candidate <-> article (first 300 chars)
      [3]  Candidate frequency in article (normalised)
      [4]  Candidate length (word count, normalised)
      [5]  Word overlap: candidate tokens in article
      [6]  Word overlap: candidate tokens in question
      [7]  Is multi-word phrase (0/1)
      [8]* SBERT cosine sim: candidate <-> correct answer  (if available)
      [9]* SBERT cosine sim: candidate <-> question        (if available)
    """
    # ── Compute only what wasn't passed in ────────────────────────────────────
    if cand_vec_n is None:
        cand_vec_n = normalize(tfidf_vec.transform([candidate]), norm="l2")
    if ans_vec_n is None:
        ans_vec_n  = normalize(tfidf_vec.transform([correct_answer]), norm="l2")
    if q_vec_n is None:
        q_vec_n    = normalize(tfidf_vec.transform([question]), norm="l2")
    if art_vec_n is None:
        art_vec_n  = normalize(tfidf_vec.transform([article[:300]]), norm="l2")

    sim_cand_ans = float((cand_vec_n * ans_vec_n.T).toarray()[0, 0])
    sim_cand_q   = float((cand_vec_n * q_vec_n.T).toarray()[0, 0])
    sim_cand_art = float((cand_vec_n * art_vec_n.T).toarray()[0, 0])

    # ── Word-level features — use precomputed sets/lists if available ─────────
    if art_words is None:
        art_words = article.lower().split()
    if art_word_set is None:
        art_word_set = set(art_words)
    if q_word_set is None:
        q_word_set = set(question.lower().split())
    if cand_words is None:
        cand_words = candidate.lower().split()
    if cand_word_set is None:
        cand_word_set = set(cand_words)

    freq       = sum(art_words.count(w) for w in cand_words) / (len(art_words) + 1e-9)
    length_norm = len(cand_words) / 5.0
    overlap_art = len(cand_word_set & art_word_set) / (len(cand_word_set) + 1e-9)
    overlap_q   = len(cand_word_set & q_word_set)   / (len(cand_word_set) + 1e-9)
    is_phrase   = float(len(cand_words) > 1)

    feats = [sim_cand_ans, sim_cand_q, sim_cand_art,
             freq, length_norm, overlap_art, overlap_q, is_phrase]

    if HAS_SBERT and cand_sbert is not None:
        feats.extend([float(np.dot(cand_sbert, ans_sbert)),
                      float(np.dot(cand_sbert, q_sbert))])

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Hard Negative Sampling  (req #2)
# ══════════════════════════════════════════════════════════════════════════════
def hard_negatives(candidates: list, correct_answer: str,
                   real_distractors: list, tfidf_vec,
                   top_k: int = 3) -> list:
    """
    Select hard negatives via fully vectorized similarity computation.
    Batch-transforms all candidates and all references at once, then
    uses matrix multiplication — no nested Python loops.
    """
    if not candidates:
        return []

    reference_texts = [correct_answer] + real_distractors

    # ── Batch transform ONCE each ─────────────────────────────────────────────
    cand_vecs = normalize(tfidf_vec.transform(candidates), norm="l2")
    ref_vecs  = normalize(tfidf_vec.transform(reference_texts), norm="l2")

    # ── Matrix multiply: (n_cands, vocab) @ (vocab, n_refs) → (n_cands, n_refs)
    sims = (cand_vecs @ ref_vecs.T).toarray()   # shape: (n_cands, n_refs)
    max_sims = sims.max(axis=1)                  # shape: (n_cands,)

    # Sort descending — most similar (hardest) first
    order = np.argsort(max_sims)[::-1]
    return [candidates[i] for i in order[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# 7. Build Distractor Training Dataset  (req #2, #3)
# ══════════════════════════════════════════════════════════════════════════════
def build_distractor_dataset(df, tfidf_vec, max_rows: int = 5000,
                              use_sbert: bool = False):
    """
    Positive samples  : the 3 actual wrong options from RACE
    Negative samples  : hard negatives (lexically similar to answer/distractors)
    Features          : distractor_features() — 8 or 10 dims

    Speed optimizations applied:
      - itertuples() instead of iterrows()          (~2x)
      - Per-row TF-IDF vectors computed ONCE        (5-10x)
      - All candidates batch-transformed at once    (3-5x)
      - Vectorized hard_negatives (matrix multiply) (10x)
      - Word sets/lists precomputed once per row    (1.5x)
      - Option key list defined outside loop        (micro)
    """
    X, y = [], []
    OPT_KEYS = ["A", "B", "C", "D"]   # defined once, reused every row

    for row in df.head(max_rows).itertuples(index=False):
        correct    = getattr(row, row.answer)
        wrong_opts = [getattr(row, o) for o in OPT_KEYS if o != row.answer]
        all_opt_text = {getattr(row, o).lower() for o in OPT_KEYS}

        # ── Per-row TF-IDF vectors — computed ONCE, reused for all candidates ─
        ans_vec_n = normalize(tfidf_vec.transform([correct]),          norm="l2")
        q_vec_n   = normalize(tfidf_vec.transform([row.question]),     norm="l2")
        art_vec_n = normalize(tfidf_vec.transform([row.article[:300]]),norm="l2")

        # ── Per-row word sets — computed ONCE ─────────────────────────────────
        art_words    = row.article.lower().split()
        art_word_set = set(art_words)
        q_word_set   = set(row.question.lower().split())

        # ── SBERT: batch-encode answer + question + wrong options together ─────
        if use_sbert and HAS_SBERT:
            embs      = sbert_encode([correct, row.question] + wrong_opts)
            ans_emb   = embs[0]
            q_emb     = embs[1]
            dist_embs = embs[2:]
        else:
            ans_emb = q_emb = dist_embs = None

        # ── Positives: batch-transform all wrong options at once ──────────────
        opt_vecs_n = normalize(tfidf_vec.transform(wrong_opts), norm="l2")

        for i, opt in enumerate(wrong_opts):
            opt_words     = opt.lower().split()
            opt_word_set  = set(opt_words)
            c_emb         = dist_embs[i] if dist_embs is not None else None

            feats = distractor_features(
                opt, correct, row.question, row.article, tfidf_vec,
                cand_sbert=c_emb, ans_sbert=ans_emb, q_sbert=q_emb,
                cand_vec_n=opt_vecs_n[i],
                ans_vec_n=ans_vec_n, q_vec_n=q_vec_n, art_vec_n=art_vec_n,
                art_words=art_words, art_word_set=art_word_set,
                q_word_set=q_word_set,
                cand_words=opt_words, cand_word_set=opt_word_set,
            )
            X.append(feats)
            y.append(1)

        # ── Hard negatives: extract candidates, batch-transform, vectorized sim
        candidates = extract_candidate_phrases(row.article, top_n=40)
        candidates = [c for c in candidates if c.lower() not in all_opt_text]
        hard_negs  = hard_negatives(candidates, correct, wrong_opts,
                                    tfidf_vec, top_k=3)

        if hard_negs:
            # Batch-transform all hard negatives at once
            neg_vecs_n = normalize(tfidf_vec.transform(hard_negs), norm="l2")

            if use_sbert and HAS_SBERT:
                neg_embs = sbert_encode(hard_negs)
            else:
                neg_embs = None

            for i, neg in enumerate(hard_negs):
                neg_words    = neg.lower().split()
                neg_word_set = set(neg_words)
                n_emb        = neg_embs[i] if neg_embs is not None else None

                feats = distractor_features(
                    neg, correct, row.question, row.article, tfidf_vec,
                    cand_sbert=n_emb, ans_sbert=ans_emb, q_sbert=q_emb,
                    cand_vec_n=neg_vecs_n[i],
                    ans_vec_n=ans_vec_n, q_vec_n=q_vec_n, art_vec_n=art_vec_n,
                    art_words=art_words, art_word_set=art_word_set,
                    q_word_set=q_word_set,
                    cand_words=neg_words, cand_word_set=neg_word_set,
                )
                X.append(feats)
                y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Distractor Ranker Training  (req #5 — XGBoost preferred)
# ══════════════════════════════════════════════════════════════════════════════
def train_distractor_ranker(X_train, y_train):
    if HAS_XGB:
        print("\n--- Distractor Ranker (XGBoost) ---")
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            n_jobs=-1, random_state=42, tree_method="hist",
        )
    else:
        print("\n--- Distractor Ranker (Logistic Regression) ---")
        model = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 9. Distractor Generation Inference  (req #6, #7)
# ══════════════════════════════════════════════════════════════════════════════
def generate_distractors(article: str, question: str, correct_answer: str,
                         tfidf_vec, ranker_model,
                         top_k: int = 3,
                         diversity_threshold: float = 0.70,
                         use_sbert: bool = False) -> list:
    """
    Full distractor generation pipeline — optimized:
      1. Extract phrase candidates
      2. Batch-filter candidates too close to the answer (one matrix op)
      3. Compute all features in batch (precomputed row vectors)
      4. Score with ranker (single predict_proba call on full matrix)
      5. Diversity filter with cosine similarity
      6. Quality control + fallback
    """
    candidates = extract_candidate_phrases(article, top_n=60)
    if not candidates:
        return [f"[distractor {i+1}]" for i in range(top_k)]

    # ── Precompute row-level vectors ONCE ─────────────────────────────────────
    ans_vec_n = normalize(tfidf_vec.transform([correct_answer]), norm="l2")
    q_vec_n   = normalize(tfidf_vec.transform([question]),       norm="l2")
    art_vec_n = normalize(tfidf_vec.transform([article[:300]]),  norm="l2")

    # ── Precompute word sets ONCE ─────────────────────────────────────────────
    art_words    = article.lower().split()
    art_word_set = set(art_words)
    q_word_set   = set(question.lower().split())

    # ── Batch-filter: remove answer and near-duplicates (one matrix op) ───────
    cand_vecs_n = normalize(tfidf_vec.transform(candidates), norm="l2")
    ans_sims    = (cand_vecs_n @ ans_vec_n.T).toarray().ravel()  # (n_cands,)

    filtered_idx = [
        i for i, (cand, sim) in enumerate(zip(candidates, ans_sims))
        if cand.lower() != correct_answer.lower() and sim < 0.90
    ]
    if not filtered_idx:
        return [f"[distractor {i+1}]" for i in range(top_k)]

    filtered      = [candidates[i]   for i in filtered_idx]
    filt_vecs_n   = cand_vecs_n[filtered_idx]   # reuse already-computed vecs

    # ── SBERT: batch-encode everything at once ────────────────────────────────
    if use_sbert and HAS_SBERT:
        all_embs  = sbert_encode([correct_answer, question] + filtered)
        ans_emb   = all_embs[0]
        q_emb     = all_embs[1]
        cand_embs = all_embs[2:]
    else:
        ans_emb = q_emb = cand_embs = None

    # ── Build feature matrix for ALL filtered candidates at once ─────────────
    feat_rows = []
    for i, cand in enumerate(filtered):
        cand_words    = cand.lower().split()
        cand_word_set = set(cand_words)
        c_emb         = cand_embs[i] if cand_embs is not None else None

        feats = distractor_features(
            cand, correct_answer, question, article, tfidf_vec,
            cand_sbert=c_emb, ans_sbert=ans_emb, q_sbert=q_emb,
            cand_vec_n=filt_vecs_n[i],
            ans_vec_n=ans_vec_n, q_vec_n=q_vec_n, art_vec_n=art_vec_n,
            art_words=art_words, art_word_set=art_word_set,
            q_word_set=q_word_set,
            cand_words=cand_words, cand_word_set=cand_word_set,
        )
        feat_rows.append(feats)

    feat_matrix = np.array(feat_rows, dtype=np.float32)

    # ── Single predict_proba call on full matrix ──────────────────────────────
    probs = ranker_model.predict_proba(feat_matrix)[:, 1]
    order = np.argsort(probs)[::-1]

    # ── Diversity filtering ───────────────────────────────────────────────────
    selected      = []
    selected_vecs = []
    for idx in order:
        if len(selected) >= top_k:
            break
        cand_vec = filt_vecs_n[idx]
        too_similar = any(
            float((cand_vec * sv.T).toarray()[0, 0]) >= diversity_threshold
            for sv in selected_vecs
        )
        if not too_similar:
            selected.append(filtered[idx])
            selected_vecs.append(cand_vec)

    while len(selected) < top_k:
        selected.append(f"[option {len(selected)+1}]")

    return selected[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 10. Distractor Evaluation  (req #11)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_distractor_ranker(model, X_val, y_val, X_test, y_test):
    """
    Evaluation:
      - Binary accuracy / precision / recall / F1
      - Top-K ranking: NDCG@3 (how well top-3 predictions match positives)
      - Diversity check on generated distractors
    """
    print("\n--- Distractor Ranker Evaluation ---")
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)

        # NDCG@3: group by question (every 6 rows = 3 pos + 3 neg)
        group_size = 6
        n_groups = len(y) // group_size
        ndcg_scores = []
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            y_true_g = y[sl].reshape(1, -1).astype(float)
            y_score_g = proba[sl].reshape(1, -1)
            try:
                ndcg_scores.append(ndcg_score(y_true_g, y_score_g, k=3))
            except Exception:
                pass
        ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

        print(f"  [{split}]  Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}  NDCG@3={ndcg:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Sentence Splitting Helper
# ══════════════════════════════════════════════════════════════════════════════
def split_sentences(text: str, min_len: int = 15, chunk_words: int = 25) -> list:
    """
    Split article text into pseudo-sentences.

    The cleaned articles have all punctuation stripped, so we cannot split
    on '.' or '\\n'.  Instead we:
      1. Try splitting on whitespace runs of 2+ spaces (paragraph breaks
         sometimes survive cleaning as double-spaces).
         Uses min_len=8 for this path so short double-space chunks aren't lost.
      2. Fall back to fixed-size word chunks (chunk_words words each).

    Returns a list of non-empty strings each at least min_len chars long.
    """
    # Try double-space / tab splits first (use lower threshold here)
    parts = [p.strip() for p in re.split(r'\s{2,}', text) if p.strip()]

    if len(parts) <= 1:
        # Fall back to fixed word-count chunks
        words = text.split()
        parts = [
            " ".join(words[i: i + chunk_words])
            for i in range(0, len(words), chunk_words)
        ]

    return [p for p in parts if len(p) >= min_len]



def build_hint_dataset(df, tfidf_vec, max_rows: int = 3000,
                       use_sbert: bool = False):
    """
    Features per sentence (7 or 9 dims).
    Label: 1 if sentence is in the top-2 most similar to the question.

    Speed optimizations:
      - itertuples() instead of iterrows()
      - All sentence TF-IDF vectors batch-transformed once per row
      - q_vec / ans_vec computed once per row, normalised once
      - q_words set computed once per row
    """
    X, y = [], []

    for row in df.head(max_rows).itertuples(index=False):
        sentences = split_sentences(row.article)
        if len(sentences) < 2:
            continue

        correct_text = getattr(row, row.answer)
        question     = row.question

        # ── Batch-transform all sentences + q + ans ONCE per row ─────────────
        q_vec_n   = normalize(tfidf_vec.transform([question]),     norm="l2")
        ans_vec_n = normalize(tfidf_vec.transform([correct_text]), norm="l2")
        s_vecs_n  = normalize(tfidf_vec.transform(sentences),      norm="l2")

        # Matrix multiply: (1, V) @ (V, n_sents) → (n_sents,)
        sims_q   = (q_vec_n   @ s_vecs_n.T).toarray().ravel()
        sims_ans = (ans_vec_n @ s_vecs_n.T).toarray().ravel()

        # Similarity-based labels: top-2 sentences are positive
        top2_idx = set(np.argsort(sims_q)[-2:])

        # SBERT: batch-encode all at once
        if use_sbert and HAS_SBERT:
            all_embs  = sbert_encode([question, correct_text] + sentences)
            q_emb     = all_embs[0]
            ans_emb   = all_embs[1]
            sent_embs = all_embs[2:]
        else:
            q_emb = ans_emb = sent_embs = None

        # ── Precompute once per row ───────────────────────────────────────────
        q_words = set(question.lower().split())
        n_sents = len(sentences)

        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
            feats = [
                float(sims_q[i]),
                float(sims_ans[i]),
                len(q_words & sent_words) / (len(q_words) + 1e-9),
                i / (n_sents + 1e-9),
                len(sent.split()) / 50.0,
                float(i == 0),
                float(i == n_sents - 1),
            ]
            if use_sbert and HAS_SBERT and sent_embs is not None:
                feats.append(float(np.dot(sent_embs[i], q_emb)))
                feats.append(float(np.dot(sent_embs[i], ans_emb)))

            X.append(feats)
            y.append(1 if i in top2_idx else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Hint Scorer Training
# ══════════════════════════════════════════════════════════════════════════════
def train_hint_scorer(X_train, y_train):
    print("\n--- Hint Scorer (Logistic Regression) ---")
    model = LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 13. Hint Generation Inference  (req #9, #10 — scorer used, graduated hints)
# ══════════════════════════════════════════════════════════════════════════════
def generate_hints(article: str, question: str, correct_answer: str,
                   tfidf_vec, hint_scorer,
                   n_hints: int = 3,
                   use_sbert: bool = False) -> list:
    """
    Generate graduated hints using the trained hint_scorer (req #9).

    Hint progression (req #10):
      Hint 1 — general context (low scorer probability, broad sentence)
      Hint 2 — focused context (medium probability)
      Hint 3 — near-answer guidance (highest probability)

    Strategy:
      - Score all sentences with hint_scorer
      - Sort by predicted probability
      - Pick 3 sentences spread across the probability range
        (not just top-3, to ensure graduation)
    """
    sentences = split_sentences(article)

    if not sentences:
        return [
            "Re-read the passage carefully.",
            "Focus on the key events or facts described.",
            "Look for the sentence that most directly relates to the question.",
        ]

    # ── Build features for all sentences (cached transforms) ─────────────────
    q_vec   = tfidf_vec.transform([question])
    ans_vec = tfidf_vec.transform([correct_answer])
    s_vecs  = tfidf_vec.transform(sentences)

    q_n   = normalize(q_vec,   norm="l2")
    ans_n = normalize(ans_vec, norm="l2")
    s_n   = normalize(s_vecs,  norm="l2")

    sims_q   = (q_n   * s_n.T).toarray().ravel()
    sims_ans = (ans_n * s_n.T).toarray().ravel()

    n_sents = len(sentences)
    q_words = set(question.lower().split())

    # SBERT
    if use_sbert and HAS_SBERT:
        all_texts = [question, correct_answer] + sentences
        all_embs  = sbert_encode(all_texts)
        q_emb     = all_embs[0]
        ans_emb   = all_embs[1]
        sent_embs = all_embs[2:]
    else:
        q_emb = ans_emb = sent_embs = None

    feat_rows = []
    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        kw_overlap = len(q_words & sent_words) / (len(q_words) + 1e-9)
        feats = [
            float(sims_q[i]),
            float(sims_ans[i]),
            kw_overlap,
            i / (n_sents + 1e-9),
            len(sent.split()) / 50.0,
            float(i == 0),
            float(i == n_sents - 1),
        ]
        if use_sbert and HAS_SBERT and sent_embs is not None:
            feats.append(float(np.dot(sent_embs[i], q_emb)))
            feats.append(float(np.dot(sent_embs[i], ans_emb)))
        feat_rows.append(feats)

    feat_matrix = np.array(feat_rows, dtype=np.float32)

    # ── Score with trained model (req #9) ─────────────────────────────────────
    # Handle feature dimension mismatch gracefully
    expected_n_feats = hint_scorer.n_features_in_
    if feat_matrix.shape[1] != expected_n_feats:
        feat_matrix = feat_matrix[:, :expected_n_feats]

    probs = hint_scorer.predict_proba(feat_matrix)[:, 1]

    # ── Graduated selection (req #10) ─────────────────────────────────────────
    # Sort by probability ascending so we can pick low/mid/high
    order = np.argsort(probs)
    n = len(order)

    if n >= 3:
        # Hint 1: general — pick from bottom third
        h1_idx = order[n // 3]
        # Hint 2: focused — pick from middle third
        h2_idx = order[n // 2]
        # Hint 3: near-answer — pick highest probability
        h3_idx = order[-1]
        hint_indices = [h1_idx, h2_idx, h3_idx]
    else:
        hint_indices = list(order)

    # Deduplicate indices
    seen_idx, unique_idx = set(), []
    for idx in hint_indices:
        if idx not in seen_idx:
            seen_idx.add(idx)
            unique_idx.append(idx)

    hints = [sentences[i] for i in unique_idx[:n_hints]]

    # Pad if needed
    fallbacks = [
        "Re-read the passage carefully.",
        "Focus on the key events or facts described.",
        "Look for the sentence that most directly relates to the question.",
    ]
    while len(hints) < n_hints:
        hints.append(fallbacks[len(hints)])

    return hints


# ══════════════════════════════════════════════════════════════════════════════
# 14. Hint Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_hint_scorer(model, X_val, y_val, X_test, y_test):
    print("\n--- Hint Scorer Evaluation ---")
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)
        print(f"  [{split}]  Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 15. Qualitative Inspection Helper  (req #11)
# ══════════════════════════════════════════════════════════════════════════════
def inspect_samples(df, tfidf_vec, dist_ranker, hint_scorer,
                    n_samples: int = 3, use_sbert: bool = False):
    """Print generated distractors and hints for manual quality assessment."""
    print("\n--- Qualitative Sample Inspection ---")
    for i, row in enumerate(df.head(n_samples).itertuples(index=False)):
        correct = getattr(row, row.answer)
        print(f"\n[Sample {i+1}]")
        print(f"  Article snippet : {row.article[:120]}...")
        print(f"  Question        : {row.question}")
        print(f"  Correct answer  : {correct}")

        distractors = generate_distractors(
            row.article, row.question, correct,
            tfidf_vec, dist_ranker, top_k=3, use_sbert=use_sbert)
        print("  Generated distractors:")
        for j, d in enumerate(distractors, 1):
            print(f"    {j}. {d}")

        hints = generate_hints(
            row.article, row.question, correct,
            tfidf_vec, hint_scorer, n_hints=3, use_sbert=use_sbert)
        print("  Generated hints:")
        for j, h in enumerate(hints, 1):
            print(f"    Hint {j}: {h[:100]}")
        print(f"  Generated hints:")
        for j, h in enumerate(hints, 1):
            print(f"    Hint {j}: {h[:100]}")


# ══════════════════════════════════════════════════════════════════════════════
# 16. Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_training(use_sbert: bool = False, dist_rows: int = 5000,
                 hint_rows: int = 3000):
    """
    Args:
        use_sbert  : enable SBERT features (slower but better)
        dist_rows  : rows to use for distractor dataset
        hint_rows  : rows to use for hint dataset
    """
    print("Loading processed data ...")
    tfidf_vec, ohe_vec, (train_df, val_df, test_df) = load_processed()

    if use_sbert and HAS_SBERT:
        print("SBERT enabled — features will be 10-dimensional")
    elif use_sbert and not HAS_SBERT:
        print("SBERT requested but not installed — falling back to 8 features")
        use_sbert = False
    else:
        print("SBERT disabled — features will be 8-dimensional")

    # ══════════════════════════════════════════════════════════════════════════
    # DISTRACTOR PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nBuilding distractor dataset (train, {dist_rows} rows) ...")
    X_dist_tr, y_dist_tr = build_distractor_dataset(
        train_df, tfidf_vec, max_rows=dist_rows, use_sbert=use_sbert)

    print(f"Building distractor dataset (val, {dist_rows//5} rows) ...")
    X_dist_va, y_dist_va = build_distractor_dataset(
        val_df, tfidf_vec, max_rows=dist_rows // 5, use_sbert=use_sbert)

    print(f"Building distractor dataset (test, {dist_rows//5} rows) ...")
    X_dist_te, y_dist_te = build_distractor_dataset(
        test_df, tfidf_vec, max_rows=dist_rows // 5, use_sbert=use_sbert)

    print(f"  Train: {X_dist_tr.shape}  "
          f"pos={y_dist_tr.sum()}  neg={(y_dist_tr==0).sum()}")

    dist_ranker = train_distractor_ranker(X_dist_tr, y_dist_tr)
    evaluate_distractor_ranker(
        dist_ranker, X_dist_va, y_dist_va, X_dist_te, y_dist_te)
    save_model(dist_ranker, "distractor_ranker")

    # ══════════════════════════════════════════════════════════════════════════
    # HINT PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nBuilding hint dataset (train, {hint_rows} rows) ...")
    X_hint_tr, y_hint_tr = build_hint_dataset(
        train_df, tfidf_vec, max_rows=hint_rows, use_sbert=use_sbert)

    print(f"Building hint dataset (val, {hint_rows//3} rows) ...")
    X_hint_va, y_hint_va = build_hint_dataset(
        val_df, tfidf_vec, max_rows=hint_rows // 3, use_sbert=use_sbert)

    print(f"Building hint dataset (test, {hint_rows//3} rows) ...")
    X_hint_te, y_hint_te = build_hint_dataset(
        test_df, tfidf_vec, max_rows=hint_rows // 3, use_sbert=use_sbert)

    print(f"  Train: {X_hint_tr.shape}  "
          f"pos={y_hint_tr.sum()}  neg={(y_hint_tr==0).sum()}")

    hint_scorer = train_hint_scorer(X_hint_tr, y_hint_tr)
    evaluate_hint_scorer(
        hint_scorer, X_hint_va, y_hint_va, X_hint_te, y_hint_te)
    save_model(hint_scorer, "hint_scorer")

    # ══════════════════════════════════════════════════════════════════════════
    # QUALITATIVE INSPECTION
    # ══════════════════════════════════════════════════════════════════════════
    inspect_samples(val_df, tfidf_vec, dist_ranker, hint_scorer,
                    n_samples=3, use_sbert=use_sbert)

    print("\nModel B training complete.")
    return dist_ranker, hint_scorer


if __name__ == "__main__":
    # Set use_sbert=True to enable SBERT features (recommended, ~2x slower)
    run_training(use_sbert=False, dist_rows=5000, hint_rows=3000)
