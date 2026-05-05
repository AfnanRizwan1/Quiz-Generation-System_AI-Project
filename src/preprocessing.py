"""
preprocessing.py
----------------
Dataset loading, cleaning, and feature engineering for Model A.

KEY DESIGN: Binary classification formulation
  Each MCQ row → 4 binary samples: (article + question + option_i, label=1/0)
  This is the correct framing for MCQ answer selection.

Feature pipeline per sample:
  1. TF-IDF with bigrams on (article + question + option) text  [sparse]
  2. Cosine similarity: article↔option, question↔option         [dense, 2 feats]
  3. Lexical features: lengths + overlap ratios                  [dense, 6 feats]
  → Combined with hstack + StandardScaler on dense block

Dataset note:
  Single dev.csv/dev.csv is auto-split 80/10/10 if separate files don't exist.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
DEV_CSV       = os.path.join("dev.csv", "dev.csv")
PROCESSED_DIR = os.path.join("data", "processed")
TFIDF_VOCAB   = 8000    # vocabulary size for TF-IDF
OHE_VOCAB     = 5000    # kept for Model B / inference compatibility
RANDOM_STATE  = 42
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Splitting
# ══════════════════════════════════════════════════════════════════════════════
def _load_single_file() -> pd.DataFrame:
    df = pd.read_csv(DEV_CSV)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def load_splits():
    """Return (train_df, val_df, test_df). Auto-splits dev.csv if needed."""
    train_path = os.path.join(RAW_DIR, "train.csv")
    val_path   = os.path.join(RAW_DIR, "val.csv")
    test_path  = os.path.join(RAW_DIR, "test.csv")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        train = pd.read_csv(train_path)
        val   = pd.read_csv(val_path)
        test  = pd.read_csv(test_path)
        print(f"Loaded  train={train.shape}  val={val.shape}  test={test.shape}")
        return train, val, test

    print(f"Splitting {DEV_CSV} 80/10/10 …")
    df = _load_single_file()
    train, temp = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE,
                                   stratify=df["answer"])
    val, test = train_test_split(temp, test_size=0.50, random_state=RANDOM_STATE,
                                 stratify=temp["answer"])
    for split, path in [(train, train_path), (val, val_path), (test, test_path)]:
        split.reset_index(drop=True).to_csv(path, index=False)
    print(f"Saved  train={train.shape}  val={val.shape}  test={test.shape}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Text Cleaning
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["article", "question", "A", "B", "C", "D"]:
        df[col] = df[col].apply(clean_text)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Binary Expansion  ← THE KEY CHANGE
#    Each MCQ row → 4 rows: (article, question, option, label)
# ══════════════════════════════════════════════════════════════════════════════
def expand_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each MCQ row into 4 binary-labelled rows.
    Returns a DataFrame with columns:
        article, question, option, label (1=correct, 0=wrong)
    """
    records = []
    for _, row in df.iterrows():
        correct = row["answer"]          # 'A', 'B', 'C', or 'D'
        for opt_key in ["A", "B", "C", "D"]:
            records.append({
                "article":  row["article"],
                "question": row["question"],
                "option":   row[opt_key],
                "label":    1 if opt_key == correct else 0,
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. TF-IDF on (article + question + option) ───────────────────────────────
def build_tfidf_features(train_bin, val_bin, test_bin):
    """
    Fit TF-IDF with unigrams + bigrams.
    Text = question + option  (articles are too long for bigrams at 281k scale;
    article content is captured via cosine-similarity dense features instead).
    """
    def concat(r):
        # Truncate article to first 200 chars to keep memory manageable
        art_snippet = r["article"][:200]
        return art_snippet + " " + r["question"] + " " + r["option"]

    train_text = train_bin.apply(concat, axis=1)
    val_text   = val_bin.apply(concat, axis=1)
    test_text  = test_bin.apply(concat, axis=1)

    vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=5,
        strip_accents="unicode",
        dtype=np.float32,       # halves memory vs float64
    )
    X_tr = vec.fit_transform(train_text)
    X_va = vec.transform(val_text)
    X_te = vec.transform(test_text)
    print(f"TF-IDF  train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")
    return X_tr, X_va, X_te, vec


# ── 4b. Dense numerical features ─────────────────────────────────────────────


def build_dense_features(bin_df: pd.DataFrame, tfidf_vec,
                         desc: str = "") -> np.ndarray:
    """
    Build (N, 8) dense feature matrix for a binary-expanded DataFrame.
    Fully vectorised — no Python row loop.
    """
    print(f"  Dense features [{desc}]: vectorising {len(bin_df):,} rows …")

    articles  = bin_df["article"].tolist()
    questions = bin_df["question"].tolist()
    options   = bin_df["option"].tolist()

    # ── Word-level features (pure Python sets → numpy) ────────────────────────
    def word_overlap(list_a, list_b):
        return np.array([
            len(set(a.split()) & set(b.split())) / (len(set(b.split())) + 1e-9)
            for a, b in zip(list_a, list_b)
        ], dtype=np.float32)

    art_lens = np.log1p([len(a.split()) for a in articles]).astype(np.float32)
    q_lens   = np.array([len(q.split()) for q in questions], dtype=np.float32)
    opt_lens = np.array([len(o.split()) for o in options],   dtype=np.float32)

    opt_art_overlap = word_overlap(articles,  options)
    opt_q_overlap   = word_overlap(questions, options)
    q_art_overlap   = word_overlap(articles,  questions)

    # ── TF-IDF cosine similarities (batch transform) ──────────────────────────
    print(f"    TF-IDF transforms …")
    BATCH = 10000   # transform in chunks to avoid peak memory spike

    def batch_cosine(list_a, list_b):
        sims = np.zeros(len(list_a), dtype=np.float32)
        for start in range(0, len(list_a), BATCH):
            end   = min(start + BATCH, len(list_a))
            va    = tfidf_vec.transform(list_a[start:end])
            vb    = tfidf_vec.transform(list_b[start:end])
            # row-wise dot product of L2-normalised vectors = cosine sim
            from sklearn.preprocessing import normalize
            va_n  = normalize(va, norm="l2")
            vb_n  = normalize(vb, norm="l2")
            sims[start:end] = np.array(va_n.multiply(vb_n).sum(axis=1)).ravel()
        return sims

    sim_art_opt = batch_cosine(articles, options)
    sim_q_opt   = batch_cosine(questions, options)

    dense = np.column_stack([
        art_lens, q_lens, opt_lens,
        opt_art_overlap, opt_q_overlap, q_art_overlap,
        sim_art_opt, sim_q_opt,
    ])
    print(f"    Done — shape {dense.shape}")
    return dense


# ── 4c. Combined sparse + dense ───────────────────────────────────────────────
def combine_features(X_tfidf, dense: np.ndarray, scaler=None, fit_scaler=False):
    """
    hstack(TF-IDF sparse, scaled dense).
    Pass fit_scaler=True for training set; pass fitted scaler for val/test.
    Returns (X_combined, scaler).
    """
    if fit_scaler:
        scaler = StandardScaler(with_mean=False)   # sparse-safe
        dense_scaled = scaler.fit_transform(dense)
    else:
        dense_scaled = scaler.transform(dense)

    X = hstack([X_tfidf, csr_matrix(dense_scaled)])
    return X, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 5. Label helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_binary_labels(bin_df: pd.DataFrame) -> np.ndarray:
    return bin_df["label"].values.astype(np.int32)


def encode_labels(train_df, val_df, test_df):
    """4-class labels (A/B/C/D → 0-3) — kept for clustering evaluation."""
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["answer"])
    y_val   = le.transform(val_df["answer"])
    y_test  = le.transform(test_df["answer"])
    return y_train, y_val, y_test, le


# ══════════════════════════════════════════════════════════════════════════════
# 6. OHE vectorizer  (kept for Model B / inference compatibility)
# ══════════════════════════════════════════════════════════════════════════════
def build_ohe_features(train_df, val_df, test_df):
    def concat_row(r):
        return " ".join([r["article"], r["question"], r["A"], r["B"], r["C"], r["D"]])

    ohe_vec = CountVectorizer(max_features=OHE_VOCAB, binary=True)
    X_train = ohe_vec.fit_transform(train_df.apply(concat_row, axis=1))
    X_val   = ohe_vec.transform(val_df.apply(concat_row, axis=1))
    X_test  = ohe_vec.transform(test_df.apply(concat_row, axis=1))
    print(f"OHE  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return X_train, X_val, X_test, ohe_vec


# ══════════════════════════════════════════════════════════════════════════════
# 7. Lexical features (original — kept for Model B)
# ══════════════════════════════════════════════════════════════════════════════
def lexical_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        art_words = set(r["article"].split())
        q_words   = set(r["question"].split())
        overlap_qa = len(q_words & art_words) / (len(q_words) + 1e-9)
        opt_overlaps = []
        for opt in ["A", "B", "C", "D"]:
            opt_words = set(r[opt].split())
            opt_overlaps.append(len(opt_words & art_words) / (len(opt_words) + 1e-9))
        rows.append([
            len(r["article"].split()), len(r["question"].split()),
            len(r["A"].split()), len(r["B"].split()),
            len(r["C"].split()), len(r["D"].split()),
            overlap_qa, *opt_overlaps,
        ])
    return np.array(rows, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_pickle(obj, filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved → {path}")


def load_pickle(filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing():
    train_df, val_df, test_df = load_splits()

    train_df = clean_df(train_df)
    val_df   = clean_df(val_df)
    test_df  = clean_df(test_df)

    # ── Binary expansion ──────────────────────────────────────────────────────
    print("\nExpanding to binary samples …")
    train_bin = expand_to_binary(train_df)
    val_bin   = expand_to_binary(val_df)
    test_bin  = expand_to_binary(test_df)
    print(f"Binary  train={train_bin.shape}  val={val_bin.shape}  test={test_bin.shape}")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("\nBuilding TF-IDF features …")
    X_tr_tfidf, X_va_tfidf, X_te_tfidf, tfidf_vec = build_tfidf_features(
        train_bin, val_bin, test_bin)

    # ── Dense features ────────────────────────────────────────────────────────
    print("\nBuilding dense features …")
    dense_train = build_dense_features(train_bin, tfidf_vec, "train")
    dense_val   = build_dense_features(val_bin,   tfidf_vec, "val")
    dense_test  = build_dense_features(test_bin,  tfidf_vec, "test")

    # ── Combine ───────────────────────────────────────────────────────────────
    print("\nCombining features …")
    X_train, scaler = combine_features(X_tr_tfidf, dense_train, fit_scaler=True)
    X_val,   _      = combine_features(X_va_tfidf, dense_val,   scaler=scaler)
    X_test,  _      = combine_features(X_te_tfidf, dense_test,  scaler=scaler)

    y_train = get_binary_labels(train_bin)
    y_val   = get_binary_labels(val_bin)
    y_test  = get_binary_labels(test_bin)

    # ── OHE (for Model B / inference) ─────────────────────────────────────────
    X_tr_ohe, X_va_ohe, X_te_ohe, ohe_vec = build_ohe_features(
        train_df, val_df, test_df)

    # ── 4-class labels (for clustering) ───────────────────────────────────────
    y4_train, y4_val, y4_test, le = encode_labels(train_df, val_df, test_df)

    # ── Lexical (for Model B) ─────────────────────────────────────────────────
    lex_train = lexical_features(train_df)
    lex_val   = lexical_features(val_df)
    lex_test  = lexical_features(test_df)

    # ── Persist ───────────────────────────────────────────────────────────────
    save_pickle((X_train, X_val, X_test),           "binary_features.pkl")
    save_pickle((y_train, y_val, y_test),            "binary_labels.pkl")
    save_pickle((train_bin, val_bin, test_bin),      "binary_dfs.pkl")
    save_pickle(tfidf_vec,                           "tfidf_vectorizer.pkl")
    save_pickle(scaler,                              "dense_scaler.pkl")
    save_pickle((X_tr_ohe, X_va_ohe, X_te_ohe),     "ohe_matrices.pkl")
    save_pickle(ohe_vec,                             "ohe_vectorizer.pkl")
    save_pickle((y4_train, y4_val, y4_test, le),     "labels.pkl")
    save_pickle((lex_train, lex_val, lex_test),      "lexical_features.pkl")
    save_pickle((train_df, val_df, test_df),         "cleaned_dfs.pkl")

    print("\nPreprocessing complete.")
    print(f"  Binary train samples : {X_train.shape[0]:,}  features: {X_train.shape[1]}")


if __name__ == "__main__":
    run_preprocessing()
