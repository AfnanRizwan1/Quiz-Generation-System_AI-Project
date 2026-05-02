"""
preprocessing.py
----------------
Dataset loading, cleaning, feature engineering, and train/val/test split management.

Primary feature representation: One-Hot Encoding (OHE)
Optional: TF-IDF vectorization (set USE_TFIDF=True)
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
USE_TFIDF = False          # Set True to also build TF-IDF matrices (optional)
MAX_VOCAB = 5000           # OHE vocabulary cap to keep memory manageable
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── 1. Loading ─────────────────────────────────────────────────────────────────
def load_splits():
    """Load train / val / test CSVs from data/raw/."""
    train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(RAW_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
    print(f"Loaded  train={train.shape}  val={val.shape}  test={test.shape}")
    return train, val, test


# ── 2. Cleaning ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to all text columns."""
    df = df.copy()
    for col in ["article", "question", "A", "B", "C", "D"]:
        df[col] = df[col].apply(clean_text)
    return df


# ── 3. Handcrafted Lexical Features ───────────────────────────────────────────
def lexical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build a matrix of handcrafted features per row:
      - article length (words)
      - question length (words)
      - option lengths (A-D)
      - keyword overlap: question ∩ article (normalised)
      - correct-option overlap with article (normalised)
    """
    rows = []
    for _, r in df.iterrows():
        art_words  = set(r["article"].split())
        q_words    = set(r["question"].split())
        overlap_qa = len(q_words & art_words) / (len(q_words) + 1e-9)

        # overlap of each option with article
        opt_overlaps = []
        for opt in ["A", "B", "C", "D"]:
            opt_words = set(r[opt].split())
            opt_overlaps.append(len(opt_words & art_words) / (len(opt_words) + 1e-9))

        row_feats = [
            len(r["article"].split()),
            len(r["question"].split()),
            len(r["A"].split()),
            len(r["B"].split()),
            len(r["C"].split()),
            len(r["D"].split()),
            overlap_qa,
            *opt_overlaps,
        ]
        rows.append(row_feats)
    return np.array(rows, dtype=np.float32)


# ── 4. One-Hot Encoding (primary) ─────────────────────────────────────────────
def build_ohe_features(train_df, val_df, test_df):
    """
    Concatenate article + question + all options into one string per row,
    then fit a binary CountVectorizer (OHE) on train and transform all splits.
    Returns sparse matrices and the fitted vectorizer.
    """
    def concat_row(r):
        return " ".join([r["article"], r["question"], r["A"], r["B"], r["C"], r["D"]])

    train_text = train_df.apply(concat_row, axis=1)
    val_text   = val_df.apply(concat_row, axis=1)
    test_text  = test_df.apply(concat_row, axis=1)

    # binary=True → One-Hot (presence/absence), not counts
    ohe_vec = CountVectorizer(max_features=MAX_VOCAB, binary=True)
    X_train = ohe_vec.fit_transform(train_text)
    X_val   = ohe_vec.transform(val_text)
    X_test  = ohe_vec.transform(test_text)

    print(f"OHE feature matrix  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return X_train, X_val, X_test, ohe_vec


# ── 5. TF-IDF (optional) ──────────────────────────────────────────────────────
def build_tfidf_features(train_df, val_df, test_df):
    """Optional TF-IDF feature matrices (same concatenation as OHE)."""
    def concat_row(r):
        return " ".join([r["article"], r["question"], r["A"], r["B"], r["C"], r["D"]])

    train_text = train_df.apply(concat_row, axis=1)
    val_text   = val_df.apply(concat_row, axis=1)
    test_text  = test_df.apply(concat_row, axis=1)

    tfidf_vec = TfidfVectorizer(max_features=MAX_VOCAB)
    X_train = tfidf_vec.fit_transform(train_text)
    X_val   = tfidf_vec.transform(val_text)
    X_test  = tfidf_vec.transform(test_text)

    print(f"TF-IDF feature matrix  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return X_train, X_val, X_test, tfidf_vec


# ── 6. Label Encoding ─────────────────────────────────────────────────────────
def encode_labels(train_df, val_df, test_df):
    """Encode answer labels A/B/C/D → 0/1/2/3."""
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["answer"])
    y_val   = le.transform(val_df["answer"])
    y_test  = le.transform(test_df["answer"])
    return y_train, y_val, y_test, le


# ── 7. Cosine Similarity Feature ──────────────────────────────────────────────
def cosine_sim_feature(ohe_vec, df: pd.DataFrame) -> np.ndarray:
    """
    For each row compute cosine similarity between the article OHE vector
    and each of the four option OHE vectors.
    Returns an (N, 4) array.
    """
    sims = []
    for _, r in df.iterrows():
        art_vec  = ohe_vec.transform([r["article"]])
        opt_sims = []
        for opt in ["A", "B", "C", "D"]:
            opt_vec = ohe_vec.transform([r[opt]])
            sim = cosine_similarity(art_vec, opt_vec)[0, 0]
            opt_sims.append(sim)
        sims.append(opt_sims)
    return np.array(sims, dtype=np.float32)


# ── 8. Save / Load helpers ────────────────────────────────────────────────────
def save_pickle(obj, filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved → {path}")


def load_pickle(filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 9. Main pipeline ──────────────────────────────────────────────────────────
def run_preprocessing():
    train_df, val_df, test_df = load_splits()

    # Clean
    train_df = clean_df(train_df)
    val_df   = clean_df(val_df)
    test_df  = clean_df(test_df)

    # Labels
    y_train, y_val, y_test, le = encode_labels(train_df, val_df, test_df)

    # OHE features
    X_train_ohe, X_val_ohe, X_test_ohe, ohe_vec = build_ohe_features(train_df, val_df, test_df)

    # Lexical features
    lex_train = lexical_features(train_df)
    lex_val   = lexical_features(val_df)
    lex_test  = lexical_features(test_df)

    # Persist
    save_pickle((X_train_ohe, X_val_ohe, X_test_ohe), "ohe_matrices.pkl")
    save_pickle(ohe_vec, "ohe_vectorizer.pkl")
    save_pickle((lex_train, lex_val, lex_test), "lexical_features.pkl")
    save_pickle((y_train, y_val, y_test, le), "labels.pkl")
    save_pickle((train_df, val_df, test_df), "cleaned_dfs.pkl")

    # Optional TF-IDF
    if USE_TFIDF:
        X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vec = build_tfidf_features(
            train_df, val_df, test_df
        )
        save_pickle((X_train_tfidf, X_val_tfidf, X_test_tfidf), "tfidf_matrices.pkl")
        save_pickle(tfidf_vec, "tfidf_vectorizer.pkl")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
