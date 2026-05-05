# 🧠 QuizGen AI — Intelligent Reading Comprehension & Quiz Generation System

> **BS (CS) Spring 2026 — AI Lab Project**  
> National University of Computer and Emerging Sciences, Islamabad Campus  
> Course: Artificial Intelligence (AL2002)

---

## Overview

An end-to-end AI-powered system that reads an English passage and automatically generates multiple-choice questions, verifies answers, produces plausible distractors, and delivers graduated hints — all through a professional dark-themed Streamlit interface.

The system is built entirely on **classical ML** (no fine-tuned transformers required) using the **RACE dataset** as its training corpus.

| Component | Role |
|-----------|------|
| **Model A** | Answer Verifier — binary classifier that scores each option and selects the most likely correct answer |
| **Model B** | Distractor & Hint Generator — ranks candidate phrases as distractors and extracts graduated hints |
| **UI** | Professional Streamlit app with RACE mode, custom article mode, quiz + hints on one page, and an analytics dashboard |

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Auto-generated train/val/test splits from dev.csv
│   │   ├── train.csv               # 70,292 rows  (80 % of dataset)
│   │   ├── val.csv                 # 8,787 rows   (10 %)
│   │   └── test.csv                # 8,787 rows   (10 %)
│   └── processed/                  # Feature matrices and fitted artifacts
│       ├── binary_features.pkl     # TF-IDF + dense features  (281168, 6008) sparse
│       ├── binary_labels.pkl       # Binary labels 0/1        (281168,)
│       ├── binary_dfs.pkl          # Expanded binary DataFrames
│       ├── tfidf_vectorizer.pkl    # Fitted TfidfVectorizer (6000 vocab, bigrams)
│       ├── dense_scaler.pkl        # StandardScaler for 8 dense features
│       ├── ohe_matrices.pkl        # OHE matrices (70292, 5000) — for Model B / clustering
│       ├── ohe_vectorizer.pkl      # Fitted CountVectorizer (binary OHE)
│       ├── labels.pkl              # 4-class labels A/B/C/D + LabelEncoder
│       ├── lexical_features.pkl    # 11 handcrafted lexical features
│       └── cleaned_dfs.pkl         # Cleaned train/val/test DataFrames
│
├── models/
│   ├── model_a/traditional/
│   │   ├── logistic_regression.pkl # LR binary verifier (best C from sweep)
│   │   ├── linear_svm.pkl          # Calibrated LinearSVC
│   │   ├── naive_bayes.pkl         # ComplementNB on TF-IDF only
│   │   ├── random_forest.pkl       # RF 200 trees, max_depth=20
│   │   ├── xgboost.pkl             # XGBoost binary classifier
│   │   ├── soft_vote_ensemble.pkl  # Soft-vote: LR + RF
│   │   ├── kmeans.pkl              # MiniBatchKMeans (unsupervised)
│   │   └── label_spreading.pkl     # LabelSpreading (semi-supervised)
│   └── model_b/traditional/
│       ├── distractor_ranker.pkl   # XGBoost distractor ranker
│       └── hint_scorer.pkl         # Logistic Regression hint scorer
│
├── src/
│   ├── preprocessing.py            # Data loading, cleaning, binary expansion, feature engineering
│   ├── model_a_train.py            # Model A training pipeline
│   ├── model_b_train.py            # Model B training pipeline
│   ├── inference.py                # Unified inference API (RACE mode + custom mode)
│   └── evaluate.py                 # Metric computation for both models
│
├── ui/
│   └── app.py                      # Streamlit application (single file)
│
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   └── experiments.ipynb           # Hyperparameter sweep & model comparison
│
├── tests/
│   └── test_inference.py           # 41 unit tests (pytest)
│
├── dev.csv/
│   └── dev.csv                     # Source dataset (87,866 rows, auto-split on first run)
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset — no manual download needed

The file `dev.csv/dev.csv` (87,866 rows) is already in the workspace.  
`preprocessing.py` automatically splits it **80 / 10 / 10** (stratified by answer label) and saves the splits to `data/raw/` on first run.

### 3. Preprocess

```bash
python src/preprocessing.py
```

This produces all artifacts in `data/processed/` including the binary-expanded feature matrices, TF-IDF vectorizer, dense scaler, OHE matrices, and label encoders.

### 4. Train Model A

```bash
python src/model_a_train.py
```

Trains and saves: Logistic Regression (with C sweep), Linear SVM (with C sweep), Complement Naive Bayes, Random Forest, XGBoost, Soft-Vote Ensemble, K-Means, Label Spreading.  
Prints a results table with val and test MCQ accuracy for every model.

### 5. Train Model B

```bash
python src/model_b_train.py
```

Trains and saves: XGBoost distractor ranker, Logistic Regression hint scorer.  
Prints Acc / Prec / Rec / F1 / NDCG@3 for the distractor ranker and Acc / Prec / Rec / F1 for the hint scorer.  
Ends with a qualitative sample inspection showing generated distractors and hints.

### 6. Run the UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`.

### Run tests

```bash
python -m pytest tests/ -v
```

41 tests covering preprocessing, model helpers, and evaluation metrics.

---

## Architecture

### Core design: binary classification formulation

The key insight driving Model A is the **binary reformulation** of MCQ answer selection:

```
Each MCQ row  →  4 binary samples
(article, question, option_A)  →  label 0
(article, question, option_B)  →  label 0
(article, question, option_C)  →  label 1   ← correct
(article, question, option_D)  →  label 0

Train a binary classifier P(option is correct | article, question, option)
At inference: score all 4 options, pick argmax
```

This is the correct ML framing for MCQ tasks and is what drives the accuracy improvement over the naive 4-class approach (~26% → 35–45%).

### Feature pipeline (Model A)

```
(article, question, option)
        │
        ├── TF-IDF (unigrams + bigrams, 6000 vocab, sublinear_tf)
        │   Text = article[:200] + question + option
        │
        └── 8 Dense features
            ├── log(article_length)
            ├── question_length
            ├── option_length
            ├── option ∩ article overlap ratio
            ├── option ∩ question overlap ratio
            ├── question ∩ article overlap ratio
            ├── TF-IDF cosine sim: article ↔ option
            └── TF-IDF cosine sim: question ↔ option
                    │
                    └── StandardScaler (with_mean=False, sparse-safe)
                    │
                    └── hstack → (N, 6008) sparse matrix
```

### Data flow

```
dev.csv/dev.csv  (87,866 rows)
        │
        └── preprocessing.py
                │  stratified 80/10/10 split
                │  clean_text (lowercase, strip punctuation)
                │  expand_to_binary (×4 rows per MCQ)
                │  TF-IDF + dense features + StandardScaler
                │  OHE features (for Model B / clustering)
                │
        ┌───────┴────────┐
        │                │
   Model A           Model B
   (verifier)    (distractor + hint)
        │                │
        └───────┬────────┘
                │
           inference.py
                │
           ui/app.py  (Streamlit)
```

---

## Model A — Answer Verifier

### Supervised models

| Model | Features | Notes |
|-------|----------|-------|
| Logistic Regression | TF-IDF + dense | C swept over [0.1, 0.5, 1.0, 2.0]; `liblinear` solver |
| Linear SVM | TF-IDF + dense | Calibrated with `CalibratedClassifierCV`; C swept over [0.1, 0.5, 1.0] |
| Complement Naive Bayes | TF-IDF only | `alpha=0.3`; non-negative features only |
| Random Forest | TF-IDF + dense | 200 trees, `max_depth=20`, `class_weight="balanced"` |
| XGBoost | TF-IDF + dense | `tree_method="hist"`, `scale_pos_weight` for class imbalance |
| **Soft-Vote Ensemble** | TF-IDF + dense | LR + RF soft voting — best overall model |

### Unsupervised / Semi-supervised

| Model | Method | Evaluation |
|-------|--------|------------|
| MiniBatchKMeans | 4 clusters on OHE features | Silhouette score + cluster-label mapping accuracy |
| LabelSpreading | KNN kernel, 15% labeled, 4000-row subsample | Macro-F1 vs supervised baseline |

### Evaluation metrics

- **MCQ Accuracy** — primary metric: score all 4 options, pick argmax, compare to ground truth
- **Binary Accuracy / Macro-F1** — training signal
- **Exact Match** — strict label match
- **Confusion Matrix** — for best model on test set

### Question generation

Template-based Wh-word generator (Who / What / Where / When / Why / How):
1. Extract candidate sentences from article via OHE cosine similarity to a seed phrase
2. Apply Wh-word templates to each candidate sentence
3. Rank by length (longer = more specific)

---

## Model B — Distractor & Hint Generator

### Distractor pipeline

```
Article text
    │
    ├── extract_candidate_phrases()
    │   Unigrams + bigrams + trigrams
    │   Filter: ≥4 chars, no stopwords, no pure digits, deduplicated
    │
    ├── hard_negatives()
    │   Vectorized: cand_vecs @ ref_vecs.T  (matrix multiply)
    │   Picks candidates most similar to correct answer / real distractors
    │
    ├── distractor_features()  — 8 features per candidate
    │   [0] TF-IDF cosine sim: candidate ↔ answer
    │   [1] TF-IDF cosine sim: candidate ↔ question
    │   [2] TF-IDF cosine sim: candidate ↔ article[:300]
    │   [3] Candidate frequency in article
    │   [4] Candidate length (normalised)
    │   [5] Word overlap: candidate ∩ article
    │   [6] Word overlap: candidate ∩ question
    │   [7] Is multi-word phrase (0/1)
    │   [8,9] SBERT cosine sims (optional, use_sbert=True)
    │
    ├── XGBoost ranker  →  P(good distractor)
    │
    └── Diversity filter  (cosine sim < 0.70 between selected distractors)
```

### Hint pipeline

```
Article text
    │
    ├── split_sentences()
    │   Double-space split → fallback to 25-word chunks
    │   (handles cleaned text with no punctuation)
    │
    ├── Batch TF-IDF transforms (once per article)
    │   sims_q   = q_vec_n @ sent_vecs_n.T
    │   sims_ans = ans_vec_n @ sent_vecs_n.T
    │
    ├── hint_scorer (Logistic Regression)
    │   Features: [sim_q, sim_ans, kw_overlap, position, length, is_first, is_last]
    │   Label: top-2 sentences by question similarity (not string matching)
    │
    └── Graduated selection
        Hint 1 → bottom-third probability  (general context)
        Hint 2 → middle probability        (more focused)
        Hint 3 → highest probability       (near-answer)
```

### Evaluation metrics

| Metric | Application |
|--------|-------------|
| Accuracy / Precision / Recall / F1 | Binary classification quality |
| NDCG@3 | Ranking quality — are the top-3 scored candidates actually good distractors? |
| Precision@3 | Of the top-3 ranked hint sentences, how many are truly relevant? |

### Speed optimizations

All hot loops are vectorized — no repeated `.transform()` calls inside loops:
- Per-row TF-IDF vectors computed **once**, passed as precomputed arguments
- Candidate phrases batch-transformed in a single `.transform(candidates)` call
- `hard_negatives` uses matrix multiplication (`cand_vecs @ ref_vecs.T`) instead of nested loops
- `itertuples()` instead of `iterrows()` throughout
- Combined speedup: **~20–50× faster** than the naive implementation

---

## Inference API

`src/inference.py` exposes two operating modes:

### RACE mode (`run_race_pipeline`)
Used when a real RACE sample is loaded. The original A/B/C/D options are used directly.  
Model A scores all 4 options. Model B generates hints only.

```python
from src.inference import run_race_pipeline

result = run_race_pipeline(
    article   = "...",
    question  = "...",
    options   = {"A": "...", "B": "...", "C": "...", "D": "..."},
    correct_key = "B",
)
# result["hints"], result["scores"], result["predicted_key"]
```

### Custom mode (`run_full_pipeline`)
Used when a custom article is pasted. Generates question, short-phrase answer candidates, distractors, and hints from scratch.

```python
from src.inference import run_full_pipeline

result = run_full_pipeline(article="...", existing_question="...")
# result["question"], result["correct_answer"], result["distractors"], result["hints"]
```

### Answer verification (`verify_answer`)

```python
from src.inference import verify_answer

vr = verify_answer(
    article        = "...",
    question       = "...",
    chosen_option  = "...",
    all_options    = {"A": "...", "B": "...", "C": "...", "D": "..."},
)
# vr["is_correct"], vr["confidence"], vr["predicted_label"]
```

---

## UI — QuizGen AI

Built with Streamlit. Dark navy + teal theme, Inter + Playfair Display fonts.

### Navigation flow

```
Article Input  →  Quiz + Hints  →  Analytics
    (01)              (02)            (03)
```

The Quiz button is disabled until a quiz has been generated. Analytics is always accessible.

### Article Input screen

Two tabs:

**RACE Dataset tab**
- Load a random passage from the RACE val split
- Preview article, question, and all 4 options (no answer highlighted)
- **🔄 Next Question** — cycles through other questions on the same article (if available), or loads a new article
- **🚀 Generate Quiz** — runs inference and navigates to Quiz screen

**Custom Article tab**
- Paste any English passage
- Optionally provide your own question (or leave blank to auto-generate)
- **🚀 Generate Quiz** — generates short-phrase options from the article

### Quiz + Hints screen (single page, two columns)

**Left column — Quiz**
- Reading passage in a collapsible expander
- Question displayed in a styled card with a teal left-border accent
- Radio options with hover effects
- **✅ Check Answer** — reveals colour-coded result (green correct / red incorrect) with confidence %
- **🔄 Try Another Question** — returns to Article Input

**Right column — Hints**
- Three graduated hints revealed one at a time
- 🟡 Hint 1 — General context
- 🟠 Hint 2 — More focused
- 🔴 Hint 3 — Near-answer guidance
- Progress bar showing `n/3 hints revealed`
- **🔓 Reveal Answer** appears after all hints are shown

### Analytics screen

- 4 metric cards: Questions Answered, Correct Answers, Session Accuracy, Avg Latency
- Custom HTML session log table with:
  - Truncated question text
  - Chosen / Correct option keys
  - ✓ Correct / ✗ Wrong result badges (pill-shaped, colour-coded)
  - Confidence % (green ≥60%, amber 40–60%, red <40%)
  - RACE / Custom mode badges
- Answer distribution bar chart (Plotly, dark theme)
- Inference latency area chart
- CSV export button

---

## Evaluation Summary

### Model A — MCQ Accuracy on val set

| Model | Val MCQ Acc |
|-------|-------------|
| Logistic Regression | ~35–40% |
| Linear SVM | ~35–40% |
| Complement Naive Bayes | ~30–35% |
| Random Forest | ~33–38% |
| XGBoost | ~36–42% |
| **Soft-Vote Ensemble** | **~37–43%** |
| Random baseline | 25.0% |

> Exact numbers depend on the random seed used during the 80/10/10 split.

### Model B — Distractor Ranker

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~96% | ~96% |
| F1 | ~0.96 | ~0.96 |
| NDCG@3 | ~0.97 | ~0.98 |

### Model B — Hint Scorer

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~80% | ~80% |
| F1 | ~0.56 | ~0.55 |

---

## Dataset

**RACE** (ReAding Comprehension from Examinations) — Lai et al., EMNLP 2017

| Property | Value |
|----------|-------|
| Total rows | 87,866 |
| Train split | 70,292 (80%) |
| Val split | 8,787 (10%) |
| Test split | 8,787 (10%) |
| Binary train samples | 281,168 (×4 expansion) |
| Feature dimensions | 6,008 (6,000 TF-IDF + 8 dense) |
| Question types | Multiple-choice (A / B / C / D) |
| Source | Chinese middle & high school English exams |
| Language | English |

Columns: `id, article, question, A, B, C, D, answer`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.4.2 | All ML models, feature engineering, metrics |
| pandas | 2.2.2 | Data loading and manipulation |
| numpy | 1.26.4 | Numerical operations |
| scipy | 1.13.0 | Sparse matrices, hstack |
| xgboost | 2.0.3 | XGBoost classifier (Model A + Model B) |
| sentence-transformers | 2.7.0 | Optional SBERT features (Model B) |
| streamlit | 1.35.0 | Web UI |
| plotly | 5.22.0 | Analytics charts |
| joblib | 1.4.2 | Model serialisation |
| matplotlib | 3.9.0 | EDA notebook plots |
| seaborn | 0.13.2 | EDA notebook plots |

---

## Grading Breakdown

| Component | Marks | Key Criteria |
|-----------|-------|--------------|
| EDA & Preprocessing | 10 / 100 | Binary expansion, TF-IDF + dense features, stratified splits |
| Model A — Traditional ML | 15 / 100 | ≥2 models, C sweep, MCQ accuracy metric |
| Model A — Unsupervised / Semi-Supervised | 20 / 100 | K-Means (silhouette + purity) + LabelSpreading (F1) |
| Model A — Ensemble | 05 / 100 | Soft-vote LR + RF, improves over individual models |
| Model B — Distractor Gen. | 15 / 100 | Phrase extraction, hard negatives, XGBoost ranker, NDCG@3 |
| Model B — Hint Gen. | 10 / 100 | Similarity-based labeling, scorer used in inference, graduated hints |
| User Interface | 15 / 100 | All screens, RACE + custom modes, quiz + hints on one page |
| Final Report | 05 / 100 | Methodology, results, discussion, limitations |
| Code Quality | 05 / 100 | Modular, documented, 41 passing unit tests |
| **TOTAL** | **100 / 100** | |

---

## References

- Lai et al. (2017). *RACE: Large-scale ReAding Comprehension Dataset From Examinations.* EMNLP 2017.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
- Du et al. (2017). *Learning to Ask: Neural Question Generation for Reading Comprehension.* ACL 2017.
- Guo et al. (2016). *Generating Distractors for Reading Comprehension Questions.* AAAI 2016.
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
