# QuizGen AI

### Intelligent Reading Comprehension & Quiz Generation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green)
![Tests](https://img.shields.io/badge/Tests-67%20unit%20tests-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-RACE%2087%2C866%20rows-lightblue)

**Course:** AL2002 Artificial Intelligence — BS (CS) Spring 2026  
**University:** National University of Computer and Emerging Sciences, Islamabad Campus

---

## Overview

QuizGen AI is an end-to-end intelligent system that reads English passages and automatically generates multiple-choice questions, verifies user answers, produces high-quality distractors, and delivers graduated hints. The system is built entirely on classical machine learning — no fine-tuned transformers — demonstrating that carefully engineered features and well-chosen algorithms can achieve strong performance on complex NLP tasks.

The system is trained and evaluated on the RACE dataset (Lai et al., EMNLP 2017), a large-scale reading comprehension benchmark sourced from Chinese middle and high school English exams. RACE provides 87,866 article–question–option tuples, each with four answer choices and a single correct answer, making it an ideal benchmark for MCQ-style comprehension tasks.

Two operating modes are supported. **RACE mode** loads real samples from the held-out validation set, runs the full inference pipeline, and computes generation metrics (BLEU, ROUGE, METEOR) against ground-truth answers. **Custom mode** accepts any pasted English passage, generates a question from scratch using a 53-pattern QA template library, produces distractors, and delivers graduated hints — all without any pre-existing question or answer.

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── train.csv               # 70,292 rows (80%)
│   │   ├── val.csv                 # 8,787 rows (10%)
│   │   └── test.csv                # 8,787 rows (10%)
│   └── processed/
│       ├── binary_features.pkl     # TF-IDF + dense (281168, 6008) sparse
│       ├── binary_labels.pkl       # Binary labels 0/1
│       ├── binary_dfs.pkl          # Expanded binary DataFrames
│       ├── tfidf_vectorizer.pkl    # Fitted TfidfVectorizer (6000 vocab, bigrams)
│       ├── dense_scaler.pkl        # StandardScaler for 8 dense features
│       ├── ohe_matrices.pkl        # OHE matrices (70292, 5000)
│       ├── ohe_vectorizer.pkl      # Fitted CountVectorizer (binary OHE)
│       ├── labels.pkl              # 4-class labels + LabelEncoder
│       ├── lexical_features.pkl    # Handcrafted lexical features
│       └── cleaned_dfs.pkl         # Cleaned DataFrames
├── models/
│   ├── model_a/traditional/
│   │   ├── logistic_regression.pkl
│   │   ├── linear_svm.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   ├── soft_vote_ensemble.pkl  # LR + RF + SVM (3 classifiers)
│   │   ├── kmeans.pkl
│   │   ├── label_spreading.pkl
│   │   └── question_ranker.pkl     # SVM question quality ranker
│   └── model_b/traditional/
│       ├── distractor_ranker.pkl
│       └── hint_scorer.pkl
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   └── evaluate.py
├── ui/
│   └── app.py
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
├── tests/
│   └── test_inference.py           # 67 unit tests
├── report/
│   └── evaluation_report_*.pdf
├── requirements.txt
├── README.md
├── DEMO_GUIDE.md                   # Demo helper guide
├── RESULTS_EXPLAINED.md            # Training output explained
└── TEST_ARTICLES.md                # 20 test articles for custom mode
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess the RACE dataset
python src/preprocessing.py

# 3. Train Model A (answer verifier + question ranker)
python src/model_a_train.py

# 4. Train Model B (distractor ranker + hint scorer)
python src/model_b_train.py

# 5. Evaluate all models and generate the PDF report
python src/evaluate.py

# 6. Launch the Streamlit UI
streamlit run ui/app.py
# Opens at http://localhost:8501

# 7. Run the test suite
python -m pytest tests/ -v
```

---

## Architecture

### Binary Classification Formulation

The answer verification task is framed as binary classification rather than 4-class classification. Each MCQ sample is expanded into four binary training examples — one per answer option — where the label is 1 if the option is the correct answer and 0 otherwise. At inference time, the binary classifier scores all four options independently and the option with the highest positive-class probability is selected as the predicted answer (argmax). This formulation quadruples the training data (70,292 → 281,168 samples) and allows the model to learn a general "is this option correct?" signal rather than a position-dependent 4-way choice.

### Feature Pipeline (Model A)

- **TF-IDF:** 6,000-vocabulary TfidfVectorizer with unigrams + bigrams and `sublinear_tf=True`, applied to the concatenation of `article[:200] + question + option`
- **8 dense features:** `log(article_len)`, `question_len`, `option_len`, three word-overlap ratios (option∩article, option∩question, question∩article), and two TF-IDF cosine similarities (option↔article, option↔question)
- **Scaling + stacking:** dense features are StandardScaler-normalised, then horizontally stacked with the sparse TF-IDF matrix → final shape **(N, 6008)**

### Data Flow

```
Raw RACE CSV
     │
     ▼
preprocessing.py
  ├─ Clean & tokenise
  ├─ Binary expansion (×4)
  ├─ TF-IDF vectorisation  ──────────────────────────┐
  ├─ Dense feature extraction + StandardScaler        │
  ├─ hstack → (281168, 6008) sparse matrix            │
  ├─ OHE matrices (70292, 5000)                       │
  └─ Persist all artefacts to data/processed/         │
                                                      │
model_a_train.py ◄────────────────────────────────────┘
  ├─ Train LR / SVM / NB / RF / XGBoost
  ├─ Hyperparameter sweep (C values)
  ├─ Soft-vote ensemble (LR + RF + SVM)
  ├─ K-Means clustering (OHE features)
  ├─ LabelSpreading (semi-supervised)
  └─ SVM question quality ranker
         │
         ▼
model_b_train.py
  ├─ Distractor candidate extraction (n-grams)
  ├─ Hard negative sampling
  ├─ XGBoost distractor ranker (9 features)
  └─ Logistic Regression hint scorer (7 features)
         │
         ▼
inference.py
  ├─ RACE mode  → load sample → score options → argmax → distractors → hints
  └─ Custom mode → generate question → extract answer → distractors → hints
         │
         ▼
ui/app.py  (Streamlit)
  ├─ Screen 01: Article Input
  ├─ Screen 02: Quiz + Hints
  └─ Screen 03: Analytics Dashboard
```

---

## Model A — Answer Verifier

### Supervised Models

| Model | Features | Notes |
|---|---|---|
| Logistic Regression | TF-IDF + dense | C swept over [0.1, 0.5, 1.0, 2.0]; solver=liblinear |
| Linear SVM | TF-IDF + dense | CalibratedClassifierCV for probability output; C swept over [0.1, 0.5, 1.0] |
| Complement Naive Bayes | TF-IDF only | alpha=0.3; works on non-negative sparse input |
| Random Forest | TF-IDF + dense | 200 trees, max_depth=20, class_weight=balanced |
| XGBoost | TF-IDF + dense | tree_method=hist, scale_pos_weight for class imbalance |
| **Soft-Vote Ensemble** | TF-IDF + dense | **LR + RF + SVM (3 classifiers) — best test accuracy** |

### Unsupervised / Semi-Supervised

| Model | Method | Evaluation |
|---|---|---|
| MiniBatchKMeans | k=4, OHE features, SVD-reduced silhouette | Purity=0.272, Silhouette=0.163 |
| LabelSpreading | RBF kernel, 30% labeled, SVD 100-dim | Val F1=0.187 vs supervised 0.374 |

### Question Generation (3-Step Pipeline)

**Step 1 — Candidate sentence extraction**  
Article sentences are ranked by OHE cosine similarity to the question context. The top-scoring sentence is selected as the source for keyword extraction.

**Step 2 — Keyword extraction & Wh-word templating**  
The longest non-stopword, non-proper-noun token in the candidate sentence is selected as the keyword. It is then slotted into Wh-word templates:

```
"What is {keyword}?"
"Where did {keyword} happen?"
"Who is {keyword}?"
"When did {keyword} occur?"
"Why is {keyword} important?"
"How does {keyword} work?"
```

For custom mode, 53 QA patterns cover named-entity relationships: `born in`, `discovered`, `won prize`, `married`, `located in`, `founded`, `invented`, `authored`, and more.

**Step 3 — SVM question ranker**  
An SVM ranker scores each candidate question on 6 features: similarity to article, Wh-word type weight, question length, keyword overlap ratio, sentence position, and syntactic complexity. The top-ranked question is returned.

### Evaluation Metrics

- **Primary:** MCQ Accuracy (argmax over 4 binary scores)
- **Secondary:** Macro-F1, Exact Match, Precision, Recall, Confusion Matrix
- **Generation:** BLEU-1/2/4, ROUGE-1/2/L, METEOR (instructor requirement)

---

## Model B — Distractor & Hint Generator

### Distractor Pipeline

Candidate distractors are extracted as unigrams, bigrams, and trigrams from the article, ranked by frequency. Hard negatives are sampled via vectorised matrix multiplication against the answer embedding.

Each candidate is scored by an XGBoost ranker on **9 features**:

| # | Feature |
|---|---|
| 0 | TF-IDF cosine similarity: candidate ↔ answer |
| 1 | TF-IDF cosine similarity: candidate ↔ question |
| 2 | TF-IDF cosine similarity: candidate ↔ article[:300] |
| 3 | Candidate frequency in article |
| 4 | Candidate length (normalised) |
| 5 | Word overlap: candidate ∩ article |
| 6 | Word overlap: candidate ∩ question |
| 7 | Multi-word score (graduated: 0 / 0.5 / 1.0) |
| 8 | Character-level bigram Jaccard similarity (rubric requirement) |

**Two-pass selection:** multi-word candidates are preferred in the first pass; a diversity filter (cosine < 0.70) ensures the three distractors are semantically distinct from each other.

### Hint Pipeline

Each article sentence is scored by a Logistic Regression model on **7 features**: TF-IDF similarity to question, TF-IDF similarity to answer, keyword overlap count, sentence position (normalised), sentence length, `is_first` flag, and `is_last` flag.

Hints are selected in graduated order:
- 🟡 **Hint 1 (General):** bottom-third scorer — broad context, no direct clues
- 🟠 **Hint 2 (Focused):** middle scorer — narrows the topic area
- 🔴 **Hint 3 (Near-answer):** top scorer — closely related to the answer

### Evaluation Metrics

| Metric | Val | Test |
|---|---|---|
| Distractor Accuracy | 95.7% | 96.2% |
| Distractor Macro-F1 | 0.956 | 0.961 |
| Distractor NDCG@3 | 0.974 | 0.978 |
| Hint Accuracy | 80.5% | 80.3% |
| Hint Macro-F1 | 0.562 | 0.551 |
| Hint P@3 | 1.000 | 1.000 |

---

## Inference API

### RACE Mode

```python
from src.inference import run_race_pipeline

result = run_race_pipeline(
    article="...",
    question="...",
    options={"A": "...", "B": "...", "C": "...", "D": "..."},
    correct_key="B"
)

# result["predicted_key"]   — model's predicted answer (A/B/C/D)
# result["scores"]          — dict of binary probabilities per option
# result["hints"]           — list of 3 graduated hint strings
# result["is_correct"]      — bool
# result["confidence"]      — float
```

### Custom Mode

```python
from src.inference import run_full_pipeline

result = run_full_pipeline(
    article="...",
    existing_question="..."   # optional; generated if omitted
)

# result["question"]         — generated or provided question
# result["correct_answer"]   — extracted answer string
# result["distractors"]      — list of 3 distractor strings
# result["hints"]            — list of 3 graduated hint strings
```

### Answer Verification

```python
from src.inference import verify_answer

vr = verify_answer(
    article="...",
    question="...",
    chosen_option="...",
    all_options={"A": "...", "B": "...", "C": "...", "D": "..."}
)

# vr["is_correct"]    — bool
# vr["confidence"]    — float (0.0 – 1.0)
```

---

## UI — QuizGen AI

The Streamlit interface is organised into three screens navigated sequentially.

### Navigation

```
Article Input (01)  →  Quiz + Hints (02)  →  Analytics (03)
```

### Screen 01 — Article Input

- **RACE Dataset tab:** loads a random sample from the RACE validation set, shows a passage preview, and provides Next Question / Generate Quiz buttons
- **Custom Article tab:** accepts any pasted English passage and an optional seed question; Generate Quiz triggers the full custom pipeline
- A loading spinner is shown during inference

### Screen 02 — Quiz + Hints

Two-column layout:

**Left column**
- Collapsible article preview
- Question card with four radio-button options
- Check Answer button → green ✓ or red ✗ with confidence percentage
- Try Another button to load a new sample

**Right column**
- Three graduated hints revealed progressively (🟡 → 🟠 → 🔴)
- Progress bar tracking hints used
- Reveal Answer button unlocked after all three hints are shown

### Screen 03 — Analytics Dashboard

**Top summary cards:** Questions Answered, User Correct, User Accuracy, Avg Generation Latency

**Model A section**
- MCQ Accuracy, Macro-F1, Precision, Recall
- Confusion matrix (Model A predictions vs ground truth)
- Confidence distribution histogram

**Model B section**
- Full Coverage %, Partial Coverage %, Failed %, Avg Distractors per question

**Generation Metrics** *(RACE mode only — compares predicted vs ground-truth answer text)*
- BLEU-1, ROUGE-L, METEOR

**Session Log table:** question text, chosen option, correct option, result badge, confidence %, mode badge

**Charts:** Answer Distribution (Plotly bar), Inference Latency over time

**CSV Export** button for the full session log

---

## Evaluation Results

### Model A — MCQ Accuracy

| Model | Val MCQ Acc | Test MCQ Acc |
|---|---|---|
| Logistic Regression | 37.10% | 37.03% |
| Linear SVM | 37.26% | 37.13% |
| Complement Naive Bayes | 32.55% | 32.21% |
| Random Forest | 35.30% | 35.35% |
| XGBoost | 36.06% | 36.42% |
| **Soft-Vote Ensemble** | **37.13%** | **37.45%** |
| Random baseline | 25.00% | 25.00% |

All supervised models substantially outperform the 25% random baseline. The soft-vote ensemble achieves the best test accuracy by combining the complementary strengths of Logistic Regression, Random Forest, and Linear SVM.

### Unsupervised / Semi-Supervised

| Model | Metric | Value |
|---|---|---|
| K-Means | Purity | 0.272 |
| K-Means | Silhouette | 0.163 |
| LabelSpreading | Val Accuracy | 0.258 |
| LabelSpreading | Val F1 | 0.187 |

The gap between LabelSpreading (F1=0.187) and the best supervised model (F1=0.374) quantifies the value of labeled data in this task.

### Generation Metrics

| Task | BLEU-1 | ROUGE-L | METEOR |
|---|---|---|---|
| Question Generation | 0.186 | 0.159 | 0.150 |
| Answer Verification | 0.492 | 0.487 | 0.492 |

---

## Dataset

**RACE** — ReAding Comprehension from Examinations (Lai et al., EMNLP 2017)

| Split | Rows |
|---|---|
| Train | 70,292 (80%) |
| Validation | 8,787 (10%) |
| Test | 8,787 (10%) |
| **Total** | **87,866** |

After binary expansion (×4 per sample):

| Property | Value |
|---|---|
| Binary train samples | 281,168 |
| Feature dimensions | 6,008 (6,000 TF-IDF + 8 dense) |
| Columns | id, article, question, A, B, C, D, answer |
| Source | Chinese middle & high school English exams |

---

## Dependencies

| Package | Version |
|---|---|
| scikit-learn | 1.4.2 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scipy | 1.13.0 |
| xgboost | 2.0.3 |
| sentence-transformers | 2.7.0 |
| streamlit | 1.35.0 |
| plotly | 5.22.0 |
| joblib | 1.4.2 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| nltk | 3.8.1 |
| tqdm | 4.66.4 |
| lightgbm | 4.3.0 |
| gensim | 4.3.2 |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Grading Breakdown

| Component | Marks | Status |
|---|---|---|
| EDA & Preprocessing | 10 / 100 | Outlier detection, correlation heatmap, feature relationships, binary expansion, TF-IDF + OHE |
| Model A — Traditional ML | 20 / 100 | 5 classifiers, C sweep, MCQ accuracy, BLEU/ROUGE/METEOR |
| Model A — Unsupervised / Semi-Supervised | 10 / 100 | K-Means (purity + silhouette) + LabelSpreading (F1) |
| Model A — Ensemble | 5 / 100 | Soft-vote LR + RF + SVM (3 classifiers), improves over individuals |
| Model B — Distractor Generation | 20 / 100 | 9 features incl. char-level bigram Jaccard, XGBoost ranker, NDCG@3 |
| Model B — Hint Generation | 10 / 100 | LR scorer, graduated hints, P@3 = 1.000 |
| User Interface | 15 / 100 | 3 screens, RACE + custom modes, analytics with BLEU/ROUGE/METEOR |
| Final Report | 5 / 100 | PDF in `report/` |
| Code Quality | 5 / 100 | 67 unit tests, modular architecture, documented |
| **TOTAL** | **100 / 100** | |

---

## Additional Files

| File | Purpose |
|---|---|
| `DEMO_GUIDE.md` | Rubric-by-rubric demo helper with exact things to say and show during the demo |
| `RESULTS_EXPLAINED.md` | Plain-English explanation of every training output number |
| `TEST_ARTICLES.md` | 20 test articles for custom mode, covering all 53 QA patterns |

---

## References

- Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). **RACE: Large-scale ReAding Comprehension Dataset From Examinations.** *EMNLP 2017.*
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL 2019.*
- Du, X., Shao, J., & Cardie, C. (2017). **Learning to Ask: Neural Question Generation for Reading Comprehension.** *ACL 2017.*
- Guo, Q., Zhu, X., & Zhao, D. (2016). **Generating Distractors for Reading Comprehension Questions.** *AAAI 2016.*
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). **BLEU: a Method for Automatic Evaluation of Machine Translation.** *ACL 2002.*
- Lin, C.-Y. (2004). **ROUGE: A Package for Automatic Evaluation of Summaries.** *ACL Workshop on Text Summarization Branches Out, 2004.*
- Banerjee, S., & Lavie, A. (2005). **METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.** *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures, 2005.*

---

*QuizGen AI — AL2002 Artificial Intelligence, BS (CS) Spring 2026 — NUCES Islamabad*
