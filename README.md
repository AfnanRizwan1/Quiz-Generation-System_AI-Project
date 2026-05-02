# Intelligent Reading Comprehension and Quiz Generation System

> BS (CS) Spring 2026 — AI Lab Project  
> National University of Computer and Emerging Sciences, Islamabad Campus

## Overview

An AI-powered Reading Comprehension and Quiz Generation System built on the **RACE dataset**.  
The system integrates two ML pipelines exposed through an interactive Streamlit UI:

| Model | Role |
|-------|------|
| **Model A** | Question & Answer Generator / Verifier (Traditional ML) |
| **Model B** | Distractor & Hint Generator (Traditional ML) |
| **UI Layer** | Streamlit multi-screen application wiring both models |

---

## Project Structure

```
race_rc_project/
├── data/
│   ├── raw/                  # Original RACE CSV files (train.csv, test.csv, val.csv)
│   └── processed/            # Feature-engineered matrices (.npz / .pkl)
├── models/
│   ├── model_a/
│   │   └── traditional/      # Pickled sklearn models for Model A
│   └── model_b/
│       └── traditional/      # Pickled sklearn / word2vec models for Model B
├── src/
│   ├── preprocessing.py      # Dataset loading & feature engineering
│   ├── model_a_train.py      # Training script for Model A
│   ├── model_b_train.py      # Training script for Model B
│   ├── inference.py          # Unified inference API
│   └── evaluate.py           # Metric computation
├── ui/
│   ├── app.py                # Streamlit entry point
│   └── components/           # Reusable UI components
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── experiments.ipynb     # Experiment tracking
├── tests/
│   └── test_inference.py     # Unit tests
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place RACE dataset files
Download the RACE dataset from Kaggle and place CSVs in `data/raw/`:
```
data/raw/train.csv
data/raw/test.csv
data/raw/val.csv
```
Expected columns: `id, article, question, A, B, C, D, answer`

### 3. Preprocess data
```bash
python src/preprocessing.py
```

### 4. Train models
```bash
python src/model_a_train.py
python src/model_b_train.py
```

### 5. Run the UI
```bash
streamlit run ui/app.py
```

---

## Models

### Model A — Question & Answer Generator / Verifier
- **Generation**: Template-based Wh-word question generation with ML ranker (SVM / Random Forest)
- **Verification**: Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost
- **Unsupervised**: K-Means Clustering, Gaussian Mixture Models, Label Propagation
- **Features**: One-Hot Encoding (primary), cosine similarity, handcrafted lexical features
- **Metrics**: Accuracy, Macro F1, Exact Match (EM)

### Model B — Distractor & Hint Generator
- **Distractor ranking**: One-Hot Encoding + cosine similarity pipeline, Word2Vec nearest neighbours
- **Hint generation**: Extractive scoring via keyword overlap + ML-scored sentence ranker
- **Metrics**: Precision, Recall, F1, Accuracy, R² Score, Confusion Matrix

---

## UI Screens

| Screen | Description |
|--------|-------------|
| **Screen 1** | Article Input — paste/upload passage or load random RACE sample |
| **Screen 2** | Quiz View — question + 4 options, answer checking with colour-coded feedback |
| **Screen 3** | Hint Panel — graduated hints (general → specific → near-explicit) |
| **Screen 4** | Analytics Dashboard — model metrics, latency, CSV export |

---

## Grading Breakdown

| Component | Marks |
|-----------|-------|
| EDA & Preprocessing | 10 |
| Model A — Traditional ML | 15 |
| Model A — Unsupervised/Semi-Supervised | 20 |
| Model A — Ensemble | 05 |
| Model B — Distractor Gen. | 15 |
| Model B — Hint Gen. | 10 |
| User Interface | 15 |
| Final Report | 05 |
| Code Quality | 05 |
| **TOTAL** | **100** |

---

## Dataset

RACE (ReAding Comprehension from Examinations) — Lai et al., EMNLP 2017  
~28,000 passages · ~100,000 questions · Multiple-choice (A/B/C/D)  
Source: Chinese middle & high school English exams (ages 12–18)

TEST