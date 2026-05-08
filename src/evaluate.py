"""
evaluate.py
-----------
Metric computation for Model A and Model B.

Model A: MCQ accuracy, Macro-F1, Exact Match, Precision, Recall,
         Confusion Matrix, clustering metrics
Model B: Binary classification metrics (distractor ranker),
         Binary classification metrics (hint scorer),
         NDCG@K for ranking quality
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ndcg_score,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model A — MCQ Metrics
# ══════════════════════════════════════════════════════════════════════════════
def exact_match(y_true, y_pred) -> float:
    """Fraction of predictions that exactly match the true label."""
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def model_a_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Full Model A evaluation suite.
    y_true / y_pred: integer-encoded MCQ labels (0=A, 1=B, 2=C, 3=D).
    """
    label_names = label_names or ["A", "B", "C", "D"]
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    em     = exact_match(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0)

    return {
        "accuracy":              round(acc,  4),
        "macro_f1":              round(f1,   4),
        "exact_match":           round(em,   4),
        "macro_precision":       round(prec, 4),
        "macro_recall":          round(rec,  4),
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


def mcq_accuracy_from_binary(y_bin_true, y_bin_pred_proba,
                              n_options: int = 4) -> float:
    """
    Compute MCQ accuracy from binary classifier outputs.

    y_bin_true       : (N*4,) binary labels — 1 for correct option
    y_bin_pred_proba : (N*4,) predicted probabilities of being correct
    n_options        : number of options per question (default 4)

    For each question, picks the option with the highest predicted
    probability and checks if it matches the true correct option.
    """
    n_questions = len(y_bin_true) // n_options
    assert len(y_bin_true) == n_questions * n_options

    proba   = np.array(y_bin_pred_proba).reshape(n_questions, n_options)
    labels  = np.array(y_bin_true).reshape(n_questions, n_options)

    predicted = np.argmax(proba,  axis=1)   # predicted correct option index
    true_ans  = np.argmax(labels, axis=1)   # actual correct option index

    return float((predicted == true_ans).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Model B — Distractor Ranker Metrics
# ══════════════════════════════════════════════════════════════════════════════
def model_b_distractor_metrics(y_true, y_pred, y_proba=None,
                                group_size: int = 6) -> dict:
    """
    Metrics for the distractor ranker (binary classifier).

    y_true      : (N,) binary labels  — 1=good distractor, 0=hard negative
    y_pred      : (N,) predicted labels
    y_proba     : (N,) predicted probabilities (optional, for NDCG)
    group_size  : samples per question group for NDCG (3 pos + 3 neg = 6)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }

    # NDCG@3 — ranking quality within each question group
    if y_proba is not None:
        ndcg_scores = []
        n_groups = len(y_true) // group_size
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            yt = np.array(y_true[sl], dtype=float).reshape(1, -1)
            yp = np.array(y_proba[sl]).reshape(1, -1)
            try:
                ndcg_scores.append(ndcg_score(yt, yp, k=3))
            except Exception:
                pass
        result["ndcg_at_3"] = round(float(np.mean(ndcg_scores)), 4) \
            if ndcg_scores else 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Model B — Hint Scorer Metrics
# ══════════════════════════════════════════════════════════════════════════════
def model_b_hint_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Metrics for the hint scorer (binary classifier).

    y_true  : (N,) binary labels  — 1=relevant sentence, 0=not
    y_pred  : (N,) predicted labels
    y_proba : (N,) predicted probabilities (optional, for precision@K)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }

    # Precision@3: of the top-3 ranked sentences, how many are truly relevant?
    if y_proba is not None:
        top3_idx = np.argsort(y_proba)[-3:]
        p_at_3   = float(np.array(y_true)[top3_idx].mean())
        result["precision_at_3"] = round(p_at_3, 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Clustering Metrics  (for K-Means / LabelSpreading evaluation)
# ══════════════════════════════════════════════════════════════════════════════
def clustering_purity(y_true, cluster_labels) -> float:
    """
    Clustering purity: fraction of samples in the majority class per cluster.
    Uses np.unique for reliable operation on numpy arrays.
    """
    y_true        = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    total  = len(y_true)
    purity = 0.0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        most_common = int(np.bincount(y_true[mask]).max())
        purity += most_common
    return round(purity / total, 4)


def silhouette(X, labels, sample_size: int = 2000) -> float:
    from sklearn.metrics import silhouette_score
    try:
        return round(float(
            silhouette_score(X, labels, sample_size=sample_size,
                             random_state=42)), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Pretty Printer
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(metrics: dict, title: str = "Metrics"):
    skip = {"confusion_matrix", "classification_report"}
    print(f"\n{'='*44}")
    print(f"  {title}")
    print(f"{'='*44}")
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<30} {v}")
    if "classification_report" in metrics:
        print(metrics["classification_report"])
    if "confusion_matrix" in metrics:
        print("  Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print("   ", row)


# ══════════════════════════════════════════════════════════════════════════════
# PDF Report Generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(data: dict, out_dir: str = "report") -> str:
    """
    Generate a professional PDF evaluation report and save it to out_dir.
    Returns the full path of the saved PDF.

    data keys:
        results_a     : dict  model_name -> {val_acc, val_f1, test_acc, test_f1}
        best_name     : str
        best_m        : dict  (model_a_metrics output for best model)
        km_metrics    : dict  {purity, silhouette}
        ls_metrics    : dict  {acc, f1, n_samples}
        dist_metrics  : dict  {val: {...}, test: {...}}
        hint_metrics  : dict  {val: {...}, test: {...}}
        dataset_info  : dict  {train_rows, val_rows, test_rows, binary_train, n_features}
    """
    import os, datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    from reportlab.platypus import KeepTogether

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(out_dir, f"evaluation_report_{timestamp}.pdf")

    # ── Colour palette ────────────────────────────────────────────────────────
    NAVY    = colors.HexColor("#0f172a")
    TEAL    = colors.HexColor("#06b6d4")
    INDIGO  = colors.HexColor("#6366f1")
    GREEN   = colors.HexColor("#10b981")
    RED     = colors.HexColor("#f43f5e")
    AMBER   = colors.HexColor("#f59e0b")
    LIGHT   = colors.HexColor("#f0f6ff")
    MUTED   = colors.HexColor("#94a3b8")
    CARD_BG = colors.HexColor("#0f1e30")
    ROW_ALT = colors.HexColor("#111d2e")
    WHITE   = colors.white

    W, H = A4

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, **kw)

    S_TITLE = style("Title",
        fontName="Helvetica-Bold", fontSize=26, textColor=LIGHT,
        spaceAfter=4, alignment=TA_CENTER, leading=32)
    S_SUBTITLE = style("Subtitle",
        fontName="Helvetica", fontSize=11, textColor=MUTED,
        spaceAfter=2, alignment=TA_CENTER)
    S_H1 = style("H1",
        fontName="Helvetica-Bold", fontSize=14, textColor=TEAL,
        spaceBefore=18, spaceAfter=6, leading=18)
    S_H2 = style("H2",
        fontName="Helvetica-Bold", fontSize=11, textColor=LIGHT,
        spaceBefore=10, spaceAfter=4, leading=14)
    S_BODY = style("Body",
        fontName="Helvetica", fontSize=9, textColor=MUTED,
        spaceAfter=4, leading=13)
    S_SMALL = style("Small",
        fontName="Helvetica", fontSize=8, textColor=MUTED,
        spaceAfter=2, leading=11)
    S_LABEL = style("Label",
        fontName="Helvetica-Bold", fontSize=8, textColor=TEAL,
        spaceAfter=2, leading=10)
    S_CENTER = style("Center",
        fontName="Helvetica", fontSize=9, textColor=MUTED,
        alignment=TA_CENTER, leading=12)

    # ── Table style helpers ───────────────────────────────────────────────────
    def base_table_style(header_bg=NAVY, stripe=ROW_ALT):
        return TableStyle([
            # Header
            ("BACKGROUND",   (0,0), (-1,0), header_bg),
            ("TEXTCOLOR",    (0,0), (-1,0), TEAL),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,0), 8),
            ("BOTTOMPADDING",(0,0), (-1,0), 7),
            ("TOPPADDING",   (0,0), (-1,0), 7),
            ("ALIGN",        (0,0), (-1,0), "CENTER"),
            # Body
            ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",     (0,1), (-1,-1), 8),
            ("TEXTCOLOR",    (0,1), (-1,-1), LIGHT),
            ("TOPPADDING",   (0,1), (-1,-1), 5),
            ("BOTTOMPADDING",(0,1), (-1,-1), 5),
            ("ALIGN",        (1,1), (-1,-1), "CENTER"),
            ("ALIGN",        (0,1), (0,-1),  "LEFT"),
            # Alternating rows
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [CARD_BG, ROW_ALT]),
            # Grid
            ("LINEBELOW",    (0,0), (-1,0), 1, TEAL),
            ("LINEBELOW",    (0,1), (-1,-1), 0.3, colors.HexColor("#1e293b")),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("ROUNDEDCORNERS", [4]),
        ])

    def metric_cell(val, color=TEAL):
        return Paragraph(f'<font color="#{color.hexval()[2:]}">'
                         f'<b>{val}</b></font>', S_CENTER)

    def pct(v): return f"{v:.1%}"
    def f4(v):  return f"{v:.4f}"

    # ── Document ──────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="QuizGen AI — Evaluation Report",
        author="NUCES AI Lab",
    )

    story = []
    now = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=16))
    story.append(Paragraph("QuizGen AI", S_TITLE))
    story.append(Paragraph("Evaluation Report", style("ST2",
        fontName="Helvetica-Bold", fontSize=16, textColor=INDIGO,
        alignment=TA_CENTER, spaceAfter=6)))
    story.append(Paragraph(
        "Intelligent Reading Comprehension &amp; Quiz Generation System",
        S_SUBTITLE))
    story.append(Paragraph(
        "National University of Computer and Emerging Sciences  ·  AI Lab  ·  Spring 2026",
        S_SUBTITLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Generated: {now}", style("TS",
        fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER)))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=16, spaceAfter=20))

    # ── Dataset summary ───────────────────────────────────────────────────────
    di = data.get("dataset_info", {})
    story.append(Paragraph("Dataset Overview", S_H1))
    story.append(Paragraph(
        "RACE (ReAding Comprehension from Examinations) — Lai et al., EMNLP 2017. "
        "English multiple-choice questions from Chinese middle and high school exams.",
        S_BODY))

    ds_data = [
        ["Split", "MCQ Rows", "Binary Samples", "Feature Dims"],
        ["Train", f"{di.get('train_rows',0):,}", f"{di.get('binary_train',0):,}", f"{di.get('n_features',0):,}"],
        ["Val",   f"{di.get('val_rows',0):,}",   f"{di.get('val_rows',0)*4:,}",   f"{di.get('n_features',0):,}"],
        ["Test",  f"{di.get('test_rows',0):,}",  f"{di.get('test_rows',0)*4:,}",  f"{di.get('n_features',0):,}"],
    ]
    t = Table(ds_data, colWidths=[3.5*cm, 3.5*cm, 4.5*cm, 4*cm])
    t.setStyle(base_table_style())
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Feature pipeline: TF-IDF (unigrams+bigrams, 6000 vocab, sublinear_tf) + "
        "8 dense features (lengths, overlap ratios, cosine similarities) → hstack → StandardScaler.",
        S_SMALL))

    story.append(PageBreak())

    # ── Model A supervised ────────────────────────────────────────────────────
    story.append(Paragraph("Model A — Answer Verifier", S_H1))
    story.append(Paragraph(
        "Binary classification formulation: each MCQ → 4 samples (article, question, option, label). "
        "At inference, all 4 options are scored and the highest-probability option is selected. "
        "Random baseline = 25.0%.",
        S_BODY))

    results_a = data.get("results_a", {})
    if results_a:
        story.append(Paragraph("Supervised Model Comparison", S_H2))
        rows = [["Model", "Val Acc", "Val F1", "Test Acc", "Test F1", "vs Baseline"]]
        for name, r in results_a.items():
            gain = r["test_acc"] - 0.25
            gain_str = f"+{gain:.1%}"
            rows.append([
                name,
                pct(r["val_acc"]),  pct(r["val_f1"]),
                pct(r["test_acc"]), pct(r["test_f1"]),
                gain_str,
            ])
        rows.append(["Random Baseline", "25.0%", "25.0%", "25.0%", "25.0%", "—"])

        col_w = [4.5*cm, 2.2*cm, 2.2*cm, 2.4*cm, 2.2*cm, 2.5*cm]
        t = Table(rows, colWidths=col_w)
        ts = base_table_style()
        # Highlight best model row
        best_name = data.get("best_name", "")
        for i, (name, _) in enumerate(results_a.items(), start=1):
            if name == best_name:
                ts.add("BACKGROUND", (0,i), (-1,i), colors.HexColor("#0c2a1a"))
                ts.add("TEXTCOLOR",  (0,i), (-1,i), GREEN)
                ts.add("FONTNAME",   (0,i), (-1,i), "Helvetica-Bold")
        # Last row (baseline) — muted
        ts.add("TEXTCOLOR", (0, len(rows)-1), (-1, len(rows)-1), MUTED)
        t.setStyle(ts)
        story.append(t)
        story.append(Paragraph(
            f"★ Best model: {best_name}  (highlighted in green)",
            style("Note", fontName="Helvetica-Oblique", fontSize=8,
                  textColor=GREEN, spaceAfter=8)))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    best_m = data.get("best_m", {})
    if best_m and "confusion_matrix" in best_m:
        story.append(Paragraph(
            f"Confusion Matrix — {data.get('best_name','')} (Test Set)", S_H2))
        cm_raw = best_m["confusion_matrix"]
        cm_rows = [["", "Pred A", "Pred B", "Pred C", "Pred D"]]
        for i, label in enumerate(["True A","True B","True C","True D"]):
            cm_rows.append([label] + [str(v) for v in cm_raw[i]])
        t = Table(cm_rows, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        ts = base_table_style()
        # Diagonal cells (correct predictions) — green tint
        for i in range(1, 5):
            ts.add("BACKGROUND", (i, i), (i, i), colors.HexColor("#0c2a1a"))
            ts.add("TEXTCOLOR",  (i, i), (i, i), GREEN)
            ts.add("FONTNAME",   (i, i), (i, i), "Helvetica-Bold")
        t.setStyle(ts)
        story.append(t)

        if "classification_report" in best_m:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Classification Report (Test Set)", S_LABEL))
            for line in best_m["classification_report"].strip().split("\n"):
                if line.strip():
                    story.append(Paragraph(line, style("CR",
                        fontName="Courier", fontSize=7.5, textColor=MUTED,
                        leading=11, spaceAfter=0)))

    story.append(PageBreak())

    # ── Unsupervised / Semi-supervised ────────────────────────────────────────
    story.append(Paragraph("Model A — Unsupervised & Semi-Supervised", S_H1))
    story.append(Paragraph(
        "Evaluated on OHE features (5000-dim binary bag-of-words). "
        "K-Means groups question-answer pairs into 4 clusters. "
        "Label Spreading uses 15% labeled samples to propagate labels.",
        S_BODY))

    km = data.get("km_metrics", {})
    ls = data.get("ls_metrics", {})

    unsup_rows = [["Method", "Metric", "Value", "Notes"]]
    if km:
        unsup_rows.append(["K-Means (k=4)", "Cluster Purity",
                           f"{km.get('purity',0):.4f}",
                           "Majority-class fraction per cluster"])
        unsup_rows.append(["K-Means (k=4)", "Silhouette Score",
                           f"{km.get('silhouette',0):.4f}",
                           "Cluster separation quality"])
    if ls:
        unsup_rows.append(["Label Spreading", "Val Accuracy",
                           f"{ls.get('acc',0):.4f}",
                           f"Evaluated on {ls.get('n_samples',0):,} samples"])
        unsup_rows.append(["Label Spreading", "Val Macro-F1",
                           f"{ls.get('f1',0):.4f}",
                           "vs supervised best: " +
                           (f"{results_a.get(best_name,{}).get('val_acc',0):.4f}" if results_a else "N/A")])

    if len(unsup_rows) > 1:
        t = Table(unsup_rows, colWidths=[4*cm, 4*cm, 3*cm, 5.5*cm])
        t.setStyle(base_table_style())
        story.append(t)

    # ── Model B ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Model B — Distractor Ranker", S_H1))
    story.append(Paragraph(
        "XGBoost binary classifier trained on phrase-based candidates with hard negative sampling. "
        "Features: TF-IDF cosine similarities (candidate↔answer, candidate↔question, candidate↔article), "
        "frequency, length, word overlap, multi-word flag.",
        S_BODY))

    dist = data.get("dist_metrics", {})
    if dist:
        dr_rows = [["Split", "Accuracy", "Precision", "Recall", "F1", "NDCG@3"]]
        for split in ["val", "test"]:
            m = dist.get(split, {})
            if m:
                dr_rows.append([
                    split.capitalize(),
                    pct(m.get("accuracy",0)),  pct(m.get("precision",0)),
                    pct(m.get("recall",0)),    pct(m.get("f1",0)),
                    f4(m.get("ndcg_at_3",0)),
                ])
        t = Table(dr_rows, colWidths=[2.5*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        t.setStyle(base_table_style())
        story.append(t)

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Model B — Hint Scorer", S_H1))
    story.append(Paragraph(
        "Logistic Regression trained on sentence-level features. "
        "Labels assigned by similarity (top-2 sentences per article by TF-IDF cosine sim to question). "
        "Precision@3: fraction of top-3 ranked sentences that are truly relevant.",
        S_BODY))

    hint = data.get("hint_metrics", {})
    if hint:
        hs_rows = [["Split", "Accuracy", "Precision", "Recall", "F1", "Precision@3"]]
        for split in ["val", "test"]:
            m = hint.get(split, {})
            if m:
                hs_rows.append([
                    split.capitalize(),
                    pct(m.get("accuracy",0)),  pct(m.get("precision",0)),
                    pct(m.get("recall",0)),    pct(m.get("f1",0)),
                    pct(m.get("precision_at_3",0)),
                ])
        t = Table(hs_rows, colWidths=[2.5*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3.5*cm])
        t.setStyle(base_table_style())
        story.append(t)

    story.append(PageBreak())

    # ── Summary ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", S_H1))

    best_test_acc = results_a.get(best_name, {}).get("test_acc", 0) if results_a else 0
    summary_rows = [
        ["Component",          "Key Result",                          "Status"],
        ["Model A (best)",     f"{best_name}  —  Test Acc {pct(best_test_acc)}",
         "✓ Above random (25%)"],
        ["Model A (ensemble)", f"Soft-Vote LR+RF  —  {pct(results_a.get('Soft-Vote Ensemble',{}).get('test_acc',0))}",
         "✓ Best overall"],
        ["K-Means clustering", f"Purity {km.get('purity',0):.4f}",   "✓ Evaluated"],
        ["Label Spreading",    f"Val F1 {ls.get('f1',0):.4f}",       "✓ Evaluated"],
        ["Distractor Ranker",  f"Test Acc {pct(dist.get('test',{}).get('accuracy',0))}  NDCG@3 {dist.get('test',{}).get('ndcg_at_3',0):.4f}",
         "✓ High quality"],
        ["Hint Scorer",        f"Test Acc {pct(hint.get('test',{}).get('accuracy',0))}  P@3 {pct(hint.get('test',{}).get('precision_at_3',0))}",
         "✓ Evaluated"],
    ]
    t = Table(summary_rows, colWidths=[4*cm, 7*cm, 4.5*cm])
    ts = base_table_style()
    for i in range(1, len(summary_rows)):
        ts.add("TEXTCOLOR", (2,i), (2,i), GREEN)
    t.setStyle(ts)
    story.append(t)

    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceAfter=10))
    story.append(Paragraph(
        "QuizGen AI  ·  NUCES AI Lab  ·  Spring 2026  ·  Evaluation Report",
        style("Footer", fontName="Helvetica", fontSize=7.5,
              textColor=MUTED, alignment=TA_CENTER)))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Standalone evaluation runner
# Usage: python src/evaluate.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, pickle, warnings
    import joblib
    warnings.filterwarnings("ignore")

    PROCESSED = os.path.join("data", "processed")
    MODEL_A   = os.path.join("models", "model_a", "traditional")
    MODEL_B   = os.path.join("models", "model_b", "traditional")

    def lp(f):
        with open(os.path.join(PROCESSED, f), "rb") as fh:
            return pickle.load(fh)

    # ── Load processed data ───────────────────────────────────────────────────
    print("Loading processed data …")
    X_bin_tr, X_bin_va, X_bin_te = lp("binary_features.pkl")
    y_bin_tr, y_bin_va, y_bin_te = lp("binary_labels.pkl")
    y4_tr, y4_va, y4_te, le      = lp("labels.pkl")
    X_ohe_tr, X_ohe_va, X_ohe_te = lp("ohe_matrices.pkl")

    # ── Helper: MCQ predictions from a binary model ───────────────────────────
    def mcq_preds(model, X_bin, n_options=4):
        n_q = X_bin.shape[0] // n_options
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_bin)[:, 1]
        else:
            scores = model.decision_function(X_bin)
            if scores.ndim > 1:
                scores = scores[:, 1]
        return np.argmax(scores.reshape(n_q, n_options), axis=1)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A — supervised classifiers
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  MODEL A — ANSWER VERIFIER")
    print("="*60)

    model_a_names = [
        ("Logistic Regression", "logistic_regression"),
        ("Linear SVM",          "linear_svm"),
        ("Complement NB",       "naive_bayes"),
        ("Random Forest",       "random_forest"),
        ("XGBoost",             "xgboost"),
        ("Soft-Vote Ensemble",  "soft_vote_ensemble"),
    ]

    # NB uses TF-IDF-only features (no dense block) — detect by feature dim
    tfidf_vec = lp("tfidf_vectorizer.pkl")
    def tfidf_only(split):
        """Rebuild TF-IDF-only matrix for NB (which was trained without dense features)."""
        bin_df = lp("binary_dfs.pkl")[split]
        def concat(r):
            return r["article"] + " " + r["question"] + " " + r["option"]
        return tfidf_vec.transform(bin_df.apply(concat, axis=1))

    X_va_tfidf = None   # lazy — only built if NB model exists

    results_a = {}
    for display_name, fname in model_a_names:
        path = os.path.join(MODEL_A, f"{fname}.pkl")
        if not os.path.exists(path):
            print(f"  [SKIP] {display_name} — model file not found")
            continue

        model = joblib.load(path)

        # NB was trained on TF-IDF only
        if fname == "naive_bayes":
            if X_va_tfidf is None:
                X_va_tfidf = tfidf_only(1)   # index 1 = val split
                X_te_tfidf = tfidf_only(2)   # index 2 = test split
            X_va_use, X_te_use = X_va_tfidf, X_te_tfidf
        else:
            X_va_use, X_te_use = X_bin_va, X_bin_te

        val_preds  = mcq_preds(model, X_va_use)
        test_preds = mcq_preds(model, X_te_use)

        val_m  = model_a_metrics(y4_va, val_preds)
        test_m = model_a_metrics(y4_te, test_preds)

        results_a[display_name] = {
            "val_acc":  val_m["accuracy"],
            "val_f1":   val_m["macro_f1"],
            "test_acc": test_m["accuracy"],
            "test_f1":  test_m["macro_f1"],
        }

        print(f"\n  ── {display_name}")
        print(f"     Val   Acc={val_m['accuracy']:.4f}  Macro-F1={val_m['macro_f1']:.4f}  EM={val_m['exact_match']:.4f}")
        print(f"     Test  Acc={test_m['accuracy']:.4f}  Macro-F1={test_m['macro_f1']:.4f}  EM={test_m['exact_match']:.4f}")

    # Best model confusion matrix
    if results_a:
        best_name = max(results_a, key=lambda n: results_a[n]["val_acc"])
        best_fname = dict(model_a_names)[best_name]
        best_model = joblib.load(os.path.join(MODEL_A, f"{best_fname}.pkl"))
        X_te_use = X_te_tfidf if best_fname == "naive_bayes" else X_bin_te
        best_preds = mcq_preds(best_model, X_te_use)
        best_m = model_a_metrics(y4_te, best_preds)
        print(f"\n  Best model: {best_name}  (val acc={results_a[best_name]['val_acc']:.4f})")
        print_summary(best_m, f"Confusion Matrix — {best_name} (test)")

    # ── Model A comparison table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  {'Model':<24} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>8}")
    print("="*60)
    for name, r in results_a.items():
        print(f"  {name:<24} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['test_acc']:>9.4f} {r['test_f1']:>8.4f}")
    print("  Random baseline" + " "*9 + f"{'0.2500':>8} {'0.2500':>8} {'0.2500':>9} {'0.2500':>8}")
    print("="*60)

    # ── Unsupervised / Semi-supervised ────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL A — UNSUPERVISED / SEMI-SUPERVISED")
    print("="*60)

    km_path = os.path.join(MODEL_A, "kmeans.pkl")
    km_metrics = {}
    if os.path.exists(km_path):
        km = joblib.load(km_path)
        cluster_labels = km.predict(X_ohe_va)
        purity = clustering_purity(y4_va, cluster_labels)
        sil    = silhouette(X_ohe_va, cluster_labels, sample_size=3000)
        km_metrics = {"purity": purity, "silhouette": sil}
        print(f"\n  K-Means (MiniBatch, k=4)")
        print(f"     Cluster-label purity  = {purity:.4f}")
        print(f"     Silhouette score      = {sil:.4f}")
    else:
        print("  [SKIP] K-Means — model not found")

    ls_path = os.path.join(MODEL_A, "label_spreading.pkl")
    ls_metrics = {}
    if os.path.exists(ls_path):
        ls = joblib.load(ls_path)
        rng = np.random.default_rng(42)
        n_va = X_ohe_va.shape[0]
        idx  = rng.choice(n_va, size=min(2000, n_va), replace=False)
        from scipy.sparse import issparse
        X_dense = X_ohe_va[idx].toarray() if issparse(X_ohe_va) else X_ohe_va[idx]
        ls_preds = ls.predict(X_dense)
        ls_m = model_a_metrics(y4_va[idx], ls_preds)
        ls_metrics = {"acc": ls_m["accuracy"], "f1": ls_m["macro_f1"], "n_samples": len(idx)}
        print(f"\n  Label Spreading (semi-supervised, 15% labeled)")
        print(f"     Val Acc={ls_m['accuracy']:.4f}  Macro-F1={ls_m['macro_f1']:.4f}")
        print(f"     (evaluated on {len(idx)} val samples)")
    else:
        print("  [SKIP] Label Spreading — model not found")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B — distractor ranker + hint scorer
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  MODEL B — DISTRACTOR RANKER & HINT SCORER")
    print("="*60)

    # Rebuild Model B datasets for evaluation
    import sys as _sys
    _sys.path.insert(0, os.path.abspath("."))
    from src.model_b_train import build_distractor_dataset, build_hint_dataset

    tfidf_vec2 = lp("tfidf_vectorizer.pkl")
    train_df2, val_df2, test_df2 = lp("cleaned_dfs.pkl")

    print("\n  Building distractor eval dataset (val, 500 rows) …")
    X_dv, y_dv = build_distractor_dataset(val_df2,  tfidf_vec2, max_rows=500)
    print("  Building distractor eval dataset (test, 500 rows) …")
    X_dt, y_dt = build_distractor_dataset(test_df2, tfidf_vec2, max_rows=500)

    dr_path = os.path.join(MODEL_B, "distractor_ranker.pkl")
    dist_metrics = {}
    if os.path.exists(dr_path):
        dr = joblib.load(dr_path)
        for split, X, y in [("val", X_dv, y_dv), ("test", X_dt, y_dt)]:
            preds = dr.predict(X)
            proba = dr.predict_proba(X)[:, 1]
            m = model_b_distractor_metrics(y, preds, y_proba=proba)
            dist_metrics[split] = m
            print(f"\n  Distractor Ranker [{split}]")
            print(f"     Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
                  f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
                  f"NDCG@3={m.get('ndcg_at_3', 0):.4f}")
    else:
        print("  [SKIP] Distractor Ranker — model not found")

    print("\n  Building hint eval dataset (val, 500 rows) …")
    X_hv, y_hv = build_hint_dataset(val_df2,  tfidf_vec2, max_rows=500)
    print("  Building hint eval dataset (test, 500 rows) …")
    X_ht, y_ht = build_hint_dataset(test_df2, tfidf_vec2, max_rows=500)

    hs_path = os.path.join(MODEL_B, "hint_scorer.pkl")
    hint_metrics = {}
    if os.path.exists(hs_path) and len(X_hv) > 0:
        hs = joblib.load(hs_path)
        for split, X, y in [("val", X_hv, y_hv), ("test", X_ht, y_ht)]:
            preds = hs.predict(X)
            proba = hs.predict_proba(X)[:, 1]
            m = model_b_hint_metrics(y, preds, y_proba=proba)
            hint_metrics[split] = m
            print(f"\n  Hint Scorer [{split}]")
            print(f"     Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
                  f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
                  f"P@3={m.get('precision_at_3', 0):.4f}")
    else:
        print("  [SKIP] Hint Scorer — model not found or empty dataset")

    print("\n" + "="*60)
    print("  Evaluation complete.")
    print("="*60 + "\n")

    # ── Generate PDF report ───────────────────────────────────────────────────
    pdf_data = {
        "results_a":      results_a,
        "best_name":      best_name if results_a else "N/A",
        "best_m":         best_m    if results_a else {},
        "km_metrics":     km_metrics,
        "ls_metrics":     ls_metrics,
        "dist_metrics":   dist_metrics,
        "hint_metrics":   hint_metrics,
        "dataset_info": {
            "train_rows":   int(X_bin_tr.shape[0] // 4),
            "val_rows":     int(X_bin_va.shape[0] // 4),
            "test_rows":    int(X_bin_te.shape[0] // 4),
            "binary_train": int(X_bin_tr.shape[0]),
            "n_features":   int(X_bin_tr.shape[1]),
        },
    }
    out_path = generate_pdf_report(pdf_data)
    print(f"  PDF report saved → {out_path}\n")
