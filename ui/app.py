"""
app.py
------
Streamlit UI — Intelligent Reading Comprehension & Quiz Generation System

Screens:
  1. Article Input
  2. Question & Answer Quiz View
  3. Hint Panel
  4. Developer / Analytics Dashboard
"""

import os
import sys
import time
import random
import pickle
import pandas as pd
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RC Quiz System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "screen":          "input",
        "article":         "",
        "question":        "",
        "correct_answer":  "",
        "options":         {},          # {"A": ..., "B": ..., "C": ..., "D": ...}
        "correct_key":     "A",
        "hints":           [],
        "hints_revealed":  0,
        "answer_checked":  False,
        "chosen_option":   None,
        "result":          None,
        "session_log":     [],          # list of dicts for analytics
        "latency_log":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load all trained models and vectorizer once."""
    try:
        from src.inference import (
            get_ohe_vec, get_verifier, get_distractor_ranker,
            get_hint_scorer, get_label_encoder,
        )
        ohe_vec       = get_ohe_vec()
        verifier, _   = get_verifier()
        dist_ranker   = get_distractor_ranker()
        hint_scorer   = get_hint_scorer()
        le            = get_label_encoder()
        return ohe_vec, verifier, dist_ranker, hint_scorer, le, None
    except Exception as e:
        return None, None, None, None, None, str(e)


@st.cache_data(show_spinner=False)
def load_race_sample(n: int = 200):
    """Load a small sample for quick testing — tries split files then raw dev.csv."""
    paths = [
        os.path.join("data", "raw", "val.csv"),
        os.path.join("data", "raw", "train.csv"),
        os.path.join("dev.csv", "dev.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df.dropna(subset=["article", "question", "answer"]).head(n)
    return None


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RC Quiz System")
    st.markdown("---")
    nav = st.radio(
        "Navigate",
        ["📄 Article Input", "❓ Quiz View", "💡 Hint Panel", "📊 Analytics"],
        index=["input", "quiz", "hints", "analytics"].index(st.session_state.screen)
              if st.session_state.screen in ["input", "quiz", "hints", "analytics"] else 0,
    )
    screen_map = {
        "📄 Article Input": "input",
        "❓ Quiz View":     "quiz",
        "💡 Hint Panel":    "hints",
        "📊 Analytics":     "analytics",
    }
    st.session_state.screen = screen_map[nav]
    st.markdown("---")
    st.caption("NUCES · AI Lab Project · Spring 2026")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":
    st.header("📄 Article Input")
    st.markdown("Paste a reading passage below, or load a random sample from the RACE dataset.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎲 Load Random RACE Sample", use_container_width=True):
            sample_df = load_race_sample()
            if sample_df is not None:
                row = sample_df.sample(1).iloc[0]
                st.session_state.article  = row["article"]
                st.session_state.question = row["question"]
                # Store ground-truth options
                st.session_state._race_row = row.to_dict()
                st.success("Loaded a random RACE sample.")
            else:
                st.warning("RACE dataset not found in data/raw/val.csv. Please paste an article manually.")

    article_input = st.text_area(
        "Reading Passage",
        value=st.session_state.article,
        height=300,
        placeholder="Paste your reading passage here…",
    )
    st.session_state.article = article_input

    question_input = st.text_input(
        "Question (optional — leave blank to auto-generate)",
        value=st.session_state.question,
        placeholder="e.g. What was the main reason for…",
    )
    st.session_state.question = question_input

    st.markdown("---")
    if st.button("🚀 Submit — Generate Quiz", type="primary", use_container_width=True):
        if not st.session_state.article.strip():
            st.error("Please enter a reading passage before submitting.")
        else:
            ohe_vec, verifier, dist_ranker, hint_scorer, le, err = load_models()

            if err:
                st.error(f"Models not loaded: {err}\n\nRun `python src/preprocessing.py` "
                         f"then `python src/model_a_train.py` and `python src/model_b_train.py` first.")
            else:
                with st.spinner("Running Model A & Model B inference…"):
                    from src.inference import run_full_pipeline
                    result = run_full_pipeline(
                        st.session_state.article,
                        existing_question=st.session_state.question or None,
                    )

                st.session_state.question       = result["question"]
                st.session_state.correct_answer = result["correct_answer"]
                st.session_state.hints          = result["hints"]
                st.session_state.hints_revealed = 0
                st.session_state.answer_checked = False
                st.session_state.chosen_option  = None
                st.session_state.result         = None

                # Build options dict: shuffle correct + distractors
                options_list = [result["correct_answer"]] + result["distractors"][:3]
                random.shuffle(options_list)
                keys = ["A", "B", "C", "D"]
                st.session_state.options = {k: v for k, v in zip(keys, options_list)}
                # Find which key holds the correct answer
                for k, v in st.session_state.options.items():
                    if v == result["correct_answer"]:
                        st.session_state.correct_key = k
                        break

                # Log latency
                st.session_state.latency_log.append(result["latency_s"])

                st.success(f"Quiz generated in {result['latency_s']}s — navigate to ❓ Quiz View")
                st.session_state.screen = "quiz"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz View
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "quiz":
    st.header("❓ Question & Answer Quiz")

    if not st.session_state.question:
        st.info("No quiz loaded yet. Go to 📄 Article Input and submit a passage.")
    else:
        # Show article (collapsed)
        with st.expander("📖 Reading Passage", expanded=False):
            st.write(st.session_state.article)

        st.markdown(f"### {st.session_state.question}")
        st.markdown("---")

        opts = st.session_state.options
        chosen = st.radio(
            "Select your answer:",
            options=list(opts.keys()),
            format_func=lambda k: f"**{k}.** {opts[k]}",
            index=None,
            key="quiz_radio",
        )
        st.session_state.chosen_option = chosen

        col_check, col_hint = st.columns(2)
        with col_check:
            if st.button("✅ Check Answer", type="primary", use_container_width=True):
                if not chosen:
                    st.warning("Please select an option first.")
                else:
                    ohe_vec, verifier, dist_ranker, hint_scorer, le, err = load_models()
                    if err:
                        # Fallback: compare directly
                        is_correct = (chosen == st.session_state.correct_key)
                        confidence = 1.0
                    else:
                        from src.inference import verify_answer
                        vr = verify_answer(
                            st.session_state.article,
                            st.session_state.question,
                            opts[chosen],
                        )
                        is_correct = (chosen == st.session_state.correct_key)
                        confidence = vr["confidence"]

                    st.session_state.answer_checked = True
                    st.session_state.result = {
                        "chosen":     chosen,
                        "correct":    is_correct,
                        "confidence": confidence,
                    }

                    # Log to session
                    st.session_state.session_log.append({
                        "question":   st.session_state.question[:60],
                        "chosen":     chosen,
                        "correct_key": st.session_state.correct_key,
                        "is_correct": is_correct,
                        "confidence": round(confidence, 3),
                    })

        with col_hint:
            if st.button("💡 Get a Hint", use_container_width=True):
                st.session_state.screen = "hints"
                st.rerun()

        # Result display
        if st.session_state.answer_checked and st.session_state.result:
            res = st.session_state.result
            if res["correct"]:
                st.success(f"✅ Correct! (Confidence: {res['confidence']:.0%})\n\n"
                           f"The answer is **{st.session_state.correct_key}**: "
                           f"{opts[st.session_state.correct_key]}")
            else:
                st.error(f"❌ Incorrect. (Confidence: {res['confidence']:.0%})\n\n"
                         f"The correct answer was **{st.session_state.correct_key}**: "
                         f"{opts[st.session_state.correct_key]}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Hint Panel
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "hints":
    st.header("💡 Hint Panel")

    if not st.session_state.hints:
        st.info("No hints available. Submit a passage first.")
    else:
        hints = st.session_state.hints
        n_revealed = st.session_state.hints_revealed

        st.markdown(f"**Question:** {st.session_state.question}")
        st.markdown("---")

        hint_labels = ["🟡 Hint 1 — General clue",
                       "🟠 Hint 2 — More specific",
                       "🔴 Hint 3 — Near-explicit"]

        for i in range(min(n_revealed, len(hints))):
            with st.expander(hint_labels[i] if i < len(hint_labels) else f"Hint {i+1}", expanded=True):
                st.write(hints[i])

        if n_revealed < len(hints):
            if st.button(f"Reveal {hint_labels[n_revealed] if n_revealed < len(hint_labels) else f'Hint {n_revealed+1}'}",
                         use_container_width=True):
                st.session_state.hints_revealed += 1
                st.rerun()
        else:
            st.info("All hints revealed.")
            if st.button("🔓 Reveal Answer", type="primary", use_container_width=True):
                opts = st.session_state.options
                ck   = st.session_state.correct_key
                st.success(f"The correct answer is **{ck}**: {opts.get(ck, st.session_state.correct_answer)}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":
    st.header("📊 Developer / Analytics Dashboard")

    log = st.session_state.session_log
    lat = st.session_state.latency_log

    col1, col2, col3, col4 = st.columns(4)
    total    = len(log)
    correct  = sum(1 for r in log if r["is_correct"])
    accuracy = correct / total if total else 0.0
    avg_lat  = sum(lat) / len(lat) if lat else 0.0

    col1.metric("Total Questions", total)
    col2.metric("Correct Answers", correct)
    col3.metric("Session Accuracy", f"{accuracy:.0%}")
    col4.metric("Avg Latency (s)", f"{avg_lat:.2f}")

    st.markdown("---")

    if log:
        st.subheader("Session Log")
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)

        # Confusion-style breakdown
        st.subheader("Answer Distribution")
        import plotly.express as px
        fig = px.histogram(df_log, x="chosen", color="is_correct",
                           barmode="group", title="Chosen Options (correct vs incorrect)",
                           color_discrete_map={True: "#2ecc71", False: "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

        # Export
        csv = df_log.to_csv(index=False)
        st.download_button(
            "⬇️ Export Session Log (CSV)",
            data=csv,
            file_name="session_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No session data yet. Answer some questions to see analytics here.")

    # Latency chart
    if lat:
        st.subheader("Inference Latency")
        import plotly.graph_objects as go
        fig2 = go.Figure(go.Scatter(y=lat, mode="lines+markers",
                                    line=dict(color="#3498db")))
        fig2.update_layout(xaxis_title="Request #", yaxis_title="Latency (s)")
        st.plotly_chart(fig2, use_container_width=True)
