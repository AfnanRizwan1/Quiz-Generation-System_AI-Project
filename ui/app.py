"""
app.py  —  Intelligent Reading Comprehension & Quiz Generation System
Professional Streamlit UI  |  NUCES AI Lab  |  Spring 2026
"""

import os, sys, random
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="QuizGen AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Inter font, dark-navy + teal palette, smooth transitions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');

/* ── Root variables ── */
:root {
    --navy:      #0f172a;
    --navy-mid:  #1e293b;
    --navy-card: #1e2d3d;
    --teal:      #0ea5e9;
    --teal-dark: #0284c7;
    --teal-glow: rgba(14,165,233,0.15);
    --green:     #10b981;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --text:      #e2e8f0;
    --text-muted:#94a3b8;
    --border:    rgba(148,163,184,0.12);
    --radius:    14px;
    --shadow:    0 4px 24px rgba(0,0,0,0.35);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}
.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1100px;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.topbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--teal);
    letter-spacing: -0.5px;
    padding-top: 0.4rem;
}
.topbar-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 2px;
    padding-bottom: 0.6rem;
}
/* ── Step progress bar ── */
.step-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2.5rem;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
}
.step-circle {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 600;
    border: 2px solid var(--border);
    background: var(--navy-mid);
    color: var(--text-muted);
    flex-shrink: 0;
    transition: all 0.3s ease;
}
.step-circle.done   { background: var(--green);    border-color: var(--green);    color: #fff; }
.step-circle.active { background: var(--teal);     border-color: var(--teal);     color: #fff; box-shadow: 0 0 0 4px var(--teal-glow); }
.step-circle.locked { background: var(--navy-mid); border-color: var(--border);   color: var(--text-muted); }
.step-label { font-size: 0.78rem; font-weight: 500; color: var(--text-muted); }
.step-label.active { color: var(--teal); }
.step-label.done   { color: var(--green); }
.step-connector {
    flex: 1; height: 2px;
    background: var(--border);
    margin: 0 8px;
    transition: background 0.3s ease;
}
.step-connector.done { background: var(--green); }

/* ── Cards ── */
.card {
    background: var(--navy-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease;
}
.card:hover { border-color: rgba(14,165,233,0.25); }
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.8rem;
}
.card-body { font-size: 0.92rem; line-height: 1.7; color: var(--text); }

/* ── Question display ── */
.question-box {
    background: linear-gradient(135deg, #0f2744 0%, #0f172a 100%);
    border: 1px solid rgba(14,165,233,0.3);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.question-box::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--teal), #6366f1);
}
.question-text {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.6;
    padding-left: 0.5rem;
}

/* ── Option buttons ── */
.option-btn {
    display: flex;
    align-items: center;
    gap: 14px;
    width: 100%;
    padding: 14px 18px;
    border-radius: 10px;
    border: 1.5px solid var(--border);
    background: var(--navy-mid);
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 400;
    cursor: pointer;
    margin-bottom: 10px;
    transition: all 0.18s ease;
    text-align: left;
}
.option-btn:hover   { border-color: var(--teal); background: var(--teal-glow); color: var(--teal); }
.option-btn.selected{ border-color: var(--teal); background: var(--teal-glow); color: var(--teal); }
.option-btn.correct { border-color: var(--green); background: rgba(16,185,129,0.12); color: var(--green); }
.option-btn.wrong   { border-color: var(--red);   background: rgba(239,68,68,0.10);  color: var(--red); }
.option-key {
    width: 28px; height: 28px;
    border-radius: 6px;
    background: rgba(148,163,184,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700;
    flex-shrink: 0;
}

/* ── Hint cards ── */
.hint-card {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
    border-left: 4px solid;
    font-size: 0.88rem;
    line-height: 1.65;
    animation: fadeSlide 0.35s ease;
}
.hint-1 { background: rgba(245,158,11,0.08);  border-color: #f59e0b; color: #fcd34d; }
.hint-2 { background: rgba(249,115,22,0.08);  border-color: #f97316; color: #fdba74; }
.hint-3 { background: rgba(239,68,68,0.08);   border-color: #ef4444; color: #fca5a5; }
.hint-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 6px; opacity: 0.75;
}

/* ── Result banners ── */
.result-correct {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    display: flex; align-items: center; gap: 14px;
    animation: fadeSlide 0.3s ease;
}
.result-wrong {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    display: flex; align-items: center; gap: 14px;
    animation: fadeSlide 0.3s ease;
}
.result-icon { font-size: 1.8rem; }
.result-title { font-size: 1rem; font-weight: 700; margin-bottom: 2px; }
.result-sub   { font-size: 0.82rem; opacity: 0.8; }

/* ── Metric cards ── */
.metric-card {
    background: var(--navy-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: var(--teal); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Mode badge ── */
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 50px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.5px;
}
.mode-race   { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
.mode-custom { background: rgba(14,165,233,0.12); color: var(--teal); border: 1px solid rgba(14,165,233,0.3); }

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* ── Animations ── */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeSlide 0.4s ease; }

/* ── Streamlit widget overrides ── */
.stTextArea textarea, .stTextInput input {
    background: var(--navy-mid) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow) !important;
}
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal), var(--teal-dark)) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
}
.stButton > button:not([kind="primary"]) {
    background: var(--navy-mid) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    background: var(--teal-glow) !important;
}
/* ── Nav pill buttons (top bar) ── */
div[data-testid="column"] > div > div > div > div[data-testid="stButton"] > button {
    border-radius: 50px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border: 1.5px solid var(--border) !important;
    background: transparent !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="column"] > div > div > div > div[data-testid="stButton"] > button:hover {
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    background: var(--teal-glow) !important;
}
.stRadio > div { gap: 0 !important; }
.stRadio label {
    background: var(--navy-mid) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    font-size: 0.9rem !important;
}
.stRadio label:hover { border-color: var(--teal) !important; background: var(--teal-glow) !important; }
.stTabs [data-baseweb="tab-list"] {
    background: var(--navy-mid) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--teal) !important;
    color: #fff !important;
}
.stExpander {
    background: var(--navy-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stDataFrame { border-radius: var(--radius) !important; }
[data-testid="stMetricValue"] { color: var(--teal) !important; font-family: 'Inter', sans-serif !important; }
.stAlert { border-radius: var(--radius) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "screen":          "input",
        "article":         "",
        "question":        "",
        "options":         {},
        "correct_key":     "A",
        "hints":           [],
        "hints_revealed":  0,
        "answer_checked":  False,
        "chosen_option":   None,
        "result":          None,
        "session_log":     [],
        "latency_log":     [],
        "mode":            "custom",
        "quiz_ready":      False,
        "_race_all_qs":    [],
        "_race_q_idx":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from src.inference import (
            get_tfidf_vec, get_dense_scaler, get_ohe_vec,
            get_verifier, get_distractor_ranker, get_hint_scorer,
        )
        get_tfidf_vec(); get_dense_scaler(); get_ohe_vec()
        get_verifier();  get_distractor_ranker(); get_hint_scorer()
        return True, None
    except Exception as e:
        return False, str(e)


@st.cache_data(show_spinner=False)
def load_race_sample(n: int = 300):
    for p in [os.path.join("data","raw","val.csv"),
              os.path.join("data","raw","train.csv"),
              os.path.join("dev.csv","dev.csv")]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df.dropna(subset=["article","question","answer","A","B","C","D"]).head(n)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ══════════════════════════════════════════════════════════════════════════════
def render_topbar():
    screen     = st.session_state.screen
    quiz_ready = st.session_state.quiz_ready

    # Brand (left) — pure HTML, no interaction needed
    col_brand, col_nav = st.columns([3, 2])

    with col_brand:
        st.markdown(
            '<div class="topbar-brand">🧠 QuizGen AI</div>'
            '<div class="topbar-sub">Intelligent Reading Comprehension System'
            ' &nbsp;·&nbsp; NUCES Spring 2026</div>',
            unsafe_allow_html=True,
        )

    # Nav buttons (right) — real Streamlit buttons, styled as pills via CSS
    with col_nav:
        b1, b2, b3 = st.columns(3)

        with b1:
            active_style = "nav-btn-active" if screen == "input" else ""
            if st.button("01  Article", key="nav_input",
                         use_container_width=True):
                st.session_state.screen = "input"
                st.rerun()

        with b2:
            disabled_quiz = not quiz_ready
            if st.button("02  Quiz", key="nav_quiz",
                         disabled=disabled_quiz,
                         use_container_width=True):
                st.session_state.screen = "quiz"
                st.rerun()

        with b3:
            if st.button("03  Analytics", key="nav_analytics",
                         use_container_width=True):
                st.session_state.screen = "analytics"
                st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_step_bar():
    s = st.session_state.screen
    quiz_ready = st.session_state.quiz_ready

    def step(n, label, state):
        # state: "done" | "active" | "locked"
        return f"""
        <div class="step-item">
            <div class="step-circle {state}">{("✓" if state=="done" else str(n))}</div>
            <span class="step-label {state}">{label}</span>
        </div>"""

    def connector(done):
        cls = "step-connector done" if done else "step-connector"
        return f'<div class="{cls}"></div>'

    if s == "input":
        s1, s2, s3 = "active", "locked", "locked"
        c1, c2 = False, False
    elif s == "quiz":
        s1, s2, s3 = "done", "active", "locked"
        c1, c2 = True, False
    else:  # analytics
        s1, s2, s3 = "done", "done" if quiz_ready else "locked", "active"
        c1 = True
        c2 = quiz_ready

    st.markdown(
        f'<div class="step-bar">'
        f'{step(1, "Load Article", s1)}'
        f'{connector(c1)}'
        f'{step(2, "Answer Quiz", s2)}'
        f'{connector(c2)}'
        f'{step(3, "Analytics", s3)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def nav_button(label, target, disabled=False):
    """Invisible nav trigger using a real Streamlit button."""
    if st.button(label, disabled=disabled, use_container_width=True):
        st.session_state.screen = target
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Render top chrome
# ══════════════════════════════════════════════════════════════════════════════
render_topbar()
render_step_bar()

models_ok, model_err = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":

    if not models_ok:
        st.markdown(f"""
        <div class="card" style="border-color:rgba(239,68,68,0.4);">
            <div class="card-title" style="color:#ef4444;">⚠ Models Not Loaded</div>
            <div class="card-body">
                Run the following commands first, then refresh:<br><br>
                <code>python src/preprocessing.py</code><br>
                <code>python src/model_a_train.py</code><br>
                <code>python src/model_b_train.py</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_race, tab_custom = st.tabs(["🎲  RACE Dataset", "✏️  Custom Article"])

    # ── RACE tab ──────────────────────────────────────────────────────────────
    with tab_race:
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-body" style="color:var(--text-muted);margin-bottom:1rem;">
            Load a real passage and question from the RACE benchmark dataset.
            The original multiple-choice options are used directly.
        </div>
        """, unsafe_allow_html=True)

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🎲  Load Random Sample", type="primary",
                         use_container_width=True, disabled=not models_ok):
                sample_df = load_race_sample()
                if sample_df is not None:
                    row = sample_df.sample(1).iloc[0]
                    # Store all questions for this article so we can cycle them
                    article_id = row["id"]
                    all_qs = sample_df[sample_df["id"] == article_id].reset_index(drop=True)
                    st.session_state.article           = row["article"]
                    st.session_state.question          = row["question"]
                    st.session_state.mode              = "race"
                    st.session_state._race_options     = {
                        "A": row["A"], "B": row["B"],
                        "C": row["C"], "D": row["D"],
                    }
                    st.session_state._race_correct     = row["answer"]
                    st.session_state._race_all_qs      = all_qs.to_dict("records")
                    st.session_state._race_q_idx       = 0
                    st.rerun()
                else:
                    st.error("Dataset not found.")

        with col_info:
            if st.session_state.mode == "race":
                st.markdown('<span class="mode-badge mode-race">🎯 RACE Mode Active</span>',
                            unsafe_allow_html=True)

        if st.session_state.mode == "race" and st.session_state.article:
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

            # Article preview
            st.markdown('<div class="card"><div class="card-title">Reading Passage</div>'
                        f'<div class="card-body">{st.session_state.article[:600]}'
                        f'{"…" if len(st.session_state.article) > 600 else ""}'
                        '</div></div>', unsafe_allow_html=True)

            # Question + options preview — no correct answer highlighted
            opts = st.session_state.get("_race_options", {})
            opt_html = "".join(
                f'<div style="padding:8px 12px;border-radius:8px;margin-bottom:6px;'
                f'background:rgba(30,41,59,0.8);'
                f'border:1px solid var(--border);'
                f'font-size:0.88rem;">'
                f'<strong style="color:var(--teal);">{k}.</strong>'
                f'&nbsp; {v}</div>'
                for k, v in opts.items()
            )

            # Question count indicator
            all_qs   = st.session_state.get("_race_all_qs", [])
            q_idx    = st.session_state.get("_race_q_idx", 0)
            n_qs     = len(all_qs)
            q_counter = (
                f'<span style="font-size:0.72rem;color:var(--text-muted);'
                f'margin-left:8px;">Question {q_idx + 1} of {n_qs}</span>'
                if n_qs > 1 else ""
            )

            st.markdown(
                f'<div class="card">'
                f'<div class="card-title">Question {q_counter}</div>'
                f'<div class="card-body" style="font-size:1rem;font-weight:500;margin-bottom:1rem;">'
                f'{st.session_state.question}'
                f'</div>'
                f'<div class="card-title">Options</div>'
                f'{opt_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

            # Action buttons row
            col_gen, col_regen = st.columns([3, 2])

            with col_gen:
                if st.button("🚀  Generate Quiz", type="primary",
                             use_container_width=True, disabled=not models_ok):
                    with st.spinner("Running inference…"):
                        from src.inference import run_race_pipeline
                        result = run_race_pipeline(
                            st.session_state.article,
                            st.session_state.question,
                            st.session_state._race_options,
                            st.session_state._race_correct,
                        )
                    st.session_state.options         = st.session_state._race_options
                    st.session_state.correct_key     = st.session_state._race_correct
                    st.session_state.hints           = result["hints"]
                    st.session_state.hints_revealed  = 0
                    st.session_state.answer_checked  = False
                    st.session_state.chosen_option   = None
                    st.session_state.result          = None
                    st.session_state.quiz_ready      = True
                    st.session_state.latency_log.append(result["latency_s"])
                    st.session_state.screen          = "quiz"
                    st.rerun()

            with col_regen:
                # Tooltip text depends on whether more questions exist
                regen_label = "🔄  Next Question" if n_qs > 1 else "🔄  New Question"
                if st.button(regen_label, use_container_width=True,
                             disabled=not models_ok):
                    sample_df = load_race_sample()
                    if sample_df is not None:
                        all_qs = st.session_state.get("_race_all_qs", [])
                        q_idx  = st.session_state.get("_race_q_idx", 0)

                        if len(all_qs) > 1:
                            # Cycle to the next question on the same article
                            new_idx = (q_idx + 1) % len(all_qs)
                            new_row = all_qs[new_idx]
                            st.session_state._race_q_idx   = new_idx
                        else:
                            # Pick a different article entirely
                            current_article = st.session_state.article
                            candidates = sample_df[
                                sample_df["article"] != current_article
                            ]
                            if candidates.empty:
                                candidates = sample_df
                            new_row_series = candidates.sample(1).iloc[0]
                            new_row = new_row_series.to_dict()
                            # Reload all questions for the new article
                            new_id  = new_row["id"]
                            new_all = sample_df[
                                sample_df["id"] == new_id
                            ].reset_index(drop=True)
                            st.session_state._race_all_qs = new_all.to_dict("records")
                            st.session_state._race_q_idx  = 0
                            st.session_state.article      = new_row["article"]

                        st.session_state.question      = new_row["question"]
                        st.session_state._race_options = {
                            "A": new_row["A"], "B": new_row["B"],
                            "C": new_row["C"], "D": new_row["D"],
                        }
                        st.session_state._race_correct = new_row["answer"]
                        # Reset quiz state since question changed
                        st.session_state.quiz_ready    = False
                        st.session_state.result        = None
                        st.session_state.answer_checked = False
                        st.rerun()

    # ── Custom tab ────────────────────────────────────────────────────────────
    with tab_custom:
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-body" style="color:var(--text-muted);margin-bottom:1rem;">
            Paste any English reading passage. The system will generate a question
            and multiple-choice options automatically.
        </div>
        """, unsafe_allow_html=True)

        article_val = st.session_state.article if st.session_state.mode == "custom" else ""
        question_val = st.session_state.question if st.session_state.mode == "custom" else ""

        article_input = st.text_area(
            "Reading Passage",
            value=article_val,
            height=260,
            placeholder="Paste your reading passage here…",
            label_visibility="collapsed",
        )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        question_input = st.text_input(
            "Question (optional)",
            value=question_val,
            placeholder="Optional: type your own question, or leave blank to auto-generate…",
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        if st.button("🚀  Generate Quiz", type="primary",
                     use_container_width=True, disabled=not models_ok,
                     key="gen_custom"):
            if not article_input.strip():
                st.error("Please enter a reading passage.")
            else:
                st.session_state.article  = article_input
                st.session_state.question = question_input
                st.session_state.mode     = "custom"
                with st.spinner("Generating question and options…"):
                    from src.inference import run_full_pipeline
                    result = run_full_pipeline(
                        article_input,
                        existing_question=question_input or None,
                    )
                correct_text = result["correct_answer"]
                options_list = [correct_text] + result["distractors"][:3]
                random.shuffle(options_list)
                keys = ["A", "B", "C", "D"]
                options = {k: v for k, v in zip(keys, options_list)}
                correct_key = next(k for k, v in options.items() if v == correct_text)

                st.session_state.question        = result["question"]
                st.session_state.options         = options
                st.session_state.correct_key     = correct_key
                st.session_state.hints           = result["hints"]
                st.session_state.hints_revealed  = 0
                st.session_state.answer_checked  = False
                st.session_state.chosen_option   = None
                st.session_state.result          = None
                st.session_state.quiz_ready      = True
                st.session_state.latency_log.append(result["latency_s"])
                st.session_state.screen          = "quiz"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz + Hints (same page, two columns)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "quiz":

    if not st.session_state.quiz_ready or not st.session_state.options:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">📄</div>
            <div style="font-size:1rem;color:var(--text-muted);">
                No quiz loaded yet. Go back to <strong>Article Input</strong> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        opts = st.session_state.options
        mode_badge = (
            '<span class="mode-badge mode-race">🎯 RACE Mode</span>'
            if st.session_state.mode == "race"
            else '<span class="mode-badge mode-custom">✏️ Custom Mode</span>'
        )

        # ── Two-column layout: Quiz (left) | Hints (right) ────────────────────
        col_quiz, col_hints = st.columns([3, 2], gap="large")

        # ── LEFT: Quiz ────────────────────────────────────────────────────────
        with col_quiz:
            # Article expander
            with st.expander("📖  Reading Passage", expanded=False):
                st.markdown(
                    f'<div style="font-size:0.88rem;line-height:1.75;color:var(--text);">'
                    f'{st.session_state.article}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

            # Question box
            st.markdown(f"""
            <div class="question-box fade-in">
                <div style="margin-bottom:8px;">{mode_badge}</div>
                <div class="question-text">{st.session_state.question}</div>
            </div>
            """, unsafe_allow_html=True)

            # Answer options via radio
            res = st.session_state.result
            if res:
                # Show styled result options (read-only)
                ck = st.session_state.correct_key
                for k, v in opts.items():
                    if k == ck:
                        cls = "correct"
                    elif k == res["chosen"] and not res["correct"]:
                        cls = "wrong"
                    else:
                        cls = ""
                    icon = "✓" if k == ck else ("✗" if cls == "wrong" else k)
                    st.markdown(f"""
                    <div class="option-btn {cls}">
                        <span class="option-key">{icon}</span>
                        <span>{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Result banner
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                if res["correct"]:
                    st.markdown(f"""
                    <div class="result-correct fade-in">
                        <div class="result-icon">✅</div>
                        <div>
                            <div class="result-title" style="color:var(--green);">Correct Answer!</div>
                            <div class="result-sub">Confidence: {res['confidence']:.0%} &nbsp;·&nbsp;
                            Answer: <strong>{st.session_state.correct_key}</strong> — {opts[st.session_state.correct_key]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-wrong fade-in">
                        <div class="result-icon">❌</div>
                        <div>
                            <div class="result-title" style="color:var(--red);">Incorrect</div>
                            <div class="result-sub">Correct answer: <strong>{st.session_state.correct_key}</strong>
                            — {opts[st.session_state.correct_key]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                if st.button("🔄  Try Another Question", use_container_width=True):
                    st.session_state.screen = "input"
                    st.session_state.quiz_ready = False
                    st.session_state.answer_checked = False
                    st.session_state.result = None
                    st.rerun()

            else:
                # Interactive radio
                chosen = st.radio(
                    "Select your answer:",
                    options=list(opts.keys()),
                    format_func=lambda k: f"{k}.  {opts[k]}",
                    index=None,
                    key="quiz_radio",
                    label_visibility="collapsed",
                )
                st.session_state.chosen_option = chosen

                st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
                if st.button("✅  Check Answer", type="primary", use_container_width=True):
                    if not chosen:
                        st.warning("Please select an option first.")
                    else:
                        is_correct = (chosen == st.session_state.correct_key)
                        try:
                            from src.inference import verify_answer
                            vr = verify_answer(
                                st.session_state.article,
                                st.session_state.question,
                                opts[chosen],
                                all_options=opts,
                            )
                            confidence = vr["confidence"]
                        except Exception:
                            confidence = 1.0 if is_correct else 0.0

                        st.session_state.answer_checked = True
                        st.session_state.result = {
                            "chosen":     chosen,
                            "correct":    is_correct,
                            "confidence": confidence,
                        }
                        st.session_state.session_log.append({
                            "question":    st.session_state.question[:60],
                            "chosen":      chosen,
                            "correct_key": st.session_state.correct_key,
                            "is_correct":  is_correct,
                            "confidence":  round(confidence, 3),
                            "mode":        st.session_state.mode,
                        })
                        st.rerun()

        # ── RIGHT: Hints ──────────────────────────────────────────────────────
        with col_hints:
            st.markdown("""
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.2px;
                        text-transform:uppercase;color:var(--teal);margin-bottom:1rem;">
                💡 Hint Panel
            </div>
            """, unsafe_allow_html=True)

            hints = st.session_state.hints
            n_rev = st.session_state.hints_revealed

            if not hints:
                st.markdown("""
                <div class="card" style="text-align:center;padding:2rem;">
                    <div style="font-size:1.5rem;margin-bottom:8px;">🔒</div>
                    <div style="font-size:0.85rem;color:var(--text-muted);">
                        Hints will appear here.<br>Click below to reveal them one by one.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                hint_classes = ["hint-1", "hint-2", "hint-3"]
                hint_labels  = ["General Clue", "More Specific", "Near Answer"]
                hint_icons   = ["🟡", "🟠", "🔴"]

                # Show revealed hints
                for i in range(min(n_rev, len(hints))):
                    cls   = hint_classes[i] if i < 3 else "hint-3"
                    label = hint_labels[i]  if i < 3 else f"Hint {i+1}"
                    icon  = hint_icons[i]   if i < 3 else "💡"
                    st.markdown(f"""
                    <div class="hint-card {cls} fade-in">
                        <div class="hint-label">{icon} Hint {i+1} — {label}</div>
                        {hints[i]}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

                # Reveal next hint button
                if n_rev < len(hints):
                    next_label = hint_labels[n_rev] if n_rev < 3 else f"Hint {n_rev+1}"
                    if st.button(f"Reveal {hint_icons[n_rev] if n_rev < 3 else '💡'} Hint {n_rev+1}",
                                 use_container_width=True):
                        st.session_state.hints_revealed += 1
                        st.rerun()
                else:
                    # All hints shown — reveal answer button
                    if not st.session_state.answer_checked:
                        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
                        if st.button("🔓  Reveal Answer", type="primary",
                                     use_container_width=True):
                            ck = st.session_state.correct_key
                            st.markdown(f"""
                            <div class="result-correct fade-in" style="margin-top:10px;">
                                <div class="result-icon">🔓</div>
                                <div>
                                    <div class="result-title" style="color:var(--green);">Answer Revealed</div>
                                    <div class="result-sub"><strong>{ck}</strong> — {opts[ck]}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            # Hint progress indicator
            if hints:
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                progress_html = "".join(
                    f'<div style="width:28px;height:6px;border-radius:3px;'
                    f'background:{"var(--teal)" if i < n_rev else "var(--border)"};'
                    f'transition:background 0.3s ease;"></div>'
                    for i in range(len(hints))
                )
                st.markdown(
                    f'<div style="display:flex;gap:6px;align-items:center;">'
                    f'{progress_html}'
                    f'<span style="font-size:0.72rem;color:var(--text-muted);margin-left:6px;">'
                    f'{n_rev}/{len(hints)} hints revealed</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":

    log = st.session_state.session_log
    lat = st.session_state.latency_log

    total    = len(log)
    correct  = sum(1 for r in log if r["is_correct"])
    accuracy = correct / total if total else 0.0
    avg_lat  = sum(lat) / len(lat) if lat else 0.0

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, str(total),          "Questions Answered"),
        (m2, str(correct),        "Correct Answers"),
        (m3, f"{accuracy:.0%}",   "Session Accuracy"),
        (m4, f"{avg_lat:.2f}s",   "Avg Latency"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    if log:

        col_table, col_chart = st.columns([3, 2], gap="large")

        with col_table:
            st.markdown('<div class="card-title">Session Log</div>',
                        unsafe_allow_html=True)

            # Build a styled HTML table — no raw dataframe
            def result_badge(is_correct):
                if is_correct:
                    return ('<span style="display:inline-flex;align-items:center;gap:5px;'
                            'padding:3px 10px;border-radius:50px;font-size:0.72rem;'
                            'font-weight:600;background:rgba(16,185,129,0.15);'
                            'color:#10b981;border:1px solid rgba(16,185,129,0.3);">'
                            '✓ Correct</span>')
                return ('<span style="display:inline-flex;align-items:center;gap:5px;'
                        'padding:3px 10px;border-radius:50px;font-size:0.72rem;'
                        'font-weight:600;background:rgba(239,68,68,0.12);'
                        'color:#ef4444;border:1px solid rgba(239,68,68,0.3);">'
                        '✗ Wrong</span>')

            def mode_badge(mode):
                if mode == "race":
                    return ('<span style="padding:2px 8px;border-radius:50px;'
                            'font-size:0.68rem;font-weight:600;'
                            'background:rgba(99,102,241,0.15);color:#a5b4fc;'
                            'border:1px solid rgba(99,102,241,0.25);">RACE</span>')
                return ('<span style="padding:2px 8px;border-radius:50px;'
                        'font-size:0.68rem;font-weight:600;'
                        'background:rgba(14,165,233,0.12);color:#38bdf8;'
                        'border:1px solid rgba(14,165,233,0.25);">Custom</span>')

            rows_html = ""
            for i, r in enumerate(log):
                bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
                q_text = r["question"]
                q_text = (q_text[:52] + "…") if len(q_text) > 52 else q_text
                conf_pct = f"{r['confidence']:.0%}"
                conf_color = "#10b981" if r["confidence"] >= 0.6 else "#f59e0b" if r["confidence"] >= 0.4 else "#ef4444"
                rows_html += (
                    f'<tr style="background:{bg};border-bottom:1px solid rgba(148,163,184,0.07);">'
                    f'<td style="padding:10px 12px;font-size:0.82rem;color:#cbd5e1;max-width:220px;">{q_text}</td>'
                    f'<td style="padding:10px 12px;text-align:center;">'
                    f'<span style="font-weight:700;color:#0ea5e9;font-size:0.9rem;">{r["chosen"]}</span></td>'
                    f'<td style="padding:10px 12px;text-align:center;">'
                    f'<span style="font-weight:700;color:#94a3b8;font-size:0.9rem;">{r["correct_key"]}</span></td>'
                    f'<td style="padding:10px 12px;text-align:center;">{result_badge(r["is_correct"])}</td>'
                    f'<td style="padding:10px 12px;text-align:center;'
                    f'font-size:0.82rem;font-weight:600;color:{conf_color};">{conf_pct}</td>'
                    f'<td style="padding:10px 12px;text-align:center;">{mode_badge(r.get("mode","custom"))}</td>'
                    f'</tr>'
                )

            table_html = (
                '<div style="background:var(--navy-card);border:1px solid var(--border);'
                'border-radius:var(--radius);overflow:hidden;margin-bottom:12px;">'
                '<table style="width:100%;border-collapse:collapse;">'
                '<thead>'
                '<tr style="background:rgba(14,165,233,0.08);border-bottom:1px solid rgba(14,165,233,0.2);">'
                '<th style="padding:10px 12px;text-align:left;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Question</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Chosen</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Correct</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Result</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Confidence</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:0.7rem;font-weight:700;'
                'letter-spacing:0.8px;text-transform:uppercase;color:#64748b;">Mode</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{rows_html}</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

            # Export button
            csv = pd.DataFrame(log).to_csv(index=False)
            st.download_button(
                "⬇️  Export CSV",
                data=csv,
                file_name="session_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_chart:
            st.markdown('<div class="card-title">Answer Distribution</div>',
                        unsafe_allow_html=True)
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Bar chart
                fig = px.histogram(
                    pd.DataFrame(log), x="chosen", color="is_correct",
                    barmode="group",
                    color_discrete_map={True: "#10b981", False: "#ef4444"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter",
                    font_color="#94a3b8",
                    showlegend=True,
                    legend_title_text="Correct",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=200,
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.1)")
                st.plotly_chart(fig, use_container_width=True)

                # Latency line
                if lat:
                    st.markdown('<div class="card-title" style="margin-top:1rem;">Inference Latency</div>',
                                unsafe_allow_html=True)
                    fig2 = go.Figure(go.Scatter(
                        y=lat, mode="lines+markers",
                        line=dict(color="#0ea5e9", width=2),
                        marker=dict(size=6, color="#0ea5e9"),
                        fill="tozeroy",
                        fillcolor="rgba(14,165,233,0.08)",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter",
                        font_color="#94a3b8",
                        xaxis_title="Request #",
                        yaxis_title="Seconds",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=180,
                    )
                    fig2.update_xaxes(showgrid=False)
                    fig2.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.1)")
                    st.plotly_chart(fig2, use_container_width=True)

            except ImportError:
                st.info("Install plotly for charts.")
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
            <div style="font-size:1rem;color:var(--text-muted);">
                No session data yet.<br>Answer some questions to see your analytics here.
            </div>
        </div>
        """, unsafe_allow_html=True)
