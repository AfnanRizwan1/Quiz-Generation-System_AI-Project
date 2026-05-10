"""
app.py  —  QuizGen AI  |  Premium Redesign
Intelligent Reading Comprehension & Quiz Generation System
NUCES AI Lab  |  Spring 2026
"""
import os, sys, random
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="QuizGen AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Design tokens ── */
:root {
  --bg-base:      #060b14;
  --bg-surface:   #0c1524;
  --bg-elevated:  #111d2e;
  --bg-card:      #0f1e30;
  --bg-glass:     rgba(15,30,48,0.7);
  --accent:       #06b6d4;
  --accent-2:     #6366f1;
  --accent-glow:  rgba(6,182,212,0.18);
  --accent-glow2: rgba(99,102,241,0.15);
  --green:        #10b981;
  --green-glow:   rgba(16,185,129,0.15);
  --red:          #f43f5e;
  --red-glow:     rgba(244,63,94,0.12);
  --amber:        #f59e0b;
  --text-1:       #f0f6ff;
  --text-2:       #94a3b8;
  --text-3:       #475569;
  --border:       rgba(148,163,184,0.08);
  --border-accent:rgba(6,182,212,0.2);
  --radius-sm:    8px;
  --radius:       14px;
  --radius-lg:    20px;
  --shadow-sm:    0 2px 8px rgba(0,0,0,0.3);
  --shadow:       0 4px 24px rgba(0,0,0,0.5);
  --shadow-lg:    0 8px 48px rgba(0,0,0,0.6);
  --shadow-glow:  0 0 40px rgba(6,182,212,0.12);
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-1) !important;
  -webkit-font-smoothing: antialiased;
}
.main .block-container {
  padding: 0 2.5rem 5rem !important;
  max-width: 1200px;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }

/* ══════════════════════════════════════════════════════════
   SPLASH SCREEN
══════════════════════════════════════════════════════════ */
@keyframes fadeUp   { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes spin     { to{transform:rotate(360deg)} }
@keyframes shimmer  { 0%{background-position:200% center} 100%{background-position:-200% center} }
@keyframes orb-float{ 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-18px) scale(1.04)} }
@keyframes bar-grow { 0%{width:0%} 20%{width:12%} 45%{width:40%} 70%{width:68%} 90%{width:88%} 100%{width:100%} }
@keyframes dot-pulse{ 0%,80%,100%{transform:scale(0.6);opacity:0.4} 40%{transform:scale(1);opacity:1} }

.splash {
  position:fixed; inset:0; z-index:9999;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(6,182,212,0.08) 0%, transparent 70%),
              radial-gradient(ellipse 60% 50% at 80% 100%, rgba(99,102,241,0.06) 0%, transparent 60%),
              var(--bg-base);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:2rem;
}
.splash-orb {
  width:180px; height:180px; border-radius:50%; margin-bottom:2.5rem;
  background: radial-gradient(circle at 35% 35%, rgba(6,182,212,0.35), rgba(99,102,241,0.2) 60%, transparent 80%);
  border:1px solid rgba(6,182,212,0.2);
  box-shadow: 0 0 60px rgba(6,182,212,0.15), 0 0 120px rgba(99,102,241,0.08), inset 0 0 40px rgba(6,182,212,0.05);
  display:flex; align-items:center; justify-content:center;
  animation: orb-float 4s ease-in-out infinite, fadeUp 0.8s ease both;
  font-size:4rem;
}
.splash-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:clamp(2rem,5vw,3.2rem); font-weight:700; letter-spacing:-1.5px;
  background: linear-gradient(135deg, #f0f6ff 0%, var(--accent) 50%, var(--accent-2) 100%);
  background-size:200% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation: fadeUp 0.8s 0.2s ease both, shimmer 4s linear infinite;
  margin-bottom:0.5rem;
}
.splash-tagline {
  font-size:0.9rem; color:var(--text-2); letter-spacing:0.3px; line-height:1.6;
  animation: fadeUp 0.8s 0.4s ease both; margin-bottom:3rem;
}
.splash-ring {
  width:56px; height:56px; border-radius:50%;
  border:2px solid rgba(6,182,212,0.12);
  border-top-color:var(--accent);
  animation: spin 0.85s linear infinite, fadeIn 0.5s 0.6s ease both;
  margin-bottom:2rem;
}
.splash-track {
  width:min(320px,80vw); height:3px; background:rgba(148,163,184,0.08);
  border-radius:2px; overflow:hidden; margin-bottom:1.2rem;
  animation: fadeIn 0.5s 0.7s ease both;
}
.splash-fill {
  height:100%; border-radius:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  animation: bar-grow 3.8s cubic-bezier(0.4,0,0.2,1) forwards;
}
.splash-status {
  font-size:0.78rem; color:var(--text-3); letter-spacing:0.8px; text-transform:uppercase;
  animation: fadeIn 0.5s 0.8s ease both;
}
.splash-status span { color:var(--accent); }
.splash-dots { display:inline-flex; gap:4px; margin-left:6px; vertical-align:middle; }
.splash-dots i {
  width:5px; height:5px; border-radius:50%; background:var(--accent); display:inline-block;
  animation: dot-pulse 1.4s ease-in-out infinite;
}
.splash-dots i:nth-child(2){animation-delay:0.2s}
.splash-dots i:nth-child(3){animation-delay:0.4s}

/* ══════════════════════════════════════════════════════════
   NAVBAR
══════════════════════════════════════════════════════════ */
.navbar {
  position:sticky; top:0; z-index:100;
  background: rgba(6,11,20,0.85);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom:1px solid var(--border);
  padding:0.9rem 0 0.7rem;
  margin-bottom:2rem;
}
.navbar-brand {
  font-family:'Space Grotesk',sans-serif;
  font-size:1.35rem; font-weight:700; letter-spacing:-0.5px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.navbar-sub { font-size:0.72rem; color:var(--text-3); margin-top:1px; }

/* ══════════════════════════════════════════════════════════
   STEPPER
══════════════════════════════════════════════════════════ */
.stepper { display:flex; align-items:center; margin-bottom:2.5rem; }
.step-node {
  display:flex; align-items:center; gap:10px; flex:1;
}
.step-dot {
  width:36px; height:36px; border-radius:50%; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  font-size:0.78rem; font-weight:700; transition:all 0.35s ease;
  position:relative;
}
.step-dot.locked {
  background:var(--bg-elevated); border:1.5px solid var(--border);
  color:var(--text-3);
}
.step-dot.active {
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  border:none; color:#fff;
  box-shadow:0 0 0 4px var(--accent-glow), 0 4px 16px rgba(6,182,212,0.3);
}
.step-dot.done {
  background:var(--green); border:none; color:#fff;
  box-shadow:0 0 0 3px var(--green-glow);
}
.step-text { font-size:0.78rem; font-weight:500; }
.step-text.locked { color:var(--text-3); }
.step-text.active { color:var(--accent); font-weight:600; }
.step-text.done   { color:var(--green); }
.step-line {
  flex:1; height:2px; margin:0 10px;
  background:var(--border); border-radius:1px; transition:background 0.4s ease;
}
.step-line.done { background:linear-gradient(90deg,var(--green),var(--accent)); }

/* ══════════════════════════════════════════════════════════
   CARDS & SURFACES
══════════════════════════════════════════════════════════ */
.card {
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.5rem 1.75rem;
  margin-bottom:1rem;
  box-shadow:var(--shadow-sm);
  transition:border-color 0.25s ease, box-shadow 0.25s ease;
}
.card:hover { border-color:var(--border-accent); box-shadow:var(--shadow-glow); }
.card-glass {
  background:var(--bg-glass);
  backdrop-filter:blur(16px);
  border:1px solid rgba(6,182,212,0.1);
  border-radius:var(--radius);
  padding:1.5rem 1.75rem;
  margin-bottom:1rem;
}
.section-label {
  font-size:0.65rem; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--accent); margin-bottom:0.75rem;
}
.card-body { font-size:0.9rem; line-height:1.75; color:var(--text-2); }

/* ══════════════════════════════════════════════════════════
   QUESTION CARD
══════════════════════════════════════════════════════════ */
.q-card {
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,rgba(6,182,212,0.06) 0%,rgba(99,102,241,0.04) 50%,var(--bg-card) 100%);
  border:1px solid rgba(6,182,212,0.2);
  border-radius:var(--radius-lg);
  padding:2rem 2.2rem 1.8rem;
  margin-bottom:1.5rem;
  box-shadow:0 0 0 1px rgba(6,182,212,0.05), var(--shadow);
}
.q-card::before {
  content:''; position:absolute; top:0; left:0;
  width:3px; height:100%;
  background:linear-gradient(180deg,var(--accent),var(--accent-2));
}
.q-card::after {
  content:''; position:absolute; top:-40px; right:-40px;
  width:160px; height:160px; border-radius:50%;
  background:radial-gradient(circle,rgba(6,182,212,0.06),transparent 70%);
  pointer-events:none;
}
.q-text {
  font-family:'Space Grotesk',sans-serif;
  font-size:1.1rem; font-weight:600; color:var(--text-1);
  line-height:1.65; padding-left:0.75rem;
}

/* ══════════════════════════════════════════════════════════
   OPTION BUTTONS (post-answer static display)
══════════════════════════════════════════════════════════ */
.opt {
  display:flex; align-items:center; gap:14px;
  padding:13px 18px; border-radius:var(--radius-sm);
  border:1.5px solid var(--border);
  background:var(--bg-elevated);
  margin-bottom:9px; font-size:0.88rem; color:var(--text-2);
  transition:all 0.2s ease;
}
.opt-key {
  width:30px; height:30px; border-radius:7px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  font-size:0.75rem; font-weight:700;
  background:rgba(148,163,184,0.08); color:var(--text-3);
}
.opt.correct { border-color:rgba(16,185,129,0.5); background:rgba(16,185,129,0.07); color:var(--green); }
.opt.correct .opt-key { background:var(--green); color:#fff; }
.opt.wrong   { border-color:rgba(244,63,94,0.4);  background:rgba(244,63,94,0.06);  color:var(--red); }
.opt.wrong   .opt-key { background:var(--red); color:#fff; }

/* ══════════════════════════════════════════════════════════
   RESULT BANNERS
══════════════════════════════════════════════════════════ */
.result-banner {
  display:flex; align-items:flex-start; gap:16px;
  padding:1.25rem 1.5rem; border-radius:var(--radius);
  margin-top:1rem; animation:fadeUp 0.3s ease;
}
.result-banner.correct {
  background:rgba(16,185,129,0.08);
  border:1px solid rgba(16,185,129,0.3);
  box-shadow:0 0 24px rgba(16,185,129,0.06);
}
.result-banner.wrong {
  background:rgba(244,63,94,0.07);
  border:1px solid rgba(244,63,94,0.25);
  box-shadow:0 0 24px rgba(244,63,94,0.05);
}
.result-icon { font-size:1.6rem; line-height:1; flex-shrink:0; margin-top:2px; }
.result-title { font-size:0.95rem; font-weight:700; margin-bottom:3px; }
.result-sub   { font-size:0.8rem; opacity:0.75; line-height:1.5; }

/* ══════════════════════════════════════════════════════════
   HINT PANEL
══════════════════════════════════════════════════════════ */
.hint-panel-title {
  font-size:0.65rem; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--accent); margin-bottom:1.2rem;
  display:flex; align-items:center; gap:8px;
}
.hint-panel-title::after {
  content:''; flex:1; height:1px; background:var(--border);
}
.hint-card {
  border-radius:var(--radius-sm);
  padding:1rem 1.25rem 1rem 1.1rem;
  margin-bottom:10px; border-left:3px solid;
  font-size:0.85rem; line-height:1.7;
  animation:fadeUp 0.35s ease;
  position:relative; overflow:hidden;
}
.hint-card::before {
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent);
  pointer-events:none;
}
.hint-1 { background:rgba(245,158,11,0.07);  border-color:#f59e0b; color:#fcd34d; }
.hint-2 { background:rgba(249,115,22,0.07);  border-color:#f97316; color:#fdba74; }
.hint-3 { background:rgba(244,63,94,0.07);   border-color:#f43f5e; color:#fda4af; }
.hint-meta {
  font-size:0.62rem; font-weight:700; letter-spacing:1px;
  text-transform:uppercase; opacity:0.6; margin-bottom:5px;
}
.hint-empty {
  background:var(--bg-elevated); border:1px dashed var(--border);
  border-radius:var(--radius); padding:2.5rem 1.5rem;
  text-align:center; color:var(--text-3); font-size:0.85rem;
}
.hint-progress {
  display:flex; gap:5px; align-items:center; margin-top:1.2rem;
}
.hint-pip {
  height:4px; border-radius:2px; flex:1;
  transition:background 0.4s ease, box-shadow 0.4s ease;
}
.hint-pip.filled {
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  box-shadow:0 0 8px rgba(6,182,212,0.4);
}
.hint-pip.empty { background:var(--border); }

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════════════════════ */
.metric-card {
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.4rem 1.5rem;
  text-align:center;
  transition:border-color 0.25s, box-shadow 0.25s;
  position:relative; overflow:hidden;
}
.metric-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  opacity:0.6;
}
.metric-card:hover { border-color:var(--border-accent); box-shadow:var(--shadow-glow); }
.metric-val {
  font-family:'Space Grotesk',sans-serif;
  font-size:2.2rem; font-weight:700; letter-spacing:-1px;
  background:linear-gradient(135deg,var(--text-1),var(--accent));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  line-height:1.1; margin-bottom:6px;
}
.metric-lbl {
  font-size:0.68rem; font-weight:600; letter-spacing:1.2px;
  text-transform:uppercase; color:var(--text-3);
}

/* ══════════════════════════════════════════════════════════
   BADGES
══════════════════════════════════════════════════════════ */
.badge {
  display:inline-flex; align-items:center; gap:5px;
  padding:3px 11px; border-radius:50px;
  font-size:0.68rem; font-weight:600; letter-spacing:0.4px;
}
.badge-race   { background:rgba(99,102,241,0.12); color:#a5b4fc; border:1px solid rgba(99,102,241,0.25); }
.badge-custom { background:rgba(6,182,212,0.1);   color:var(--accent); border:1px solid rgba(6,182,212,0.25); }

/* ══════════════════════════════════════════════════════════
   EMPTY STATES
══════════════════════════════════════════════════════════ */
.empty-state {
  text-align:center; padding:4rem 2rem;
  background:var(--bg-card); border:1px dashed var(--border);
  border-radius:var(--radius-lg);
}
.empty-icon { font-size:2.8rem; margin-bottom:1rem; opacity:0.5; }
.empty-title { font-size:1rem; font-weight:600; color:var(--text-2); margin-bottom:6px; }
.empty-sub   { font-size:0.82rem; color:var(--text-3); line-height:1.6; }

/* ══════════════════════════════════════════════════════════
   STREAMLIT OVERRIDES
══════════════════════════════════════════════════════════ */
.stTextArea textarea, .stTextInput input {
  background:var(--bg-elevated) !important;
  border:1.5px solid var(--border) !important;
  border-radius:var(--radius-sm) !important;
  color:var(--text-1) !important;
  font-family:'Inter',sans-serif !important;
  font-size:0.9rem !important;
  transition:border-color 0.2s, box-shadow 0.2s !important;
  padding:0.75rem 1rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color:var(--accent) !important;
  box-shadow:0 0 0 3px var(--accent-glow) !important;
  outline:none !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
  color:var(--text-3) !important;
}
.stButton > button {
  font-family:'Inter',sans-serif !important;
  font-weight:600 !important; font-size:0.85rem !important;
  border-radius:var(--radius-sm) !important;
  transition:all 0.2s ease !important;
  letter-spacing:0.2px !important;
  padding:0.6rem 1.2rem !important;
}
.stButton > button[kind="primary"] {
  background:linear-gradient(135deg,var(--accent) 0%,#0891b2 100%) !important;
  color:#fff !important; border:none !important;
  box-shadow:0 4px 16px rgba(6,182,212,0.3), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}
.stButton > button[kind="primary"]:hover {
  transform:translateY(-1px) !important;
  box-shadow:0 6px 24px rgba(6,182,212,0.4) !important;
}
.stButton > button[kind="primary"]:active { transform:translateY(0) !important; }
.stButton > button:not([kind="primary"]) {
  background:var(--bg-elevated) !important;
  color:var(--text-2) !important;
  border:1.5px solid var(--border) !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color:var(--accent) !important;
  color:var(--accent) !important;
  background:var(--accent-glow) !important;
}
.stButton > button:disabled {
  opacity:0.35 !important; cursor:not-allowed !important;
  transform:none !important;
}
/* Nav buttons */
div[data-testid="column"] > div > div > div > div[data-testid="stButton"] > button {
  border-radius:50px !important;
  font-size:0.78rem !important; font-weight:500 !important;
  padding:5px 16px !important;
  border:1.5px solid var(--border) !important;
  background:transparent !important;
  color:var(--text-3) !important;
  letter-spacing:0.3px !important;
}
div[data-testid="column"] > div > div > div > div[data-testid="stButton"] > button:hover {
  border-color:var(--accent) !important;
  color:var(--accent) !important;
  background:var(--accent-glow) !important;
}
.stRadio > div { gap:0 !important; }
.stRadio label {
  background:var(--bg-elevated) !important;
  border:1.5px solid var(--border) !important;
  border-radius:var(--radius-sm) !important;
  padding:13px 16px !important;
  margin-bottom:8px !important;
  cursor:pointer !important;
  transition:all 0.18s ease !important;
  font-size:0.88rem !important;
  color:var(--text-2) !important;
}
.stRadio label:hover {
  border-color:var(--accent) !important;
  background:var(--accent-glow) !important;
  color:var(--text-1) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background:var(--bg-elevated) !important;
  border-radius:var(--radius-sm) !important;
  padding:4px !important; gap:3px !important;
  border:1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius:6px !important;
  font-family:'Inter',sans-serif !important;
  font-weight:500 !important; font-size:0.83rem !important;
  color:var(--text-3) !important; padding:7px 18px !important;
  transition:all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,var(--accent),#0891b2) !important;
  color:#fff !important;
  box-shadow:0 2px 8px rgba(6,182,212,0.3) !important;
}
.stExpander {
  background:var(--bg-card) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius) !important;
}
.stExpander:hover { border-color:var(--border-accent) !important; }
[data-testid="stMetricValue"] { color:var(--accent) !important; }
.stAlert { border-radius:var(--radius) !important; }
.stDownloadButton > button {
  background:var(--bg-elevated) !important;
  border:1.5px solid var(--border) !important;
  color:var(--text-2) !important;
  border-radius:var(--radius-sm) !important;
  font-size:0.83rem !important;
}
.stDownloadButton > button:hover {
  border-color:var(--accent) !important;
  color:var(--accent) !important;
}

/* ── Misc ── */
.divider { height:1px; background:var(--border); margin:1.5rem 0; }
@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.fade-in { animation:fadeUp 0.35s ease; }
code {
  background:rgba(6,182,212,0.08) !important;
  color:var(--accent) !important;
  border-radius:4px !important;
  padding:2px 6px !important;
  font-size:0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "screen":               "input",
        "article":              "",
        "question":             "",
        "options":              {},
        "correct_key":          "A",
        "hints":                [],
        "hints_revealed":       0,
        "answer_checked":       False,
        "chosen_option":        None,
        "result":               None,
        "session_log":          [],
        "latency_log":          [],
        "mode":                 "custom",
        "quiz_ready":           False,
        "_race_all_qs":         [],
        "_race_q_idx":          0,
        "models_loaded":        False,
        # Model A predicted key (set when quiz is generated, used in analytics)
        "_predicted_key":       "A",
        # RACE ground-truth question (for BLEU/ROUGE/METEOR reference)
        "_race_true_question":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
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
# STARTUP SPLASH GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.models_loaded:
    st.markdown("""
    <div class="splash">
      <div class="splash-orb">🧠</div>
      <div class="splash-title">QuizGen AI</div>
      <div class="splash-tagline">
        Intelligent Reading Comprehension &amp; Quiz Generation<br>
        <span style="color:var(--text-3);">NUCES &nbsp;·&nbsp; AI Lab &nbsp;·&nbsp; Spring 2026</span>
      </div>
      <div class="splash-ring"></div>
      <div class="splash-track"><div class="splash-fill"></div></div>
      <div class="splash-status">
        <span>Initialising AI models</span>
        <span class="splash-dots"><i></i><i></i><i></i></span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    ok, err = load_models()
    st.session_state.models_loaded = True
    st.session_state._model_ok  = ok
    st.session_state._model_err = err
    st.rerun()
    st.stop()

_models_ok = st.session_state.get("_model_ok",  True)
_model_err = st.session_state.get("_model_err", None)


# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_navbar():
    screen     = st.session_state.screen
    quiz_ready = st.session_state.quiz_ready

    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    col_brand, col_nav = st.columns([3, 2])

    with col_brand:
        st.markdown(
            '<div class="navbar-brand">🧠 QuizGen AI</div>'
            '<div class="navbar-sub">Intelligent Reading Comprehension System'
            ' &nbsp;·&nbsp; NUCES Spring 2026</div>',
            unsafe_allow_html=True,
        )

    with col_nav:
        n1, n2, n3 = st.columns(3)
        with n1:
            if st.button("01  Article", key="nav_input", use_container_width=True):
                st.session_state.screen = "input"; st.rerun()
        with n2:
            if st.button("02  Quiz", key="nav_quiz",
                         disabled=not quiz_ready, use_container_width=True):
                st.session_state.screen = "quiz"; st.rerun()
        with n3:
            if st.button("03  Analytics", key="nav_analytics", use_container_width=True):
                st.session_state.screen = "analytics"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEPPER
# ══════════════════════════════════════════════════════════════════════════════
def render_stepper():
    s  = st.session_state.screen
    qr = st.session_state.quiz_ready

    if s == "input":
        s1,s2,s3 = "active","locked","locked"; c1,c2 = False,False
    elif s == "quiz":
        s1,s2,s3 = "done","active","locked";   c1,c2 = True,False
    else:
        s1,s2,s3 = "done","done" if qr else "locked","active"
        c1=True; c2=qr

    def dot(n, label, state):
        icon = "✓" if state == "done" else str(n)
        return (f'<div class="step-node">'
                f'<div class="step-dot {state}">{icon}</div>'
                f'<span class="step-text {state}">{label}</span>'
                f'</div>')

    def line(done):
        return f'<div class="step-line {"done" if done else ""}"></div>'

    st.markdown(
        f'<div class="stepper">'
        f'{dot(1,"Load Article",s1)}{line(c1)}'
        f'{dot(2,"Answer Quiz",s2)}{line(c2)}'
        f'{dot(3,"Analytics",s3)}'
        f'</div>',
        unsafe_allow_html=True,
    )


render_navbar()
render_stepper()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":

    if not _models_ok:
        st.markdown(
            '<div class="card" style="border-color:rgba(244,63,94,0.3);">'
            '<div class="section-label" style="color:var(--red);">⚠ Models Not Loaded</div>'
            '<div class="card-body">'
            + (f'<p style="color:var(--red);font-size:0.82rem;">{_model_err}</p>' if _model_err else '')
            + 'Run these commands first, then restart:<br><br>'
            '<code>python src/preprocessing.py</code><br>'
            '<code>python src/model_a_train.py</code><br>'
            '<code>python src/model_b_train.py</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    tab_race, tab_custom = st.tabs(["  🎲  RACE Dataset  ", "  ✏️  Custom Article  "])

    # ── RACE TAB ──────────────────────────────────────────────────────────────
    with tab_race:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-body" style="margin-bottom:1.25rem;">'
            'Load a real passage and question from the RACE benchmark. '
            'Original multiple-choice options are used directly.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_btn, col_badge = st.columns([1, 2])
        with col_btn:
            if st.button("🎲  Load Random Sample", type="primary",
                         use_container_width=True, disabled=not _models_ok):
                sample_df = load_race_sample()
                if sample_df is not None:
                    row = sample_df.sample(1).iloc[0]
                    article_id = row["id"]
                    all_qs = sample_df[sample_df["id"] == article_id].reset_index(drop=True)
                    st.session_state.article       = row["article"]
                    st.session_state.question      = row["question"]
                    st.session_state.mode          = "race"
                    st.session_state._race_options = {"A":row["A"],"B":row["B"],"C":row["C"],"D":row["D"]}
                    st.session_state._race_correct = row["answer"]
                    st.session_state._race_all_qs  = all_qs.to_dict("records")
                    st.session_state._race_q_idx   = 0
                    st.rerun()
                else:
                    st.error("Dataset not found.")

        with col_badge:
            if st.session_state.mode == "race":
                st.markdown('<span class="badge badge-race">🎯 RACE Mode Active</span>',
                            unsafe_allow_html=True)

        if st.session_state.mode == "race" and st.session_state.article:
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            # Article card
            art_preview = st.session_state.article[:600]
            if len(st.session_state.article) > 600:
                art_preview += "…"
            st.markdown(
                f'<div class="card">'
                f'<div class="section-label">Reading Passage</div>'
                f'<div class="card-body">{art_preview}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Question + options card
            opts = st.session_state.get("_race_options", {})
            all_qs = st.session_state.get("_race_all_qs", [])
            q_idx  = st.session_state.get("_race_q_idx", 0)
            n_qs   = len(all_qs)
            q_counter = (
                f' <span style="font-size:0.68rem;color:var(--text-3);'
                f'font-weight:400;margin-left:6px;">{q_idx+1} / {n_qs}</span>'
                if n_qs > 1 else ""
            )
            opt_rows = "".join(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'padding:10px 14px;border-radius:var(--radius-sm);margin-bottom:7px;'
                f'background:var(--bg-elevated);border:1.5px solid var(--border);'
                f'font-size:0.86rem;color:var(--text-2);">'
                f'<span style="width:26px;height:26px;border-radius:6px;flex-shrink:0;'
                f'display:flex;align-items:center;justify-content:center;'
                f'background:rgba(6,182,212,0.1);color:var(--accent);'
                f'font-size:0.72rem;font-weight:700;">{k}</span>'
                f'<span>{v}</span></div>'
                for k, v in opts.items()
            )
            st.markdown(
                f'<div class="card">'
                f'<div class="section-label">Question{q_counter}</div>'
                f'<div style="font-size:1rem;font-weight:600;color:var(--text-1);'
                f'line-height:1.6;margin-bottom:1.2rem;">{st.session_state.question}</div>'
                f'<div class="section-label">Options</div>'
                f'{opt_rows}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            col_gen, col_regen = st.columns([3, 2])

            with col_gen:
                if st.button("🚀  Generate Quiz", type="primary",
                             use_container_width=True, disabled=not _models_ok):
                    with st.spinner("Running inference…"):
                        from src.inference import run_race_pipeline
                        result = run_race_pipeline(
                            st.session_state.article, st.session_state.question,
                            st.session_state._race_options, st.session_state._race_correct,
                        )
                    st.session_state.options        = st.session_state._race_options
                    st.session_state.correct_key    = st.session_state._race_correct
                    st.session_state.hints          = result["hints"]
                    st.session_state.hints_revealed = 0
                    st.session_state.answer_checked = False
                    st.session_state.chosen_option  = None
                    st.session_state.result         = None
                    st.session_state.quiz_ready     = True
                    st.session_state.latency_log.append(result["latency_s"])
                    # Store Model A's predicted key for analytics confusion matrix
                    st.session_state._predicted_key      = result.get("predicted_key", "A")
                    # Store the RACE ground-truth question for BLEU/ROUGE/METEOR
                    st.session_state._race_true_question = st.session_state.question
                    st.session_state.screen         = "quiz"
                    st.rerun()

            with col_regen:
                regen_label = "🔄  Next Question" if n_qs > 1 else "🔄  New Question"
                if st.button(regen_label, use_container_width=True, disabled=not _models_ok):
                    sample_df = load_race_sample()
                    if sample_df is not None:
                        all_qs = st.session_state.get("_race_all_qs", [])
                        q_idx  = st.session_state.get("_race_q_idx", 0)
                        if len(all_qs) > 1:
                            new_idx = (q_idx + 1) % len(all_qs)
                            new_row = all_qs[new_idx]
                            st.session_state._race_q_idx = new_idx
                        else:
                            candidates = sample_df[sample_df["article"] != st.session_state.article]
                            if candidates.empty: candidates = sample_df
                            new_row = candidates.sample(1).iloc[0].to_dict()
                            new_all = sample_df[sample_df["id"] == new_row["id"]].reset_index(drop=True)
                            st.session_state._race_all_qs = new_all.to_dict("records")
                            st.session_state._race_q_idx  = 0
                            st.session_state.article      = new_row["article"]
                        st.session_state.question      = new_row["question"]
                        st.session_state._race_options = {"A":new_row["A"],"B":new_row["B"],"C":new_row["C"],"D":new_row["D"]}
                        st.session_state._race_correct = new_row["answer"]
                        st.session_state.quiz_ready    = False
                        st.session_state.result        = None
                        st.session_state.answer_checked = False
                        st.rerun()

    # ── CUSTOM TAB ────────────────────────────────────────────────────────────
    with tab_custom:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-body" style="margin-bottom:1.25rem;">'
            'Paste any English reading passage. The system will generate a question '
            'and multiple-choice options automatically.'
            '</div>',
            unsafe_allow_html=True,
        )

        article_val  = st.session_state.article  if st.session_state.mode == "custom" else ""
        question_val = st.session_state.question if st.session_state.mode == "custom" else ""

        st.markdown('<div class="section-label">Reading Passage</div>', unsafe_allow_html=True)
        article_input = st.text_area(
            "Reading Passage", value=article_val, height=240,
            placeholder="Paste your reading passage here…",
            label_visibility="collapsed",
        )
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Question <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3);font-size:0.72rem;">(optional — leave blank to auto-generate)</span></div>', unsafe_allow_html=True)
        question_input = st.text_input(
            "Question", value=question_val,
            placeholder="Type your own question, or leave blank…",
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        if st.button("🚀  Generate Quiz", type="primary",
                     use_container_width=True, disabled=not _models_ok, key="gen_custom"):
            if not article_input.strip():
                st.error("Please enter a reading passage.")
            else:
                st.session_state.article  = article_input
                st.session_state.question = question_input
                st.session_state.mode     = "custom"
                with st.spinner("Generating question and options…"):
                    from src.inference import run_full_pipeline
                    result = run_full_pipeline(article_input, existing_question=question_input or None)
                correct_text = result["correct_answer"]
                options_list = [correct_text] + result["distractors"][:3]
                random.shuffle(options_list)
                keys = ["A","B","C","D"]
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
                # Custom mode: no ground-truth predicted key (Model A not used for selection)
                # Store correct_key as predicted for analytics (best we can do)
                st.session_state._predicted_key      = correct_key
                st.session_state._race_true_question = ""  # no reference in custom mode
                st.session_state.screen          = "quiz"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — QUIZ + HINTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "quiz":

    if not st.session_state.quiz_ready or not st.session_state.options:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📄</div>'
            '<div class="empty-title">No Quiz Loaded</div>'
            '<div class="empty-sub">Go to Article Input and generate a quiz first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        opts = st.session_state.options
        mode_badge = (
            '<span class="badge badge-race">🎯 RACE</span>'
            if st.session_state.mode == "race"
            else '<span class="badge badge-custom">✏️ Custom</span>'
        )

        col_quiz, col_hints = st.columns([3, 2], gap="large")

        # ── QUIZ COLUMN ───────────────────────────────────────────────────────
        with col_quiz:
            with st.expander("📖  Reading Passage", expanded=False):
                st.markdown(
                    f'<div style="font-size:0.87rem;line-height:1.8;color:var(--text-2);">'
                    f'{st.session_state.article}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

            # Question card
            st.markdown(
                f'<div class="q-card fade-in">'
                f'<div style="margin-bottom:10px;">{mode_badge}</div>'
                f'<div class="q-text">{st.session_state.question}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            res = st.session_state.result

            if res:
                # Post-answer: static styled options
                ck = st.session_state.correct_key
                for k, v in opts.items():
                    if k == ck:
                        cls = "correct"; icon = "✓"
                    elif k == res["chosen"] and not res["correct"]:
                        cls = "wrong"; icon = "✗"
                    else:
                        cls = ""; icon = k
                    st.markdown(
                        f'<div class="opt {cls}">'
                        f'<span class="opt-key">{icon}</span>'
                        f'<span>{v}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Result banner
                if res["correct"]:
                    st.markdown(
                        f'<div class="result-banner correct fade-in">'
                        f'<div class="result-icon">✅</div>'
                        f'<div>'
                        f'<div class="result-title" style="color:var(--green);">Correct Answer!</div>'
                        f'<div class="result-sub">Confidence {res["confidence"]:.0%} &nbsp;·&nbsp; '
                        f'<strong>{ck}</strong> — {opts[ck]}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-banner wrong fade-in">'
                        f'<div class="result-icon">❌</div>'
                        f'<div>'
                        f'<div class="result-title" style="color:var(--red);">Incorrect</div>'
                        f'<div class="result-sub">Correct answer: '
                        f'<strong>{ck}</strong> — {opts[ck]}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                if st.button("🔄  Try Another Question", use_container_width=True):
                    sample_df = load_race_sample()
                    if st.session_state.mode == "race" and sample_df is not None:
                        all_qs = st.session_state.get("_race_all_qs", [])
                        q_idx  = st.session_state.get("_race_q_idx", 0)
                        if len(all_qs) > 1:
                            new_idx = (q_idx + 1) % len(all_qs)
                            new_row = all_qs[new_idx]
                            st.session_state._race_q_idx = new_idx
                        else:
                            candidates = sample_df[sample_df["article"] != st.session_state.article]
                            if candidates.empty: candidates = sample_df
                            new_row = candidates.sample(1).iloc[0].to_dict()
                            new_all = sample_df[sample_df["id"] == new_row["id"]].reset_index(drop=True)
                            st.session_state._race_all_qs = new_all.to_dict("records")
                            st.session_state._race_q_idx  = 0
                            st.session_state.article      = new_row["article"]
                        st.session_state.question      = new_row["question"]
                        st.session_state._race_options = {"A":new_row["A"],"B":new_row["B"],"C":new_row["C"],"D":new_row["D"]}
                        st.session_state._race_correct = new_row["answer"]
                    else:
                        st.session_state.article  = ""
                        st.session_state.question = ""
                    st.session_state.quiz_ready     = False
                    st.session_state.answer_checked = False
                    st.session_state.result         = None
                    st.session_state.options        = {}
                    st.session_state.hints          = []
                    st.session_state.hints_revealed = 0
                    st.session_state.screen         = "input"
                    st.rerun()

            else:
                # Pre-answer: interactive radio
                chosen = st.radio(
                    "Select your answer:",
                    options=list(opts.keys()),
                    format_func=lambda k: f"{k}.  {opts[k]}",
                    index=None, key="quiz_radio",
                    label_visibility="collapsed",
                )
                st.session_state.chosen_option = chosen
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

                if st.button("✅  Check Answer", type="primary", use_container_width=True):
                    if not chosen:
                        st.warning("Please select an option first.")
                    else:
                        is_correct = (chosen == st.session_state.correct_key)
                        try:
                            from src.inference import verify_answer
                            vr = verify_answer(
                                st.session_state.article, st.session_state.question,
                                opts[chosen], all_options=opts,
                            )
                            confidence = vr["confidence"]
                        except Exception:
                            confidence = 1.0 if is_correct else 0.0
                        st.session_state.answer_checked = True
                        st.session_state.result = {"chosen": chosen, "correct": is_correct, "confidence": confidence}

                        # ── Build session log entry ───────────────────────────
                        mode = st.session_state.mode

                        # n_distractors: count real (non-fallback) distractor slots
                        fallback_markers = {"[option", "[distractor", "none of the above",
                                            "None of the above"}
                        n_dist = sum(
                            1 for k, v in st.session_state.options.items()
                            if k != st.session_state.correct_key
                            and not any(v.lower().startswith(m.lower())
                                        for m in fallback_markers)
                        )

                        # For Model A confusion matrix:
                        # compare Model A's predicted key vs ground-truth correct key
                        # (not user's choice — that's user accuracy, not model accuracy)
                        predicted_key = st.session_state.get("_predicted_key", chosen)
                        pred_idx  = ["A","B","C","D"].index(predicted_key) \
                                    if predicted_key in ["A","B","C","D"] else -1
                        true_idx  = ["A","B","C","D"].index(st.session_state.correct_key) \
                                    if st.session_state.correct_key in ["A","B","C","D"] else -1

                        # BLEU/ROUGE/METEOR: only meaningful in RACE mode where we have
                        # a real ground-truth question to compare against
                        race_true_q = st.session_state.get("_race_true_question", "")

                        # For BLEU/ROUGE/METEOR: compare Model A's predicted answer text
                        # vs the RACE ground-truth correct answer text
                        predicted_key_for_bleu = st.session_state.get("_predicted_key", "")
                        predicted_ans_text = opts.get(predicted_key_for_bleu, "")
                        true_ans_text      = opts.get(st.session_state.correct_key, "")

                        st.session_state.session_log.append({
                            "question":      st.session_state.question[:60],
                            "chosen":        chosen,
                            "correct_key":   st.session_state.correct_key,
                            "is_correct":    is_correct,
                            "confidence":    round(confidence, 3),
                            "mode":          mode,
                            # Model A: predicted (by model) vs true correct
                            "chosen_idx":    pred_idx,
                            "correct_idx":   true_idx,
                            # Model B: number of real distractors generated
                            "n_distractors": n_dist,
                            # BLEU/ROUGE/METEOR: Model A predicted answer vs ground-truth answer
                            # Only populated in RACE mode (custom has no reference)
                            "generated_question": predicted_ans_text,
                            "reference_question": true_ans_text if mode == "race" else "",
                        })
                        st.rerun()

        # ── HINTS COLUMN ──────────────────────────────────────────────────────
        with col_hints:
            hints = st.session_state.hints
            n_rev = st.session_state.hints_revealed

            st.markdown(
                '<div class="hint-panel-title">💡 AI Hints</div>',
                unsafe_allow_html=True,
            )

            if not hints:
                st.markdown(
                    '<div class="hint-empty">'
                    '<div style="font-size:1.8rem;margin-bottom:10px;opacity:0.4;">🔒</div>'
                    '<div style="font-weight:600;color:var(--text-2);margin-bottom:4px;">Hints Locked</div>'
                    '<div>Reveal hints one by one to guide your thinking.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                hint_meta  = ["General Context", "More Focused", "Near Answer"]
                hint_icons = ["🟡", "🟠", "🔴"]
                hint_cls   = ["hint-1", "hint-2", "hint-3"]

                for i in range(min(n_rev, len(hints))):
                    cls  = hint_cls[i]  if i < 3 else "hint-3"
                    meta = hint_meta[i] if i < 3 else f"Hint {i+1}"
                    icon = hint_icons[i] if i < 3 else "💡"
                    st.markdown(
                        f'<div class="hint-card {cls} fade-in">'
                        f'<div class="hint-meta">{icon} Hint {i+1} — {meta}</div>'
                        f'{hints[i]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

                if n_rev < len(hints):
                    next_meta = hint_meta[n_rev] if n_rev < 3 else f"Hint {n_rev+1}"
                    next_icon = hint_icons[n_rev] if n_rev < 3 else "💡"
                    if st.button(f"Reveal {next_icon} Hint {n_rev+1}", use_container_width=True):
                        st.session_state.hints_revealed += 1
                        st.rerun()
                else:
                    if not st.session_state.answer_checked:
                        if st.button("🔓  Reveal Answer", type="primary", use_container_width=True):
                            ck = st.session_state.correct_key
                            st.markdown(
                                f'<div class="result-banner correct fade-in">'
                                f'<div class="result-icon">🔓</div>'
                                f'<div>'
                                f'<div class="result-title" style="color:var(--green);">Answer Revealed</div>'
                                f'<div class="result-sub"><strong>{ck}</strong> — {opts[ck]}</div>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

                # Progress pips
                pips = "".join(
                    f'<div class="hint-pip {"filled" if i < n_rev else "empty"}"></div>'
                    for i in range(len(hints))
                )
                st.markdown(
                    f'<div class="hint-progress">{pips}'
                    f'<span style="font-size:0.68rem;color:var(--text-3);margin-left:8px;">'
                    f'{n_rev}/{len(hints)}</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":

    log = st.session_state.session_log
    lat = st.session_state.latency_log

    total    = len(log)
    correct  = sum(1 for r in log if r["is_correct"])
    accuracy = correct / total if total else 0.0
    avg_lat  = sum(lat) / len(lat) if lat else 0.0

    # ── Top metric cards ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl, icon in [
        (m1, str(total),         "Questions Answered", "📝"),
        (m2, str(correct),       "User Correct",       "✅"),
        (m3, f"{accuracy:.0%}", "User Accuracy",      "🎯"),
        (m4, f"{avg_lat:.2f}s", "Avg Gen. Latency",   "⚡"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    if log:
        import numpy as np

        # ── Pre-compute all metrics from session log ───────────────────────────
        # chosen_idx  = Model A's predicted answer (A/B/C/D encoded as 0-3)
        # correct_idx = Ground-truth correct answer (A/B/C/D encoded as 0-3)
        # Only include RACE mode entries where Model A actually scored all 4 options
        race_entries = [r for r in log
                        if r.get("mode") == "race"
                        and r.get("correct_idx", -1) >= 0
                        and r.get("chosen_idx", -1) >= 0]
        y_true = [r["correct_idx"] for r in race_entries]
        y_pred = [r["chosen_idx"]  for r in race_entries]
        confs  = [r["confidence"]  for r in log]

        # Model A metrics — how well Model A predicts the correct answer
        # (independent of what the user chose)
        model_a_stats = {}
        cm_data       = None
        if len(y_true) >= 2:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, confusion_matrix,
            )
            y_t = np.array(y_true)
            y_p = np.array(y_pred)
            model_a_stats = {
                "accuracy":  accuracy_score(y_t, y_p),
                "f1":        f1_score(y_t, y_p, average="macro", zero_division=0),
                "precision": precision_score(y_t, y_p, average="macro", zero_division=0),
                "recall":    recall_score(y_t, y_p, average="macro", zero_division=0),
            }
            cm_data = confusion_matrix(y_t, y_p, labels=[0,1,2,3])

        # Model B proxy metrics
        dist_logs = [r for r in log if "n_distractors" in r]
        model_b_stats = {}
        if dist_logs:
            good    = sum(1 for r in dist_logs if r["n_distractors"] >= 3)
            partial = sum(1 for r in dist_logs if 0 < r["n_distractors"] < 3)
            failed  = sum(1 for r in dist_logs if r["n_distractors"] <= 0)
            total_b = len(dist_logs)
            model_b_stats = {
                "full_coverage":    good / total_b,
                "partial_coverage": partial / total_b,
                "failed":           failed / total_b,
                "avg_distractors":  sum(r["n_distractors"] for r in dist_logs) / total_b,
                "total":            total_b,
            }

        # ── MODEL A SECTION ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model A — Answer Verifier Performance (RACE Mode)</div>',
                    unsafe_allow_html=True)

        if model_a_stats:
            sa1, sa2, sa3, sa4 = st.columns(4)
            for col, val, lbl in [
                (sa1, f"{model_a_stats['accuracy']:.1%}",  "MCQ Accuracy"),
                (sa2, f"{model_a_stats['f1']:.3f}",        "Macro F1"),
                (sa3, f"{model_a_stats['precision']:.3f}", "Precision"),
                (sa4, f"{model_a_stats['recall']:.3f}",    "Recall"),
            ]:
                col.markdown(
                    f'<div style="background:var(--bg-elevated);border:1px solid var(--border);'
                    f'border-radius:var(--radius-sm);padding:1rem;text-align:center;">'
                    f'<div style="font-size:1.4rem;font-weight:700;color:var(--accent);'
                    f'font-family:Space Grotesk,sans-serif;">{val}</div>'
                    f'<div style="font-size:0.65rem;font-weight:600;letter-spacing:1px;'
                    f'text-transform:uppercase;color:var(--text-3);margin-top:4px;">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            col_cm, col_conf = st.columns([1, 1], gap="large")

            with col_cm:
                st.markdown('<div class="section-label">Confusion Matrix — Model A Predictions vs Ground Truth</div>',
                            unsafe_allow_html=True)
                th_s = ('font-size:0.65rem;font-weight:700;letter-spacing:1px;'
                        'text-transform:uppercase;color:var(--text-3);'
                        'padding:8px 10px;text-align:center;')
                td_s = 'font-size:0.85rem;font-weight:600;padding:9px 10px;text-align:center;'
                header_row = ('<tr><td style="' + th_s + '"></td>' +
                              ''.join(f'<th style="{th_s}">Pred {l}</th>'
                                      for l in ["A","B","C","D"]) + '</tr>')
                cm_rows_html = ""
                for i, true_l in enumerate(["A","B","C","D"]):
                    cm_rows_html += f'<tr><th style="{th_s}">True {true_l}</th>'
                    for j in range(4):
                        val = int(cm_data[i][j]) if cm_data is not None else 0
                        if i == j:
                            bg_c, fg_c = "rgba(16,185,129,0.12)", "#10b981"
                        elif val > 0:
                            bg_c, fg_c = "rgba(244,63,94,0.07)", "#f43f5e"
                        else:
                            bg_c, fg_c = "transparent", "var(--text-3)"
                        cm_rows_html += (f'<td style="{td_s}background:{bg_c};color:{fg_c};">'
                                         f'{val}</td>')
                    cm_rows_html += '</tr>'
                st.markdown(
                    f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                    f'border-radius:var(--radius-sm);overflow:hidden;">'
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead>{header_row}</thead><tbody>{cm_rows_html}</tbody>'
                    f'</table></div>'
                    f'<div style="font-size:0.7rem;color:var(--text-3);margin-top:6px;">'
                    f'Rows = ground-truth answer · Columns = Model A prediction · '
                    f'Based on {len(y_true)} RACE-mode questions this session.</div>',
                    unsafe_allow_html=True,
                )

            with col_conf:
                st.markdown('<div class="section-label">Confidence Distribution</div>',
                            unsafe_allow_html=True)
                try:
                    import plotly.graph_objects as _go
                    fig_conf = _go.Figure(_go.Histogram(
                        x=confs, nbinsx=10,
                        marker_color="#06b6d4", marker_line_width=0, opacity=0.85,
                    ))
                    fig_conf.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter", font_color="#475569",
                        xaxis_title="Confidence", yaxis_title="Count",
                        margin=dict(l=0,r=0,t=8,b=0), height=200, bargap=0.05,
                    )
                    fig_conf.update_xaxes(showgrid=False, range=[0,1])
                    fig_conf.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.06)")
                    st.plotly_chart(fig_conf, use_container_width=True)
                except Exception:
                    pass
        else:
            st.markdown(
                '<div style="background:var(--bg-elevated);border:1px dashed var(--border);'
                'border-radius:var(--radius-sm);padding:1.2rem;color:var(--text-3);'
                'font-size:0.85rem;text-align:center;">'
                'Answer at least 2 RACE-mode questions to see Model A metrics.<br>'
                '<span style="font-size:0.75rem;opacity:0.7;">Model A metrics measure how accurately the verifier predicts the correct option — independent of your choices.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # ── MODEL B SECTION ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model B — Distractor Generator Performance</div>',
                    unsafe_allow_html=True)

        if model_b_stats:
            sb1, sb2, sb3, sb4 = st.columns(4)
            for col, val, lbl, color in [
                (sb1, f"{model_b_stats['full_coverage']:.1%}",    "Full Coverage",   "#10b981"),
                (sb2, f"{model_b_stats['partial_coverage']:.1%}", "Partial Coverage","#f59e0b"),
                (sb3, f"{model_b_stats['failed']:.1%}",           "Failed",          "#f43f5e"),
                (sb4, f"{model_b_stats['avg_distractors']:.1f}",  "Avg Distractors", "#06b6d4"),
            ]:
                col.markdown(
                    f'<div style="background:var(--bg-elevated);border:1px solid var(--border);'
                    f'border-radius:var(--radius-sm);padding:1rem;text-align:center;">'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color};'
                    f'font-family:Space Grotesk,sans-serif;">{val}</div>'
                    f'<div style="font-size:0.65rem;font-weight:600;letter-spacing:1px;'
                    f'text-transform:uppercase;color:var(--text-3);margin-top:4px;">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div style="font-size:0.72rem;color:var(--text-3);margin-top:8px;">'
                f'Full Coverage = all 3 distractor slots filled with real phrases. '
                f'Based on {model_b_stats["total"]} inferences.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:var(--bg-elevated);border:1px dashed var(--border);'
                'border-radius:var(--radius-sm);padding:1.2rem;color:var(--text-3);'
                'font-size:0.85rem;text-align:center;">'
                'Generate at least one quiz to see Model B metrics.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # ── GENERATION METRICS (BLEU / ROUGE / METEOR) ───────────────────────
        # Instructor requirement: evaluate question & answer generation quality
        # Only meaningful in RACE mode where we have a real reference question
        st.markdown('<div class="section-label">Generation Metrics — BLEU / ROUGE / METEOR</div>',
                    unsafe_allow_html=True)

        # Only use entries that have a real reference question (RACE mode)
        gen_logs = [r for r in log
                    if r.get("reference_question", "").strip()
                    and r.get("generated_question", "").strip()
                    and r["reference_question"] != r["generated_question"]]

        if gen_logs:
            from src.evaluate import bleu_score, rouge_score, meteor_score
            bleu1_scores, rougeL_scores, meteor_scores = [], [], []
            for r in gen_logs:
                ref = r["reference_question"]
                hyp = r["generated_question"]
                b  = bleu_score(ref, hyp)
                ro = rouge_score(ref, hyp)
                bleu1_scores.append(b["bleu_1"])
                rougeL_scores.append(ro["rouge_l"]["f1"])
                meteor_scores.append(meteor_score(ref, hyp))

            avg_b1 = sum(bleu1_scores) / len(bleu1_scores)
            avg_rl = sum(rougeL_scores) / len(rougeL_scores)
            avg_mt = sum(meteor_scores) / len(meteor_scores)

            sg1, sg2, sg3 = st.columns(3)
            for col, val, lbl, tip in [
                (sg1, f"{avg_b1:.3f}", "BLEU-1",  "Unigram precision overlap"),
                (sg2, f"{avg_rl:.3f}", "ROUGE-L", "Longest common subsequence F1"),
                (sg3, f"{avg_mt:.3f}", "METEOR",  "Unigram F-mean with fragmentation penalty"),
            ]:
                col.markdown(
                    f'<div style="background:var(--bg-elevated);border:1px solid var(--border);'
                    f'border-radius:var(--radius-sm);padding:1rem;text-align:center;">'
                    f'<div style="font-size:1.4rem;font-weight:700;color:var(--accent-2);'
                    f'font-family:Space Grotesk,sans-serif;">{val}</div>'
                    f'<div style="font-size:0.65rem;font-weight:600;letter-spacing:1px;'
                    f'text-transform:uppercase;color:var(--text-3);margin-top:4px;">{lbl}</div>'
                    f'<div style="font-size:0.62rem;color:var(--text-3);margin-top:3px;'
                    f'font-style:italic;">{tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div style="font-size:0.7rem;color:var(--text-3);margin-top:8px;">'
                f'Compares Model A\'s predicted answer text against the RACE ground-truth answer. '
                f'Based on {len(gen_logs)} RACE-mode inferences. '
                f'Reference: Papineni et al. (2002) BLEU · Lin (2004) ROUGE · Banerjee &amp; Lavie (2005) METEOR.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:var(--bg-elevated);border:1px dashed var(--border);'
                'border-radius:var(--radius-sm);padding:1.2rem;color:var(--text-3);'
                'font-size:0.85rem;text-align:center;">'
                'Answer RACE-mode questions to see BLEU / ROUGE / METEOR scores here.<br>'
                '<span style="font-size:0.75rem;opacity:0.7;">These metrics compare the model\'s predicted answer text against the RACE ground-truth answer.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # ── SESSION LOG + CHARTS ──────────────────────────────────────────────
        col_table, col_chart = st.columns([3, 2], gap="large")

        with col_table:
            st.markdown('<div class="section-label">Session Log</div>', unsafe_allow_html=True)

            def result_badge(ok):
                if ok:
                    return ('<span style="display:inline-flex;align-items:center;gap:4px;'
                            'padding:3px 10px;border-radius:50px;font-size:0.7rem;font-weight:600;'
                            'background:rgba(16,185,129,0.12);color:#10b981;'
                            'border:1px solid rgba(16,185,129,0.25);">✓ Correct</span>')
                return ('<span style="display:inline-flex;align-items:center;gap:4px;'
                        'padding:3px 10px;border-radius:50px;font-size:0.7rem;font-weight:600;'
                        'background:rgba(244,63,94,0.1);color:#f43f5e;'
                        'border:1px solid rgba(244,63,94,0.25);">✗ Wrong</span>')

            def mode_badge_tbl(m):
                if m == "race":
                    return ('<span style="padding:2px 8px;border-radius:50px;font-size:0.66rem;'
                            'font-weight:600;background:rgba(99,102,241,0.12);color:#a5b4fc;'
                            'border:1px solid rgba(99,102,241,0.2);">RACE</span>')
                return ('<span style="padding:2px 8px;border-radius:50px;font-size:0.66rem;'
                        'font-weight:600;background:rgba(6,182,212,0.1);color:var(--accent);'
                        'border:1px solid rgba(6,182,212,0.2);">Custom</span>')

            rows = ""
            for i, r in enumerate(log):
                bg = "rgba(255,255,255,0.015)" if i % 2 == 0 else "transparent"
                q  = (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"]
                cf = r["confidence"]
                cf_color = "#10b981" if cf >= 0.6 else "#f59e0b" if cf >= 0.4 else "#f43f5e"
                rows += (
                    f'<tr style="background:{bg};border-bottom:1px solid rgba(148,163,184,0.05);">'
                    f'<td style="padding:11px 13px;font-size:0.8rem;color:var(--text-2);">{q}</td>'
                    f'<td style="padding:11px 13px;text-align:center;font-weight:700;color:var(--accent);font-size:0.88rem;">{r["chosen"]}</td>'
                    f'<td style="padding:11px 13px;text-align:center;font-weight:700;color:var(--text-3);font-size:0.88rem;">{r["correct_key"]}</td>'
                    f'<td style="padding:11px 13px;text-align:center;">{result_badge(r["is_correct"])}</td>'
                    f'<td style="padding:11px 13px;text-align:center;font-size:0.8rem;font-weight:600;color:{cf_color};">{cf:.0%}</td>'
                    f'<td style="padding:11px 13px;text-align:center;">{mode_badge_tbl(r.get("mode","custom"))}</td>'
                    f'</tr>'
                )

            th = ('padding:10px 13px;font-size:0.65rem;font-weight:700;'
                  'letter-spacing:1px;text-transform:uppercase;color:var(--text-3);')
            table_html = (
                '<div style="background:var(--bg-card);border:1px solid var(--border);'
                'border-radius:var(--radius);overflow:hidden;margin-bottom:12px;">'
                '<table style="width:100%;border-collapse:collapse;">'
                '<thead>'
                f'<tr style="background:rgba(6,182,212,0.05);border-bottom:1px solid rgba(6,182,212,0.12);">'
                f'<th style="{th}text-align:left;">Question</th>'
                f'<th style="{th}text-align:center;">Chosen</th>'
                f'<th style="{th}text-align:center;">Correct</th>'
                f'<th style="{th}text-align:center;">Result</th>'
                f'<th style="{th}text-align:center;">Confidence</th>'
                f'<th style="{th}text-align:center;">Mode</th>'
                '</tr></thead>'
                f'<tbody>{rows}</tbody>'
                '</table></div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

            csv = pd.DataFrame(log).to_csv(index=False)
            st.download_button(
                "⬇️  Export CSV", data=csv,
                file_name="session_log.csv", mime="text/csv",
                use_container_width=True,
            )

        with col_chart:
            st.markdown('<div class="section-label">Answer Distribution</div>', unsafe_allow_html=True)
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                fig = px.histogram(
                    pd.DataFrame(log), x="chosen", color="is_correct",
                    barmode="group",
                    color_discrete_map={True: "#10b981", False: "#f43f5e"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter", font_color="#475569",
                    showlegend=True, legend_title_text="",
                    legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0,r=0,t=8,b=0), height=210,
                    bargap=0.25,
                )
                fig.update_xaxes(showgrid=False, title_text="Option", title_font_size=11)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.06)", title_text="Count", title_font_size=11)
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

                if lat:
                    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Inference Latency</div>', unsafe_allow_html=True)
                    fig2 = go.Figure(go.Scatter(
                        y=lat, mode="lines+markers",
                        line=dict(color="#06b6d4", width=2),
                        marker=dict(size=5, color="#06b6d4",
                                    line=dict(color="rgba(6,182,212,0.3)", width=3)),
                        fill="tozeroy",
                        fillcolor="rgba(6,182,212,0.05)",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter", font_color="#475569",
                        xaxis_title="Request #", yaxis_title="Seconds",
                        margin=dict(l=0,r=0,t=8,b=0), height=190,
                    )
                    fig2.update_xaxes(showgrid=False)
                    fig2.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.06)")
                    st.plotly_chart(fig2, use_container_width=True)

            except ImportError:
                st.info("Install plotly for charts: pip install plotly")
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📊</div>'
            '<div class="empty-title">No Session Data Yet</div>'
            '<div class="empty-sub">Answer some questions to see your analytics here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
