# web_app_demo_en.py
from pathlib import Path
import time
import streamlit as st

# ---------- Page & theme ----------
st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------- Global CSS ----------
st.markdown("""
<style>
:root {
  --card-bg: #ffffff;
  --card-radius: 18px;
  --shadow: 0 8px 30px rgba(0,0,0,.06);
}
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
}
.main > div {
  padding-top: 1rem !important;
}
.hero {
  margin: 16px 0 26px 0;
  display: grid;
  grid-template-columns: 140px auto;
  gap: 24px;
  align-items: center;
}
.hero h1 {
  font-size: 48px;
  line-height: 1.05;
  margin: 0;
  letter-spacing: -0.02em;
}
.hero p {
  margin: 6px 0 0 0;
  color: #6b7280;
  font-size: 15.5px;
}
.card {
  background: var(--card-bg);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow);
  padding: 22px 22px;
  border: 1px solid #eef2f7;
}
.inline {
  display: grid;
  grid-template-columns: 1fr 140px;
  gap: 12px;
}
.examples {
  display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;
}
.example-chip {
  border: 1px solid #e5e7eb; padding: 6px 10px; border-radius: 999px;
  font-size: 13px; color: #374151; background: #fafafa; cursor: pointer;
}
.example-chip:hover { background: #f3f4f6; }
.stButton > button {
  border-radius: 12px !important;
  height: 44px;
  font-weight: 600;
}
.ask { background: #ef4444 !important; border: 1px solid #ef4444 !important; }
.ask:hover { filter: brightness(0.95); }
.section-title {
  font-weight: 700; margin: 8px 0 8px 0; font-size: 16px;
}
.bubble {
  border-radius: 16px; padding: 14px 16px; border: 1px solid #eef2f7;
  background: #f8fafc;
}
.qa-row { display: grid; grid-template-columns: 60px auto; gap: 10px; margin-top: 14px; }
.qtag, .atag {
  width: 60px; text-align:center; font-weight:600; font-size:12px;
  color:#6b7280;
}
.footer {
  margin-top: 28px; color: #9ca3af; font-size: 12.5px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col = st.container()
with col:
  st.markdown('<div class="hero">', unsafe_allow_html=True)
  c1, c2 = st.columns([1, 7])
  with c1:
    if LOGO_PATH.exists():
      st.image(LOGO_PATH, width=120)
  with c2:
    st.markdown("<h1>Uni-Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p>", unsafe_allow_html=True)
  st.markdown('</div>', unsafe_allow_html=True)

# ---------- Sidebar (demo knobs) ----------
st.sidebar.header("Demo Settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 20, 5)
latency_ms = st.sidebar.slider("Simulated latency (ms)", 0, 3000, 500, 50)

# ---------- State ----------
if "history" not in st.session_state:
  st.session_state.history = []  # list of dicts: {"q": str, "a": str, "sources": [...]}

# ---------- Ask Card ----------
with st.container():
  st.markdown('<div class="card">', unsafe_allow_html=True)

  st.markdown('<div class="section-title">Enter your question</div>', unsafe_allow_html=True)
  c_in = st.container()
  with c_in:
    # inline input + button
    st.markdown('<div class="inline">', unsafe_allow_html=True)
    qcol, bcol = st.columns([1, 0.18])
    with qcol:
      question = st.text_input(
        label="",
        placeholder="For example: What are the prerequisites for Differential Calculus?",
        label_visibility="collapsed",
      )
    with bcol:
      ask_clicked = st.button("Ask (Demo)", use_container_width=True, type="primary", key="ask_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    # example chips
    st.markdown('<div class="examples">', unsafe_allow_html=True)
    examples = [
      "What courses can I take in Semester A with no prerequisites?",
      "How do I transfer credits from another university?",
      "Where can I find the 2024–25 CS program handbook?",
    ]
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
      with ex_cols[i]:
        if st.button(ex, key=f"ex_{i}", help="Insert example", use_container_width=True):
          st.session_state["__set_q"] = ex
    st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('</div>', unsafe_allow_html=True)  # /card

# populate example click
if "__set_q" in st.session_state:
  st.experimental_rerun()

# ---------- Demo answer ----------
if ask_clicked:
  if not question.strip():
    st.warning("Please enter a question.")
  else:
    with st.spinner("Thinking…"):
      time.sleep(latency_ms / 1000.0)

    # Stubbed answer + pretend sources
    answer = (
      "In most tracks, Differential Calculus requires meeting the math placement requirement or successfully "
      "completing the introductory math course. Always verify in your program handbook for the exact rule."
    )
    sources = [
      {"score": 0.93, "text": "Program Handbook 2024–25, p. 4: Students must meet the placement requirement..."},
      {"score": 0.86, "text": "Mathematics Department FAQ: Intro course required if placement is not met."},
    ][:top_k]
    st.session_state.history.append({"q": question, "a": answer, "sources": sources})

# ---------- History / Results ----------
if st.session_state.history:
  with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    for turn in reversed(st.session_state.history):
      st.markdown('<div class="qa-row"><div class="qtag">Question</div><div class="bubble">', unsafe_allow_html=True)
      st.write(turn["q"])
      st.markdown('</div></div>', unsafe_allow_html=True)

      st.markdown('<div class="qa-row"><div class="atag">Answer</div><div class="bubble">', unsafe_allow_html=True)
      st.write(turn["a"])
      st.markdown('</div></div>', unsafe_allow_html=True)

      if turn["sources"]:
        with st.expander("Sources"):
          for i, s in enumerate(turn["sources"], 1):
            st.markdown(f"**Source {i} • score {s['score']:.2f}**")
            st.write(s["text"])

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Demo build • No real backend calls</div>', unsafe_allow_html=True)
