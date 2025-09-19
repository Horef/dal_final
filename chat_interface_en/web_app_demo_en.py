from pathlib import Path
import time
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------- CSS ----------
st.markdown("""
<style>
/* Constrain the main column width */
.block-container { max-width: 980px; padding-top: 1.2rem; }

/* Typography & base */
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}

/* Hero */
.hero {
  display: grid; grid-template-columns: 110px auto; gap: 18px; align-items: center;
  margin-bottom: 10px;
}
.hero h1 { font-size: 40px; line-height: 1.08; margin: 0; letter-spacing: -.01em; }
.hero p { margin: 4px 0 0 0; color: #6b7280; }

/* Card */
.card {
  background: #fff; border: 1px solid #eef2f7; border-radius: 16px; padding: 18px 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,.05);
}

/* Input row */
.inline { display: grid; grid-template-columns: 1fr 132px; gap: 10px; }
.stTextInput > div > div > input {
  height: 44px; border-radius: 12px;
}

/* Primary button */
.stButton > button { border-radius: 12px !important; height: 44px; font-weight: 600; }
.ask { background: #ef4444 !important; border: 1px solid #ef4444 !important; }
.ask:hover { filter: brightness(0.97); }

/* Example chips */
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
  border: 1px solid #e5e7eb; background: #fafafa; color: #374151; font-size: 13px;
  padding: 6px 10px; border-radius: 999px; cursor: pointer;
}
.chip:hover { background: #f3f4f6; }

/* Q/A bubbles */
.section-title { font-weight: 700; margin: 6px 0 10px 0; font-size: 16px; }
.qa-row { display: grid; grid-template-columns: 68px auto; gap: 10px; margin-top: 12px; }
.qtag, .atag { width: 68px; text-align: center; font-weight: 600; font-size: 12px; color: #6b7280; }
.bubble { border-radius: 14px; padding: 12px 14px; border: 1px solid #eef2f7; background: #f8fafc; }

/* Footer */
.footer { margin-top: 18px; color: #9ca3af; font-size: 12.5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
c1, c2 = st.columns([1, 10])
with c1:
    if LOGO_PATH.exists():
        st.image(LOGO_PATH, width=100)
with c2:
    st.markdown("<h1>Uni-Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Sidebar (demo knobs) ----------
st.sidebar.header("Demo settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 20, 5)
latency_ms = st.sidebar.slider("Simulated latency (ms)", 0, 3000, 400, 50)

# ---------- State ----------
if "question" not in st.session_state:
    st.session_state.question = ""
if "history" not in st.session_state:
    st.session_state.history = []  # [{q,a,sources}]

def set_example(q: str):
    st.session_state.question = q

# ---------- Ask card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('**Enter your question**', unsafe_allow_html=True)

col_input, col_btn = st.columns([1, 0.18])
with col_input:
    question = st.text_input(
        label="",
        value=st.session_state.question,
        placeholder="For example: What are the prerequisites for Differential Calculus?",
        label_visibility="collapsed",
        key="question",
    )
with col_btn:
    ask_clicked = st.button("Ask (Demo)", use_container_width=True, type="primary")

# Example chips
st.markdown('<div class="examples">', unsafe_allow_html=True)
examples = [
    "What courses can I take in Semester A with no prerequisites?",
    "How do I transfer credits from another university?",
    "Where can I find the 2024–25 CS program handbook?",
]
e1, e2, e3 = st.columns(3)
for i, ex in enumerate(examples):
    with (e1, e2, e3)[i]:
        st.button(ex, key=f"ex_{i}", use_container_width=True, on_click=set_example, args=(ex,))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /card

# ---------- Demo answer ----------
if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            time.sleep(latency_ms / 1000.0)

        answer = (
            "In most programs, Differential Calculus requires meeting the math placement requirement "
            "or completing the introductory math course. Check your program handbook for the precise rule."
        )
        sources = [
            {"score": 0.93, "text": "Program Handbook 2024–25, p. 4: Students must meet the placement requirement..."},
            {"score": 0.86, "text": "Mathematics Dept FAQ: Intro course required if placement is not met."},
        ][:top_k]
        st.session_state.history.append({"q": question, "a": answer, "sources": sources})

# ---------- Results ----------
if st.session_state.history:
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
