from pathlib import Path
import time
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------- CLEAN, MINIMAL THEME ----------
st.markdown("""
<style>
/* Layout width */
.block-container { max-width: 880px; padding-top: .8rem; }

/* Base typography */
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}

/* Header (logo + title in one line) */
.header {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 14px;
  margin-bottom: 6px;
}
.header .logo img {
  max-height: 56px;    /* keep full logo, never oversized */
  width: auto;
}
.header .title {
  display:flex; flex-direction:column; justify-content:center;
}
.header .title h1 {
  font-size: 34px; line-height: 1.15; letter-spacing:-.01em; margin: 0;
}
.header .title p {
  margin: 4px 0 0 0; color: #6b7280;
}

/* Sections (no heavy cards) */
.section { margin-top: 16px; }
.section h3 { margin: 0 0 8px 0; font-size: 16px; }

/* Input row */
.input-row { display: grid; grid-template-columns: 1fr 132px; gap: 10px; }
.stTextInput > div > div > input { height: 44px; border-radius: 12px; }

/* Primary button — coral (your preferred color) */
.stButton > button { border-radius: 12px !important; height: 44px; font-weight: 600; }
.btn-primary { background: #ef4444 !important; border: 1px solid #ef4444 !important; }
.btn-primary:hover { filter: brightness(0.97); }

/* Example chips — responsive grid; no overflow */
.examples {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 10px;
}
.chip {
  border: 1px solid #e5e7eb; background: #fafafa; color: #374151;
  border-radius: 999px; padding: 8px 12px; font-size: 13px;
  text-align: center; white-space: normal; line-height: 1.2; cursor: pointer;
}
.chip:hover { background: #f3f4f6; }

/* Q/A rows — simple, no frames */
.qa { margin-top: 10px; }
.qa-label { color:#6b7280; font-weight:600; font-size:12px; margin-bottom:4px; }
.qa-bubble {
  background: #f8fafc; border: 1px solid #eef2f7; border-radius: 12px; padding: 12px 14px;
}

/* Sources */
.source {
  border: 1px solid #eef2f7; border-radius: 10px; padding: 10px 12px; background: #fcfcfd;
}

/* Remove any heavy outer shadows globally */
div[data-testid="stDecoration"] { box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "question" not in st.session_state: st.session_state.question = ""
if "history" not in st.session_state: st.session_state.history = []  # [{q,a,sources}]

def set_example(q: str):
    st.session_state.question = q

# ---------- HEADER ----------
st.markdown('<div class="header">', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.markdown('<div class="logo">', unsafe_allow_html=True)
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="title"><h1>Uni-Assistant</h1>'
                    '<p>Ask anything about Technion courses, prerequisites, and study programs. '
                    '<b>(Demo mode)</b></p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- ASK ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h3>Enter your question</h3>', unsafe_allow_html=True)
st.markdown('<div class="input-row">', unsafe_allow_html=True)
left, right = st.columns([1, 0.2])
with left:
    st.session_state.question = st.text_input(
        label="",
        value=st.session_state.question,
        placeholder="For example: What are the prerequisites for Differential Calculus?",
        label_visibility="collapsed",
    )
with right:
    ask_clicked = st.button("Ask (Demo)", use_container_width=True, key="ask_btn")
st.markdown('</div>', unsafe_allow_html=True)

# Example chips (no overflow; crisp grid)
st.markdown('<div class="examples">', unsafe_allow_html=True)
examples = [
    "What courses can I take in Semester A with no prerequisites?",
    "How do I transfer credits from another university?",
    "Where can I find the 2024–25 CS program handbook?",
]
for i, ex in enumerate(examples):
    st.button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # /section

# ---------- DEMO BACKEND ----------
if ask_clicked and st.session_state.question.strip():
    time.sleep(0.25)
    q = st.session_state.question.strip()
    a = ("In most programs, Differential Calculus requires meeting the math placement requirement "
         "or completing the introductory math course. Check your program handbook for the precise rule.")
    sources = [
        {"title": "Program Handbook 2024–25", "score": 0.93,
         "snippet": "Students must meet the placement requirement or complete the introductory math course…"},
        {"title": "Mathematics Dept FAQ", "score": 0.86,
         "snippet": "Intro course may be required if placement is not met."},
    ]
    st.session_state.history.append({"q": q, "a": a, "sources": sources})

# ---------- RESULTS ----------
if st.session_state.history:
    turn = st.session_state.history[-1]

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('### Results')

    st.markdown('<div class="qa">', unsafe_allow_html=True)
    st.markdown('<div class="qa-label">Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="qa-bubble">', unsafe_allow_html=True)
    st.write(turn["q"])
    st.markdown('</div>', unsafe_allow_html=True)  # bubble

    st.markdown('<div class="qa-label" style="margin-top:10px;">Answer</div>', unsafe_allow_html=True)
    st.markdown('<div class="qa-bubble">', unsafe_allow_html=True)
    st.write(turn["a"])
    st.markdown('</div>', unsafe_allow_html=True)  # bubble
    st.markdown('</div>', unsafe_allow_html=True)  # qa

    if turn["sources"]:
        with st.expander("Sources"):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**{i}. {s['title']}**  •  score {s['score']:.2f}")
                st.markdown(f'<div class="source">{s["snippet"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # /section

# Ensure the Ask button is coral without JS hacks
st.markdown("""
<script>
for (const b of parent.document.querySelectorAll('button')) {
  if (b.innerText.trim().startsWith('Ask')) b.classList.add('btn-primary');
}
</script>
""", unsafe_allow_html=True)
