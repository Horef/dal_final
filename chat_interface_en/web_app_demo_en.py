from pathlib import Path
import time
import textwrap
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------------- Minimal, clean theme ----------------
st.markdown("""
<style>
/* Page width & base font */
.block-container { max-width: 880px; padding-top: 1.0rem; }
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}

/* Hero: center logo + title */
.hero { text-align:center; margin-bottom: 10px; }
.hero img { max-height: 70px; margin-bottom: 6px; }
.hero h1 { font-size: 40px; line-height: 1.1; letter-spacing:-0.01em; margin: 0 0 4px 0; }
.hero p { color:#6b7280; margin: 0; }

/* Cards (very light) */
.card { background:#fff; border:1px solid #eef2f7; border-radius:16px; padding:16px; box-shadow: 0 6px 22px rgba(0,0,0,.04); }

/* Input row */
.inline { display:grid; grid-template-columns: 1fr 135px; gap: 10px; }
.stTextInput > div > div > input { height: 44px; border-radius: 12px; }

/* Buttons */
.stButton > button { border-radius:12px !important; height:44px; font-weight:600; }
.btn-primary { background:#ef4444 !important; border: 1px solid #ef4444 !important; }

/* Chips */
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { border:1px solid #e5e7eb; background:#fafafa; color:#374151; font-size:13px;
        padding:6px 10px; border-radius:999px; cursor:pointer; }
.chip:hover { background:#f3f4f6; }

/* Q/A bubbles */
.section-title { font-weight:700; margin: 6px 0 10px 0; font-size:16px; }
.row { display:grid; grid-template-columns: 74px auto; gap:10px; margin-top:12px; }
.tag { width:74px; text-align:center; font-weight:600; font-size:12px; color:#6b7280; }
.bubble { border-radius: 14px; padding: 12px 14px; border:1px solid #eef2f7; background:#f8fafc; }

/* Toolbar */
.toolbar { display:flex; gap:8px; align-items:center; margin-top:8px; }
.muted { color:#9ca3af; font-size:12.5px; }

/* Sources */
.source { border:1px solid #eef2f7; border-radius:12px; padding:10px 12px; background:#fcfcfd; }
</style>
""", unsafe_allow_html=True)

# ---------------- State ----------------
if "question" not in st.session_state: st.session_state.question = ""
if "history" not in st.session_state: st.session_state.history = []  # [{q,a,sources}]

def set_example(q: str):
    st.session_state.question = q

# ---------------- Hero (centered) ----------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), output_format="PNG")  # center, full logo
st.markdown("<h1>Uni-Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Ask card ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.form("ask_form", clear_on_submit=False):
    st.markdown("**Enter your question**")
    left, right = st.columns([1, 0.2])
    with left:
        st.session_state.question = st.text_input(
            label="",
            value=st.session_state.question,
            placeholder="For example: What are the prerequisites for Differential Calculus?",
            label_visibility="collapsed",
        )
    with right:
        ask_clicked = st.form_submit_button("Ask (Demo)")
# small examples row
e1, e2, e3 = st.columns(3)
for i, ex in enumerate([
    "What courses can I take in Semester A with no prerequisites?",
    "How do I transfer credits from another university?",
    "Where can I find the 2024–25 CS program handbook?",
]):
    with (e1, e2, e3)[i]:
        st.button(ex, key=f"ex_{i}", use_container_width=True, on_click=set_example, args=(ex,))
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Demo backend ----------------
if ask_clicked and st.session_state.question.strip():
    time.sleep(0.25)
    q = st.session_state.question.strip()
    a = ("In most programs, Differential Calculus requires meeting the math placement requirement "
         "or completing the introductory math course. Check your program handbook for the precise rule.")
    sources = [
        {"title": "Program Handbook 2024–25", "score": 0.93, "snippet": "Students must meet the placement requirement or complete the introductory math course…"},
        {"title": "Mathematics Dept FAQ", "score": 0.86, "snippet": "Intro course may be required if placement is not met."},
    ]
    st.session_state.history.append({"q": q, "a": a, "sources": sources})

# ---------------- Results ----------------
if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")

    turn = st.session_state.history[-1]
    # Question
    st.markdown('<div class="row"><div class="tag">Question</div><div class="bubble">', unsafe_allow_html=True)
    st.write(turn["q"])
    st.markdown('</div></div>', unsafe_allow_html=True)
    # Answer
    st.markdown('<div class="row"><div class="tag">Answer</div><div class="bubble">', unsafe_allow_html=True)
    st.write(turn["a"])
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Simple toolbar
    with st.container():
        st.markdown('<div class="toolbar">', unsafe_allow_html=True)
        st.download_button("Download (.md)", data=f"# Answer\\n\\n{turn['a']}\n",
                           file_name="answer.md", mime="text/markdown")
        st.markdown('<span class="muted">— save answer</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if turn["sources"]:
        with st.expander("Sources"):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**{i}. {s['title']}** • score {s['score']:.2f}")
                st.markdown(f'<div class="source">{s["snippet"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# keep the coral button style (Streamlit wraps it), so patch class on the last rendered button
st.markdown("""
<script>
const btns = parent.document.querySelectorAll('button');
btns.forEach(b => {
  if (b.innerText.trim().startsWith('Ask')) { b.classList.add('btn-primary'); }
});
</script>
""", unsafe_allow_html=True)
