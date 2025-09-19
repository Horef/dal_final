from pathlib import Path
import time
import textwrap
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container { max-width: 980px; padding-top: 1.1rem; }

/* Typography */
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}

/* Hero */
.hero { display:grid; grid-template-columns:100px auto; gap:16px; align-items:center; margin-bottom:8px; }
.hero h1 { font-size:40px; line-height:1.08; margin:0; letter-spacing:-.01em; }
.hero p { margin:4px 0 0 0; color:#6b7280; }

/* Cards */
.card { background:#fff; border:1px solid #eef2f7; border-radius:16px; padding:16px 16px;
        box-shadow:0 8px 30px rgba(0,0,0,.05); }

/* Input row */
.inline { display:grid; grid-template-columns: 1fr 130px; gap:10px; }
.stTextInput > div > div > input { height:44px; border-radius:12px; }
.stButton > button { border-radius:12px !important; height:44px; font-weight:600; }
.ask { background:#ef4444 !important; border:1px solid #ef4444 !important; }

/* Chips */
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { border:1px solid #e5e7eb; background:#fafafa; color:#374151; font-size:13px;
        padding:6px 10px; border-radius:999px; cursor:pointer; }
.chip:hover { background:#f3f4f6; }

/* Q/A bubbles */
.section-title { font-weight:700; margin:6px 0 10px 0; font-size:16px; }
.qa-row { display:grid; grid-template-columns:72px auto; gap:10px; margin-top:12px; }
.qtag, .atag { width:72px; text-align:center; font-weight:600; font-size:12px; color:#6b7280; }
.bubble { border-radius:14px; padding:12px 14px; border:1px solid #eef2f7; background:#f8fafc; }

/* Toolbar */
.toolbar { display:flex; gap:8px; align-items:center; margin-top:8px; }
.toolbar .muted { color:#9ca3af; font-size:12.5px; }

/* Source card */
.source { border:1px solid #eef2f7; border-radius:12px; padding:10px 12px; background:#fcfcfd; }

/* Sidebar history */
.hist-item { padding:6px 8px; border-radius:8px; border:1px solid #eef2f7; margin-bottom:6px; cursor:pointer; background:#fff; }
.hist-item:hover { background:#f8fafc; }
.clear-btn button { background:#f3f4f6 !important; color:#374151 !important; border:1px solid #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- State ----------------
if "question" not in st.session_state: st.session_state.question = ""
if "history" not in st.session_state:  # list of {q,a,sources,ts}
    st.session_state.history = []
if "last_answer" not in st.session_state: st.session_state.last_answer = ""

def use_example(q: str):
    st.session_state.question = q

# ---------------- Hero ----------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
c1, c2 = st.columns([1, 12])
with c1:
    if LOGO_PATH.exists(): st.image(LOGO_PATH, width=90)
with c2:
    st.markdown("<h1>Uni-Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Sidebar: history ----------------
st.sidebar.header("History")
if st.session_state.history:
    for i, turn in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        if st.sidebar.button(turn["q"][:60] + ("…" if len(turn["q"]) > 60 else ""), key=f"h_{idx}"):
            st.session_state.question = turn["q"]
else:
    st.sidebar.caption("No questions yet.")
st.sidebar.markdown('<div class="clear-btn">', unsafe_allow_html=True)
if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history = []
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ---------------- Ask card ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.form("ask_form", clear_on_submit=False):
    col_input, col_btn = st.columns([1, 0.2])
    with col_input:
        st.session_state.question = st.text_input(
            label="Enter your question",
            value=st.session_state.question,
            placeholder="For example: What are the prerequisites for Differential Calculus?",
        )
    with col_btn:
        ask_clicked = st.form_submit_button("Ask (Demo)")
if not st.session_state.question:
    st.caption("Tip: press **Enter** to submit. ")
# quick examples
ex1, ex2, ex3 = st.columns(3)
examples = [
    "What courses can I take in Semester A with no prerequisites?",
    "How do I transfer credits from another university?",
    "Where can I find the 2024–25 CS program handbook?",
]
for i, ex in enumerate(examples):
    with (ex1, ex2, ex3)[i]:
        st.button(ex, key=f"ex_{i}", use_container_width=True, on_click=use_example, args=(ex,))
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Demo “backend” ----------------
if ask_clicked and st.session_state.question.strip():
    time.sleep(0.3)  # tiny latency to feel responsive
    q = st.session_state.question.strip()
    a = (
        "In most programs, Differential Calculus requires meeting the math placement requirement "
        "or completing the introductory math course. Check your program handbook for the precise rule."
    )
    sources = [
        {"title": "Program Handbook 2024–25", "score": 0.93, "url": None,
         "snippet": "Students must meet the placement requirement or complete the introductory math course…"},
        {"title": "Mathematics Dept FAQ", "score": 0.86, "url": None,
         "snippet": "Intro course may be required if placement is not met."},
    ]
    st.session_state.history.append({"q": q, "a": a, "sources": sources})
    st.session_state.last_answer = a

# ---------------- Results ----------------
if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")

    latest = st.session_state.history[-1]
    # Question
    st.markdown('<div class="qa-row"><div class="qtag">Question</div><div class="bubble">', unsafe_allow_html=True)
    st.write(latest["q"])
    st.markdown('</div></div>', unsafe_allow_html=True)
    # Answer
    st.markdown('<div class="qa-row"><div class="atag">Answer</div><div class="bubble">', unsafe_allow_html=True)
    st.write(latest["a"])
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Answer toolbar (copy / download / feedback)
    with st.container():
        st.markdown('<div class="toolbar">', unsafe_allow_html=True)
        copy_code = textwrap.dedent(f"""
            <button id="copyBtn" style="padding:6px 10px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;cursor:pointer;">
              Copy
            </button>
            <script>
            const btn = document.getElementById('copyBtn');
            if (btn) {{
              btn.onclick = () => navigator.clipboard.writeText({latest["a"]!r});
            }}
            </script>
        """)
        st.markdown(copy_code, unsafe_allow_html=True)

        st.download_button("Download (.md)", data=f"# Answer\\n\\n{latest['a']}\n",
                           file_name="answer.md", mime="text/markdown")
        col_up, col_down = st.columns([0.07, 0.07])
        with col_up:
            st.button("👍", key="fb_up")
        with col_down:
            st.button("👎", key="fb_down")
        st.markdown('<span class="muted">— actions</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sources
    if latest["sources"]:
        with st.expander("Sources"):
            for i, s in enumerate(latest["sources"], 1):
                with st.container():
                    st.markdown(f"**{i}. {s['title']}**  •  score {s['score']:.2f}")
                    if s.get("url"):
                        st.write(s["url"])
                    st.markdown(f'<div class="source">{s["snippet"]}</div>', unsafe_allow_html=True)

    # Follow-ups
    st.markdown("##### Follow-up questions")
    f1, f2, f3 = st.columns(3)
    followups = [
        "Can I take Linear Algebra before Calculus 1?",
        "What is the typical course order for first-year CS?",
        "Where do I apply for math placement?",
    ]
    for i, fq in enumerate(followups):
        with (f1, f2, f3)[i]:
            st.button(fq, key=f"fu_{i}", use_container_width=True, on_click=use_example, args=(fq,))
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Demo build • No real backend calls")
