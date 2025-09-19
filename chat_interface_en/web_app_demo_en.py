from pathlib import Path
import time
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ---------- CLEAN THEME ----------
st.markdown("""
<style>
.block-container { max-width: 880px; padding-top: .8rem; }
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial; }

/* Header */
.header{display:grid;grid-template-columns:auto 1fr;align-items:center;gap:14px;margin-bottom:6px}
.header .logo img{max-height:56px;width:auto}
.header .title h1{font-size:34px;line-height:1.15;letter-spacing:-.01em;margin:0}
.header .title p{margin:4px 0 0 0;color:#6b7280}

/* Sections */
.section{margin-top:16px}
.section h3{margin:0 0 8px 0;font-size:16px}

/* Input row */
.input-row{display:grid;grid-template-columns:1fr 132px;gap:10px}
.stTextInput > div > div > input{height:44px;border-radius:12px}

/* Buttons */
.stButton > button{border-radius:12px !important;height:44px;font-weight:600}
.btn-primary{background:#ef4444 !important;border:1px solid #ef4444 !important}
.btn-primary:hover{filter:brightness(0.97)}

/* Example chips – responsive, no overflow */
.examples{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.chip{border:1px solid #e5e7eb;background:#fafafa;color:#374151;border-radius:999px;padding:8px 12px;font-size:13px;text-align:center;white-space:normal;line-height:1.2;cursor:pointer}
.chip:hover{background:#f3f4f6}

/* Q/A rows */
.qa{margin-top:10px}
.qa-label{color:#6b7280;font-weight:600;font-size:12px;margin-bottom:4px}
.qa-bubble{background:#f8fafc;border:1px solid #eef2f7;border-radius:12px;padding:12px 14px}

/* Sources */
.source-wrap{margin-top:6px}
.source-title{font-weight:700}
.source{
  border:1px solid #eef2f7;border-radius:10px;padding:10px 12px;background:#fcfcfd;
  white-space:pre-wrap;   /* keep line breaks */
  word-break:break-word;
  max-height:6.5em;       /* ~4-5 lines preview */
  overflow:hidden;
  transition:max-height .15s ease;
}
.source:hover{
  max-height:999em;       /* expand to full chunk on hover */
}

/* Sidebar history items */
.hist-item{padding:6px 8px;border-radius:8px;border:1px solid #eef2f7;margin-bottom:6px;background:#fff;cursor:pointer}
.hist-item:hover{background:#f8fafc}
.clear-btn button{background:#f3f4f6 !important;color:#374151 !important;border:1px solid #e5e7eb !important}
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "question" not in st.session_state: st.session_state.question = ""
if "history" not in st.session_state: st.session_state.history = []  # [{q,a,sources}]
def set_example(q:str): st.session_state.question = q
def load_from_history(idx:int): st.session_state.question = st.session_state.history[idx]["q"]

# ---------- SIDEBAR HISTORY ----------
st.sidebar.header("History")
if st.session_state.history:
    for i, turn in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        st.sidebar.button(
            turn["q"][:60] + ("…" if len(turn["q"])>60 else ""),
            key=f"h_{idx}", on_click=load_from_history, args=(idx,),
            use_container_width=True
        )
else:
    st.sidebar.caption("No questions yet.")
st.sidebar.markdown('<div class="clear-btn">', unsafe_allow_html=True)
if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history = []
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="header">', unsafe_allow_html=True)
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown('<div class="logo">', unsafe_allow_html=True)
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH))
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="title"><h1>Uni-Assistant</h1>'
                '<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p></div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- ASK ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h3>Enter your question</h3>', unsafe_allow_html=True)
left, right = st.columns([1, 0.2])
with left:
    st.session_state.question = st.text_input(
        label="", value=st.session_state.question,
        placeholder="For example: What are the prerequisites for Differential Calculus?",
        label_visibility="collapsed",
    )
with right:
    ask_clicked = st.button("Ask (Demo)", use_container_width=True, key="ask_btn")

st.markdown('<div class="examples">', unsafe_allow_html=True)
for i, ex in enumerate([
    "What courses can I take in Semester A with no prerequisites?",
    "How do I transfer credits from another university?",
    "Where can I find the 2024–25 CS program handbook?",
]):
    st.button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- DEMO BACKEND ----------
if ask_clicked and st.session_state.question.strip():
    time.sleep(0.25)
    q = st.session_state.question.strip()
    a = ("In most programs, Differential Calculus requires meeting the math placement requirement "
         "or completing the introductory math course. Check your program handbook for the precise rule.")
    # NOTE: no "…" ellipses in snippets anymore
    sources = [
        {"title":"Program Handbook 2024–25","score":0.93,
         "chunk":"Students must meet the mathematics placement requirement or complete the introductory course prior to enrolling in Differential Calculus. This applies to most tracks; exceptions are listed in Appendix B."},
        {"title":"Mathematics Dept FAQ","score":0.86,
         "chunk":"If the placement requirement is not met, students are required to take the introductory math course first. See the FAQ for equivalency tables and exemptions."},
    ]
    st.session_state.history.append({"q": q, "a": a, "sources": sources})

# ---------- RESULTS ----------
if st.session_state.history:
    turn = st.session_state.history[-1]
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('### Results')

    st.markdown('<div class="qa">', unsafe_allow_html=True)
    st.markdown('<div class="qa-label">Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="qa-bubble">', unsafe_allow_html=True); st.write(turn["q"]); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="qa-label" style="margin-top:10px;">Answer</div>', unsafe_allow_html=True)
    st.markdown('<div class="qa-bubble">', unsafe_allow_html=True); st.write(turn["a"]); st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if turn["sources"]:
        with st.expander("Sources", expanded=False):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**{i}. {s['title']}** • score {s['score']:.2f}")
                # Preview that expands on hover
                st.markdown(f'<div class="source-wrap"><div class="source">{s["chunk"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# keep coral style on Ask buttons
st.markdown("""
<script>
for (const b of parent.document.querySelectorAll('button')) {
  if (b.innerText.trim().startsWith('Ask')) b.classList.add('btn-primary');
}
</script>
""", unsafe_allow_html=True)
