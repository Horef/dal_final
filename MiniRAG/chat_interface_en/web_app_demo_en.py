from pathlib import Path
import time
import streamlit as st

st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ===================== THEME =====================
st.markdown("""
<style>
/* Page width */
.block-container { max-width: 880px; padding-top: .8rem; }

/* Base font */
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}

/* Header (logo + title) */
.header { display:grid; grid-template-columns:auto 1fr; align-items:center; gap:14px; margin-bottom:6px; }
.header .logo img { max-height:56px; width:auto; }
.header .title h1 { font-size:34px; line-height:1.15; letter-spacing:-.01em; margin:0; }
.header .title p { margin:4px 0 0 0; color:#6b7280; }

/* Sections */
.section { margin-top:16px; }
.section h3 { margin:0 0 8px 0; font-size:20px; font-weight:700; }

/* Inputs */
.stTextInput > div > div > input { height:44px; border-radius:12px; }

/* Default buttons = WHITE (chips, sidebar, etc.) */
.stButton > button{
  border-radius:12px; height:44px; font-weight:600;
  background:#ffffff; border:1px solid #e5e7eb; color:#111827;
}

/* Example chips — white, wrap nicely */
.examples { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:10px; }
.examples .stButton > button{
  height:auto; line-height:1.2; white-space:normal; text-align:center;
  border-radius:999px; padding:8px 12px;
}
.examples .stButton > button:hover { background:#f8fafc; }

/* Q/A (plain text) */
.qa { margin-top:10px; }
.qa-label { color:#6b7280; font-weight:600; font-size:12px; margin:4px 0; }

/* Sidebar history buttons */
[data-testid="stSidebar"] .stButton > button{
  border:1px solid #e5e7eb; background:#fff; color:#111827;
  border-radius:10px; padding:8px 10px; height:auto; line-height:1.25;
  white-space:normal; text-align:left; overflow-wrap:anywhere;
}
[data-testid="stSidebar"] .clear-btn > button{
  background:#f3f4f6; color:#374151; border:1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# --- JS: keep ONLY the Ask button coral (even after rerenders) ---
st.markdown("""
<script>
(function(){
  const APPLY = () => {
    const root = parent.document;
    const btns = root.querySelectorAll('div[data-ask] button');
    btns.forEach(b=>{
      const t = (b.innerText || '').trim();
      if (t === 'Ask' || t === 'Ask (Demo)') {
        b.style.background = '#ef4444';
        b.style.border = '1px solid #ef4444';
        b.style.color = '#ffffff';
        b.style.boxShadow = 'none';
        b.style.borderRadius = '12px';
        b.style.height = '44px';
        b.style.fontWeight = '600';
      }
    });
  };
  APPLY();
  new MutationObserver(APPLY).observe(parent.document.body, {subtree:true, childList:true});
})();
</script>
""", unsafe_allow_html=True)

# ===================== STATE =====================
if "question" not in st.session_state:
    st.session_state.question = ""
if "history" not in st.session_state:
    st.session_state.history = []    # [{q, a}]

def set_example(q: str):
    st.session_state.question = q

def load_from_history(idx: int):
    st.session_state.question = st.session_state.history[idx]["q"]

# Hard-coded demo Q&A only
DEMO_QA = {
    "what courses are required in all curricula as foundational?": "mathematics",
    "what are the main units within the undergraduate studies center?": "the Undergraduate Program and the Undergraduate Research Program",
    "which parts does the physics placement exam include?": "Physics, Chemistry, and Maths",
}

def normalize(txt: str) -> str:
    return " ".join((txt or "").strip().lower().split())

# ===================== SIDEBAR: HISTORY =====================
st.sidebar.header("History")
if st.session_state.history:
    for i, turn in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        st.sidebar.button(turn["q"], key=f"h_{idx}", on_click=load_from_history, args=(idx,), use_container_width=True)
else:
    st.sidebar.caption("No questions yet.")
st.sidebar.markdown('<div class="clear-btn">', unsafe_allow_html=True)
if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history = []
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown('<div class="header">', unsafe_allow_html=True)
col1, col2 = st.columns([0.15, 0.85], vertical_alignment="center")
with col1:
    st.markdown('<div class="logo">', unsafe_allow_html=True)
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH))
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        '<div class="title"><h1>Uni-Assistant</h1>'
        '<p>Ask anything about Technion courses, prerequisites, and study programs. <b>(Demo mode)</b></p></div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ===================== ASK =====================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h3>Enter your question</h3>', unsafe_allow_html=True)

left, right = st.columns([1, 0.2], vertical_alignment="center")
with left:
    st.session_state.question = st.text_input(
        label="",
        value=st.session_state.question,
        placeholder="For example: What courses are required in all curricula as foundational?",
        label_visibility="collapsed",
    )
with right:
    st.markdown('<div data-ask class="ask-wrap">', unsafe_allow_html=True)
    ask_clicked = st.button("Ask (Demo)", type="primary", use_container_width=True, key="ask_btn_demo")
    st.markdown('</div>', unsafe_allow_html=True)

# Example chips use your three questions
examples = [
    "What courses are required in all curricula as foundational?",
    "What are the main units within the Undergraduate Studies Center?",
    "Which parts does the Physics placement exam include?",
]
st.markdown('<div class="examples">', unsafe_allow_html=True)
for i, ex in enumerate(examples):
    st.button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # /section

# ===================== DEMO BACKEND (Q -> A only) =====================
if ask_clicked and st.session_state.question.strip():
    time.sleep(0.25)
    q = st.session_state.question.strip()
    a = DEMO_QA.get(normalize(q), "This demo supports only the three sample questions shown above.")
    st.session_state.history.append({"q": q, "a": a})

# ===================== RESULTS (ONLY Q & A) =====================
if st.session_state.history:
    turn = st.session_state.history[-1]

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Results")

    st.markdown('<div class="qa">', unsafe_allow_html=True)
    st.markdown('<div class="qa-label">Question</div>', unsafe_allow_html=True)
    st.write(turn["q"])
    st.markdown('<div class="qa-label" style="margin-top:10px;">Answer</div>', unsafe_allow_html=True)
    st.write(turn["a"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
