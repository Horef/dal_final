from pathlib import Path
import json
import requests
import streamlit as st

st.set_page_config(page_title="Uni-Assistant", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# ===================== THEME (same as demo) =====================
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
.section h3{margin:0 0 8px 0;font-size:20px;font-weight:700}

/* Inputs */
.stTextInput > div > div > input{height:44px;border-radius:12px}

/* Default buttons = white */
.stButton > button{
  border-radius:12px; height:44px; font-weight:600;
  background:#ffffff; border:1px solid #e5e7eb; color:#111827;
}

/* Sources */
.source-title{font-weight:700;margin:6px 0}
.source-box{border:1px solid #eef2f7;border-radius:12px;padding:10px 12px;background:#fcfcfd;white-space:pre-wrap;word-break:break-word}

/* Sidebar history & controls */
[data-testid="stSidebar"] .stButton > button{
  border:1px solid #e5e7eb; background:#fff; color:#111827;
  border-radius:10px; padding:8px 10px; height:auto; line-height:1.25;
  white-space:normal; text-align:left; overflow-wrap:anywhere;
}
[data-testid="stSidebar"] .muted{color:#9ca3af; font-size:12.5px}
[data-testid="stSidebar"] .neutral > button{
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
      if (t === 'Ask') {
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
    st.session_state.history = []  # [{q,a,sources}]
def load_from_history(idx: int): st.session_state.question = st.session_state.history[idx]["q"]

# ===================== HEADER =====================
st.markdown('<div class="header">', unsafe_allow_html=True)
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown('<div class="logo">', unsafe_allow_html=True)
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH))
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="title"><h1>Uni-Assistant</h1><p>Ask anything about Technion courses, prerequisites, and study programs.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===================== SIDEBAR =====================
# st.sidebar.header("Settings")
# BACKEND_URL = st.sidebar.text_input("Backend base URL", value="http://localhost:8000")
# top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 20, 5)
# temperature = st.sidebar.slider("Answer creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
# data_dir = st.sidebar.text_input("Data folder (for indexing)", value="./dataset/Technion/data/")

# if st.sidebar.button("Build / Update Index", use_container_width=True, key="index_btn"):
#     try:
#         with st.spinner("Indexing…"):
#             r = requests.post(f"{BACKEND_URL}/index", json={"data_dir": data_dir}, timeout=300)
#         if r.status_code != 200:
#             st.error(f"Backend {r.status_code}: {r.text}")
#         else:
#             st.success("Indexing completed.")
#             st.code(json.dumps(r.json(), ensure_ascii=False, indent=2))
#     except Exception as e:
#         st.exception(e)

# st.sidebar.markdown("---")
st.sidebar.subheader("History")
if st.session_state.history:
    for i, turn in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        st.sidebar.button(turn["q"], key=f"h_{idx}", on_click=load_from_history, args=(idx,), use_container_width=True)
else:
    st.sidebar.caption("No questions yet.")
st.sidebar.markdown('<div class="neutral">', unsafe_allow_html=True)
if st.sidebar.button("Clear history", use_container_width=True, key="clear_hist"): st.session_state.history = []
st.sidebar.markdown('</div><span class="muted">Backend calls use /ask and /index.</span>', unsafe_allow_html=True)

# ===================== ASK =====================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h3>Enter your question</h3>', unsafe_allow_html=True)
left, right = st.columns([1, 0.2], vertical_alignment="center")
with left:
    st.session_state.question = st.text_input(
        label="",
        value=st.session_state.question,
        placeholder="For example: Can I take Linear Algebra before Calculus 1?",
        label_visibility="collapsed",
    )
with right:
    st.markdown('<div data-ask class="ask-wrap">', unsafe_allow_html=True)
    ask_clicked = st.button("Ask", type="primary", use_container_width=True, key="ask_btn_real")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== BACKEND CALL =====================
if ask_clicked and st.session_state.question.strip():
    payload = {"query": st.session_state.question.strip(), "top_k": top_k, "temperature": temperature}
    try:
        with st.spinner("Thinking…"):
            r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"Backend {r.status_code}: {r.text}")
        else:
            data = r.json()
            st.session_state.history.append({
                "q": payload["query"],
                "a": data.get("answer", "No answer returned."),
                "sources": [
                    {
                        "title": s.get("title") or s.get("meta", {}).get("title") or f"Source {i+1}",
                        "score": s.get("score"),
                        "chunk": s.get("text") or s.get("chunk") or "",
                    }
                    for i, s in enumerate(data.get("sources", []))
                ],
            })
    except Exception as e:
        st.exception(e)

# ===================== RESULTS =====================
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

    if turn["sources"]:
        with st.expander("Sources", expanded=False):
            for i, s in enumerate(turn["sources"], 1):
                title = s.get("title") or f"Source {i}"
                score = s.get("score")
                head = f"{i}. {title}" + (f" • score {score:.2f}" if isinstance(score, (int, float)) else "")
                st.markdown(f'<div class="source-title">{head}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="source-box">{s.get("chunk","")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
