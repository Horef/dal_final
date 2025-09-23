# app.py
from pathlib import Path
import os
import sys
import torch
import streamlit as st
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minirag import MiniRAG, QueryParam
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc

# ===================== PAGE & THEME =====================
st.set_page_config(page_title="Uni-Assistant", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

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

# Hide "Press Enter to submit" hint globally
st.markdown(
    """
    <style>
    .stFormSubmitButton + div p {
        display: none;
    }
    .stTextInput p {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== CONSTANTS (no user knobs) =====================
LLM_MODEL = "bigscience/bloomz-560m"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORKING_DIR = "./Step0_res"                
#DATA_PATH = "./dataset/Technion/data/"    

# ===================== CACHED LOADERS =====================
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
_hf_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=False)

use_cuda = torch.cuda.is_available()
_dtype = torch.float16 if use_cuda else torch.float32

if use_cuda:
    torch.cuda.empty_cache()

_hf_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=_dtype,
    low_cpu_mem_usage=True,
    device_map={"": 0} if use_cuda else None,
    trust_remote_code=False,
)

_hf_model.eval()


if _hf_tokenizer.pad_token_id is None:
    if _hf_tokenizer.eos_token_id is not None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    else:
        _hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _hf_model.resize_token_embeddings(len(_hf_tokenizer))

if use_cuda:
    torch.cuda.empty_cache()



if torch.cuda.is_available():
    device_str = "cuda:0"
    device_arg = 0
    try:
        maj, minr = torch.cuda.get_device_capability(0)
        print(f"Detected GPU: {torch.cuda.get_device_name(0)} (capability {maj}.{minr})")
    except Exception as e:
        print(f"GPU info probe failed: {e}")
    _hf_model.to(device_str)
    print("Model successfully moved to CUDA:0")
else:
    device_str = "cpu"
    device_arg = -1
    print("CUDA is NOT available, running on CPU")

print(f"Final device_arg = {device_arg}")


_hf_pipe = pipeline(
    "text-generation",
    model=_hf_model,
    tokenizer=_hf_tokenizer,
    return_full_text=False,
)


async def hf_model_complete(prompt: str, **kwargs) -> str:
    """Async wrapper compatible with MiniRAG's awaited LLM interface."""
    gen = _hf_pipe(
        prompt,
        max_new_tokens=5000,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=_hf_tokenizer.pad_token_id,
        eos_token_id=_hf_tokenizer.eos_token_id,
    )
    return gen[0]["generated_text"].strip()


# ----------------------------
# Build MiniRAG with HF-only stack
# ----------------------------

_EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
_EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL).eval()

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=_EMB_TOKENIZER,
            embed_model=_EMB_MODEL,
        ),
    ),
)

# ===================== STATE =====================
if "question" not in st.session_state:
    st.session_state.question = ""
if "history" not in st.session_state:
    st.session_state.history = []  # [{q,a,sources}]
def load_from_history(idx: int): st.session_state.question = st.session_state.history[idx]["q"]

# ===================== HEADER =====================
st.markdown('<div class="header">', unsafe_allow_html=True)
c1, c2 = st.columns([0.15, 0.85])
with c1:
    st.markdown('<div class="logo">', unsafe_allow_html=True)
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH))
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="title"><h1>Uni-Assistant</h1><p>Ask anything about Technion courses, prerequisites, and study programs.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===================== SIDEBAR (no parameter controls) =====================
st.sidebar.subheader("History")
if st.session_state.history:
    for i, turn in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        st.sidebar.button(turn["q"], key=f"h_{idx}", on_click=load_from_history, args=(idx,), use_container_width=True)
else:
    st.sidebar.caption("No questions yet.")
st.sidebar.markdown('<div class="neutral">', unsafe_allow_html=True)
if st.sidebar.button("Clear history", use_container_width=True, key="clear_hist"): st.session_state.history = []
st.sidebar.markdown('</div><span class="muted">Powered by MiniRAG (HF-only stack).</span>', unsafe_allow_html=True)

# ===================== ASK =====================
# st.markdown('<div class="section">', unsafe_allow_html=True)
# st.markdown('<h3>Enter your question</h3>', unsafe_allow_html=True)
# left, right = st.columns([1, 0.2], vertical_alignment="center")
# with left:
#     st.session_state.question = st.text_input(
#         label="",
#         value=st.session_state.question,
#         placeholder="Enter your question here!",
#         label_visibility="collapsed",
#     )
# with right:
#     st.markdown('<div data-ask class="ask-wrap">', unsafe_allow_html=True)
#     ask_clicked = st.button("Ask", type="primary", use_container_width=True, key="ask_btn_real")
#     st.markdown('</div>', unsafe_allow_html=True)

with st.form(key="ask_form", clear_on_submit=False):
    st.session_state.question = st.text_input(
        label="",
        value=st.session_state.question,
        placeholder="Enter your question here!",
        label_visibility="collapsed",
    )
    ask_clicked = st.form_submit_button("Ask", use_container_width=True)

if ask_clicked:
    st.write(f"Submitted: {st.session_state.question}")

# ===================== QUERY (MiniRAG, no user params) =====================
def answer_with_sources(q: str):
    # Answer
    ans = rag.query(q, param=QueryParam(mode="mini"))
    # Retrieved context (optional “sources” view)
    ctx = ""
    try:
        ctx = rag.query(q, param=QueryParam(mode="mini", only_need_context=True))
    except Exception:
        pass
    # Build a single “source” block from context (MiniRAG returns a stitched context string)
    sources = []
    if ctx:
        sources.append({
            "title": "Retrieved context",
            "score": None,
            "chunk": ctx,
        })
    return ans, sources

if ask_clicked and st.session_state.question.strip():
    q = st.session_state.question.strip()
    with st.spinner("Thinking…"):
        try:
            a, sources = answer_with_sources(q)
        except Exception as e:
            a = f"Error: {e}"
            sources = []
    st.session_state.history.append({"q": q, "a": a, "sources": sources})

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
