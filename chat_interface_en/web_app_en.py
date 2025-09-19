# web_app_en.py
from pathlib import Path
import json
import time
import requests
import streamlit as st

# ====== Page setup ======
st.set_page_config(page_title="Uni-Assistant", page_icon="🎓", layout="wide")

ASSETS_DIR = Path(__file__).parent
LOGO_PATH = ASSETS_DIR / "technion_logo.png"

# Optional: custom CSS (LTR + Inter font)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'Inter', sans-serif;
  direction: ltr;
  text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ====== Header ======
col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    if LOGO_PATH.exists():
        st.image(LOGO_PATH, width=140)
with col_title:
    st.title("Uni-Assistant")

st.caption("Ask anything about Technion courses, prerequisites, and study programs.")

# ====== Sidebar (settings) ======
st.sidebar.header("Settings")
BACKEND_URL = st.sidebar.text_input(
    "Backend base URL",
    value="http://localhost:8000",  # change to your server, e.g. http://127.0.0.1:8000
    help="Your MiniRAG API (e.g., FastAPI) base URL."
)
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 20, 5)
temperature = st.sidebar.slider("Answer creativity (temperature)", 0.0, 1.0, 0.2, 0.1)

# ====== Tabs ======
tab_ask, tab_index = st.tabs(["Ask", "Index data"])

# ====== ASK TAB ======
with tab_ask:
    question = st.text_input(
        "Enter your question:",
        placeholder="For example: What are the prerequisites for Differential Calculus?",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Call your backend /ask endpoint
            payload = {"query": question, "top_k": top_k, "temperature": temperature}
            try:
                with st.spinner("Thinking…"):
                    resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=60)
                if resp.status_code != 200:
                    st.error(f"Backend returned {resp.status_code}: {resp.text}")
                else:
                    data = resp.json()
                    # Expecting: {"answer": "...", "sources": [{"text": "...", "score": 0.42, "meta": {...}}, ...]}
                    st.subheader("Answer")
                    st.write(data.get("answer", "No answer field returned."))

                    sources = data.get("sources", [])
                    if sources:
                        st.subheader("Sources")
                        for i, src in enumerate(sources, 1):
                            with st.expander(f"Source {i} • score={src.get('score')!s}"):
                                st.write(src.get("text", ""))
                                meta = src.get("meta", {})
                                if meta:
                                    st.code(json.dumps(meta, ensure_ascii=False, indent=2))
                    else:
                        st.info("No sources returned.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Make sure it’s running and the URL is correct.")
            except Exception as e:
                st.exception(e)

# ====== INDEX TAB ======
with tab_index:
    st.write("Create/refresh the index for your dataset.")

    data_dir = st.text_input(
        "Data folder path",
        value="./dataset/Technion/data/",
        help="Folder with your course syllabi, policies, etc."
    )
    index_clicked = st.button("Build / Update Index")

    if index_clicked:
        try:
            with st.spinner("Indexing…"):
                resp = requests.post(f"{BACKEND_URL}/index", json={"data_dir": data_dir}, timeout=300)
            if resp.status_code != 200:
                st.error(f"Backend returned {resp.status_code}: {resp.text}")
            else:
                result = resp.json()
                st.success("Indexing completed.")
                st.code(json.dumps(result, ensure_ascii=False, indent=2))
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Make sure it’s running and the URL is correct.")
        except Exception as e:
            st.exception(e)
