# web_app_demo_en.py
from pathlib import Path
import streamlit as st
import time

# ====== Page setup ======
st.set_page_config(page_title="Uni-Assistant (Demo)", page_icon="🎓", layout="wide")

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

st.caption("Ask anything about Technion courses, prerequisites, and study programs. (Demo mode)")

# ====== Sidebar (demo knobs) ======
st.sidebar.header("Demo Settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 20, 5)
latency_ms = st.sidebar.slider("Simulated latency (ms)", 0, 3000, 600, 100)

# ====== Main ======
question = st.text_input(
    "Enter your question:",
    placeholder="For example: What are the prerequisites for Differential Calculus?",
)
ask_clicked = st.button("Ask (Demo)", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            time.sleep(latency_ms / 1000.0)

        # Stubbed answer + pretend sources
        st.subheader("Answer")
        st.write(
            "In most tracks, Differential Calculus requires a basic mathematics placement or successful completion "
            "of an introductory math course. For exact rules, consult your program’s handbook."
        )

        st.subheader("Sources (demo)")
        for i in range(1, top_k + 1):
            with st.expander(f"Source {i} • score≈{round(1.0 - i*0.07, 2)}"):
                st.write(
                    "Excerpt: “…Students must meet the mathematics placement requirement or complete the "
                    "introductory course prior to enrolling in Differential Calculus…”"
                )
                st.code(
                    '{"course": "Differential Calculus", "document": "Program Handbook 2024–25", "page": %d}' % (i + 3)
                )
