"""
This script creates a Streamlit web application for the Technion University Curriculum Q&A system.
This file should only be run after the index has been created (and saved) using Step_0_index.py.

In order to run this file, use the command:
    streamlit run chat_interface/web_app.py from the root directory of the project.
"""

import streamlit as st
import argparse
from MiniRAG.minirag import MiniRAG, QueryParam
import pickle
import os
import sys

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG (HF-only)")
    parser.add_argument("--model", type=str, default="PHI",
                        help="PHI | aya | GLM | MiniCPM | qwen")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument("--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv")
    return parser.parse_args()

args = get_args()

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_rag_system():
    """Loads the RAG system."""
    # Loading the pickled RAG index saved during indexing
    index_path = os.path.join(args.workingdir, 'checkpoints', 'rag_final.pkl')
    if not os.path.exists(index_path):
        st.error(f"Index file not found at {index_path}. Please run Step_0_index.py first.")
        st.stop()
    with open(index_path, 'rb') as f:
        rag = pickle.load(f)
    return rag

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Uni-Assistant",
    page_icon="🎓",
    layout="centered"
)

# --- App Content ---
st.title("🎓 Uni-Assistant")
st.write("Ask any question about the Technion's courses, prerequisites, and content.")

# Load the RAG system
rag_system = load_rag_system()

# Get user input
question = st.text_input(
    "Ask your question here:",
    placeholder="e.g., What are the prerequisites for Advanced Calculus?"
)

# Handle the question
if question:
    with st.spinner("Searching the curriculum..."):
        # Get the answer from your RAG system
        try:
            answer = (rag_system.query(question, param=QueryParam(mode="mini"))
                        .replace("\n", "")
                        .replace("\r", ""))
        except Exception as e:
            st.error(f"An error occurred while querying the RAG system: {e}")
            st.stop()

        # Display the answer
        st.success("Here's what I found:")
        st.write(answer)

        # Optionally, display the sources used
        # TODO: I am not sure how to use this, need to check the MiniRAG docs
        # if answer["documents"]:
        #     with st.expander("Show Sources"):
        #         for doc in answer["documents"]:
        #             st.info(doc)