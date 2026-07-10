import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    TOP_K
)

from src.rag import answer_question


st.set_page_config(
    page_title="Textbook AI Assistant",
    page_icon="📘",
    layout="wide"
)


def check_index_files():
    """
    Check whether FAISS index and chunks.json exist.
    """

    missing_files = []

    if not os.path.exists(FAISS_INDEX_PATH):
        missing_files.append(FAISS_INDEX_PATH)

    if not os.path.exists(CHUNKS_PATH):
        missing_files.append(CHUNKS_PATH)

    return missing_files


st.title("📘 Textbook AI Assistant")

st.markdown(
    """
    Ask questions from the **Kaveri English Textbook for Grade 9**.

    The assistant answers using the indexed textbook content and shows source citations.
    """
)

missing_files = check_index_files()

if missing_files:
    st.error("Vector index files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.code(file)

    st.warning(
        "Please run the textbook processing script first: "
        "`python scripts/process_textbook.py`"
    )

    st.stop()


with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Number of textbook chunks to retrieve",
        min_value=3,
        max_value=10,
        value=TOP_K,
        step=1
    )

    st.markdown("---")

    st.subheader("Example Questions")

    example_1 = "What is the Kaveri textbook designed to help learners do?"
    example_2 = "What role does language education play?"
    example_3 = "How does the textbook use technology?"
    example_4 = "What is mentioned about Indian Knowledge Systems?"

    st.write(example_1)
    st.write(example_2)
    st.write(example_3)
    st.write(example_4)


question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is the Kaveri textbook designed to help learners do?",
    height=120
)


generate_button = st.button(
    "Generate Answer",
    type="primary"
)


if generate_button:

    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Searching textbook and generating answer..."):
        try:
            result = answer_question(
                question=question,
                top_k=top_k
            )

        except Exception as error:
            st.error("An error occurred while generating the answer.")
            st.exception(error)
            st.stop()

    st.markdown("## Answer")

    st.write(result["answer"])

    st.markdown("---")

    st.markdown("## Source Citations")

    for index, citation in enumerate(result["citations"], start=1):

        with st.expander(
            f"Source {index} | Page {citation['page']} | Score {citation['score']:.4f}"
        ):
            st.write(f"**Source PDF:** {citation['source']}")
            st.write(f"**Page:** {citation['page']}")
            st.write(f"**Chunk ID:** {citation['chunk_id']}")
            st.write(f"**Similarity Score:** {citation['score']:.4f}")

            st.markdown("**Text Preview:**")
            st.write(citation["preview"])


else:
    st.info("Enter a textbook question and click Generate Answer.")