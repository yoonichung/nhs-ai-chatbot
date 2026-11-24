import os
import tempfile
from pathlib import Path
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai.chat_models import GoogleRateLimitError

import config
from embedding_function import get_embedding_function
from data import add_to_chroma_db, clear_chroma_db
from load_pdf import load_pdfs, split_documents
from query_data import query_rag

GITHUB_REPO_URL = "https://github.com//yoonichung/nhs-ai-chatbot"

st.set_page_config(
    page_title="NHS Brochure AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS INJECTION TO FIT SIDEBAR ON SCREEN ---
st.markdown(
    """
    <style>
        /* Reduce default top padding in sidebar */
        [data-testid="stSidebarUserContent"] {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }
        /* Tighten vertical spacing between sidebar widgets */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.3rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- HELPER: Fetch Indexed Documents ---
def get_indexed_documents():
    try:
        db = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
        data = db.get(include=["metadatas"])
        if not data or not data["metadatas"]:
            return []

        sources = set()
        for meta in data["metadatas"]:
            if meta and "source" in meta:
                filename = Path(meta["source"]).name
                sources.add(filename)
        return sorted(list(sources))
    except Exception:
        return []


# --- COMPACT SIDEBAR ---
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF Brochures",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📥 Process & Index", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Indexing..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    docs = load_pdfs(temp_dir)
                    chunks = split_documents(docs)
                    add_to_chroma_db(chunks)

                st.success("Indexed!")
                st.rerun()
        else:
            st.warning("Select PDFs first.")

    indexed_docs = get_indexed_documents()

    with st.expander(f"📄 Active Brochures ({len(indexed_docs)})", expanded=False):
        if indexed_docs:
            for doc in indexed_docs:
                st.caption(f"• `{doc}`")
        else:
            st.caption("Database is empty.")

    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        clear_chroma_db()
        st.success("Cleared!")
        st.rerun()

# --- MAIN APP ---
st.title("🏥 NHS Medical Brochure Assistant")

st.info(
    "👈 Upload brochure PDFs using the left sidebar to populate the knowledge base before asking questions."
)

st.warning(
    "⚠️ **Privacy Notice:** Answers are derived strictly from uploaded NHS brochures and no other external sources. "
    "Do **not** enter personal health details or patient names."
)

# Wrapped in a form so both hitting Enter and clicking "Ask Assistant" submit the query
with st.form(key="query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the risks of varicose vein surgery?",
    )
    submit_button = st.form_submit_button(label="Ask Assistant", type="primary")

if submit_button:
    if not indexed_docs:
        st.error(
            "Knowledge Base is empty! Upload a brochure PDF using the sidebar first."
        )
    elif user_query.strip():
        with st.spinner("Searching context..."):
            try:
                answer, sources = query_rag(user_query)

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Citations & Sources")
                if sources:
                    for source in sources:
                        st.caption(f"• `{source}`")
                else:
                    st.caption("No direct citations matched.")
            except GoogleRateLimitError:
                st.error(
                    "⚠️ Ran out of AI tokens/quota on both primary and fallback models."
                    "Please try again later."
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")
st.divider()
st.caption(
    f"This app is a proof of concept (PoC) and not for clinical use. "
    f"[View codebase on GitHub]({GITHUB_REPO_URL})"
)
