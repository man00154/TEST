import streamlit as st
import os

from rag.excel_loader import load_excel_files
from rag.vector_store import create_vector_store
from rag.agent import create_rag_agent

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Excel RAG AI", layout="wide")

st.title("📊 Excel RAG AI Assistant")
st.write("Ask questions from your Excel files (sheet, row, column aware)")

DATA_FOLDER = "C-Folder"

# -----------------------------
# Cache RAG Setup
# -----------------------------
@st.cache_resource
def setup_rag():
    try:
        if not os.path.exists(DATA_FOLDER):
            st.error(f"Folder '{DATA_FOLDER}' not found")
            return None, None

        st.info("Loading Excel files...")

        documents = load_excel_files(DATA_FOLDER)

        if not documents:
            st.error("No data found in Excel files")
            return None, None

        st.info(f"Loaded {len(documents)} rows from Excel")

        vector_store = create_vector_store(documents)
        rag_agent = create_rag_agent(vector_store)

        return vector_store, rag_agent

    except Exception as e:
        st.error(f"Setup error: {e}")
        return None, None


vector_store, rag_agent = setup_rag()

# -----------------------------
# User Input
# -----------------------------
query = st.text_input("🔍 Ask your question:")

if st.button("Ask AI"):
    if not query:
        st.warning("Please enter a question")

    elif rag_agent is None:
        st.error("RAG system not initialized")

    else:
        with st.spinner("Thinking..."):
            try:
                response = rag_agent({"query": query})

                st.success("Answer:")
                st.write(response["result"])

                with st.expander("📄 Source Data"):
                    for doc in response["source_documents"]:
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# Debug Info
# -----------------------------
with st.expander("⚙️ Debug Info"):
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir("."))
