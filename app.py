import streamlit as st
import os
import pandas as pd

from rag.vector_store import create_documents, create_vector_store
from rag.agent import create_rag_agent

st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG AI App")
st.write("Ask questions from your Excel files using AI")

DATA_FOLDER = "C-Folder"


# -----------------------------
# RAG Setup
# -----------------------------
@st.cache_resource
def setup_rag():
    try:
        all_data = []

        if not os.path.exists(DATA_FOLDER):
            st.error(f"Data folder '{DATA_FOLDER}' not found")
            return None

        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".xlsx", ".xls"))]

        if not files:
            st.warning("No Excel files found")
            return None

        for file in files:
            file_path = os.path.join(DATA_FOLDER, file)

            try:
                # ✅ FIX: Handle xls vs xlsx properly
                if file.endswith(".xlsx"):
                    df = pd.read_excel(file_path, engine="openpyxl")
                else:
                    df = pd.read_excel(file_path, engine="xlrd")

                df["source_file"] = file
                all_data.append(df)

            except Exception as e:
                st.warning(f"Skipping {file}: {e}")

        if not all_data:
            st.error("No valid Excel files loaded")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)

        # ✅ FIX: Convert DataFrame → text properly
        documents = create_documents(combined_df)

        vector_store = create_vector_store(documents)

        rag_agent = create_rag_agent(vector_store)

        return rag_agent

    except Exception as e:
        st.error(f"Setup failed: {e}")
        return None


rag_agent = setup_rag()

# -----------------------------
# UI
# -----------------------------
st.subheader("Ask Your Question")

with st.form("query_form"):
    query = st.text_input("Enter your query:")
    submit = st.form_submit_button("Ask AI")

if submit:
    if not query:
        st.warning("Enter a question")
    elif rag_agent is None:
        st.error("RAG not initialized")
    else:
        with st.spinner("Thinking..."):
            try:
                response = rag_agent(query)
                st.success("Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")


# -----------------------------
# Debug
# -----------------------------
with st.expander("Debug Info"):
    st.write("Working Dir:", os.getcwd())
    if os.path.exists(DATA_FOLDER):
        st.write("Files:", os.listdir(DATA_FOLDER))
