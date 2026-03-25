import streamlit as st
import os
import pandas as pd

from rag.vector_store import create_documents, create_vector_store
from rag.agent import create_rag_agent


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG AI App")
st.write("Ask questions from your Excel files using AI")


# -----------------------------
# Config
# -----------------------------
DATA_FOLDER = "data"   


# -----------------------------
# RAG Setup Function
# -----------------------------
@st.cache_resource
def setup_rag():
    try:
        all_data = []

        # Check folder exists
        if not os.path.exists(DATA_FOLDER):
            st.error(f"Data folder '{DATA_FOLDER}' not found")
            return None, None

        files = os.listdir(DATA_FOLDER)

        if not files:
            st.warning("No files found in data folder")
            return None, None

        # Load Excel files
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                file_path = os.path.join(DATA_FOLDER, file)

                try:
                    df = pd.read_excel(file_path)
                    df["source_file"] = file  # Track file origin
                    all_data.append(df)

                except Exception as e:
                    st.error(f"Error reading {file}: {e}")

        if not all_data:
            st.error("No valid Excel data loaded")
            return None, None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to documents
        documents = create_documents(combined_df)

        # Create vector store
        vector_store = create_vector_store(documents)

        # Create RAG agent
        rag_agent = create_rag_agent(vector_store)

        return vector_store, rag_agent

    except Exception as e:
        st.error(f"Setup failed: {e}")
        return None, None


# -----------------------------
# Initialize RAG
# -----------------------------
vector_store, rag_agent = setup_rag()


# -----------------------------
# UI - Query Section
# -----------------------------
st.subheader("Ask Your Question")

query = st.text_input("Enter your query:")

if st.button("Ask AI"):
    if not query:
        st.warning("Please enter a question")

    elif rag_agent is None:
        st.error("RAG system not initialized. Check errors above.")

    else:
        with st.spinner("Thinking..."):
            try:
                response = rag_agent(query)

                st.success("Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"Error during query: {e}")


# -----------------------------
# Debug Section
# -----------------------------
with st.expander("Debug Info"):
    st.write("Current working directory:")
    st.write(os.getcwd())

    st.write("Files in project root:")
    st.write(os.listdir("."))

    st.write("Files in data folder:")
    if os.path.exists(DATA_FOLDER):
        st.write(os.listdir(DATA_FOLDER))
    else:
        st.write("data folder not found")
