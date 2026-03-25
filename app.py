import streamlit as st
import os
import pandas as pd

# Ensure these modules exist in your project structure
try:
    from rag.vector_store import create_documents, create_vector_store
    from rag.agent import create_rag_agent
except ImportError as e:
    st.error(f"Module Import Error: {e}. Check your folder structure.")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG AI App")
st.write("Ask questions from your Excel files using AI")

# -----------------------------
# Config
# -----------------------------
DATA_FOLDER = "C-Folder"   

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
            return None

        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".xlsx", ".xls"))]

        if not files:
            st.warning("No Excel files found in data folder")
            return None

        # Load Excel files
        for file in files:
            file_path = os.path.join(DATA_FOLDER, file)
            try:
                # Use openpyxl as engine for better .xlsx support
                df = pd.read_excel(file_path)
                df["source_file"] = file  # Track file origin
                all_data.append(df)
            except Exception as e:
                st.error(f"Error reading {file}: {e}")

        if not all_data:
            st.error("No valid Excel data loaded")
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to documents
        documents = create_documents(combined_df)

        # Create vector store
        vector_store = create_vector_store(documents)

        # Create RAG agent
        rag_agent = create_rag_agent(vector_store)

        return rag_agent

    except Exception as e:
        st.error(f"Setup failed: {e}")
        return None

# -----------------------------
# Initialize RAG
# -----------------------------
# We only need the rag_agent for the UI
rag_agent = setup_rag()

# -----------------------------
# UI - Query Section
# -----------------------------
st.subheader("Ask Your Question")

# Use a form to prevent the app from rerunning on every keystroke
with st.form("query_form"):
    query = st.text_input("Enter your query:")
    submit_button = st.form_submit_button("Ask AI")

if submit_button:
    if not query:
        st.warning("Please enter a question")
    elif rag_agent is None:
        st.error("RAG system not initialized. Check errors above.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Assuming rag_agent is a callable or has an .invoke() method
                # Adjust 'response = rag_agent(query)' if your agent uses a different method
                response = rag_agent(query)
                
                st.success("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error during query: {e}")

# -----------------------------
# Debug Section
# -----------------------------
with st.expander("Debug Info"):
    st.write(f"**Current working directory:** `{os.getcwd()}`")
    
    st.write("**Files in data folder:**")
    if os.path.exists(DATA_FOLDER):
        st.write(os.listdir(DATA_FOLDER))
    else:
        st.write("❌ data folder not found")
