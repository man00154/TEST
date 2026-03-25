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

        # 1. Check folder exists
        if not os.path.exists(DATA_FOLDER):
            st.error(f"Data folder '{DATA_FOLDER}' not found")
            return None

        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".xlsx", ".xls"))]

        if not files:
            st.warning("No Excel files found in data folder")
            return None

        # 2. Load Excel files
        for file in files:
            file_path = os.path.join(DATA_FOLDER, file)
            try:
                # Use openpyxl for better .xlsx support
                df = pd.read_excel(file_path, engine='openpyxl')
                df["source_file"] = file  
                all_data.append(df)
            except Exception as e:
                st.error(f"Error reading {file}: {e}")

        if not all_data:
            return None

        # 3. Process Data
        combined_df = pd.concat(all_data, ignore_index=True)
        documents = create_documents(combined_df)
        vector_store = create_vector_store(documents)

        # 4. Create RAG agent
        # We pass the vector_store here to link the AI to your data
        rag_agent = create_rag_agent(vector_store)

        return rag_agent

    except Exception as e:
        st.error(f"Setup failed: {e}")
        return None

# -----------------------------
# Initialize RAG
# -----------------------------
# FIXED: Only assigning one variable because setup_rag() returns one value
rag_agent = setup_rag()

# -----------------------------
# UI - Query Section
# -----------------------------
st.subheader("Ask Your Question")

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
                # Most agents use .invoke() or are directly callable
                if hasattr(rag_agent, "invoke"):
                    response = rag_agent.invoke(query)
                else:
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
    if os.path.exists(DATA_FOLDER):
        st.write(f"**Files in {DATA_FOLDER}:**", os.listdir(DATA_FOLDER))
