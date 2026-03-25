import streamlit as st
import os
import pandas as pd

# Ensure these modules exist in your project structure
try:
    from rag.vector_store import create_documents, create_vector_store
    from rag.agent import create_rag_agent
except ImportError as e:
    st.error(f"Module Import Error: {e}. Check your folder structure (rag/ folder must have __init__.py)")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG AI App")
st.write("Ask questions from your Excel files using AI")

# -----------------------------
# Config
# -----------------------------
# Ensure this folder exists in your project root
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
            st.error(f"Data folder '{DATA_FOLDER}' not found. Please create it in the root directory.")
            return None

        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".xlsx", ".xls"))]

        if not files:
            st.warning(f"No Excel files found in '{DATA_FOLDER}'")
            return None

        # Load Excel files
        for file in files:
            file_path = os.path.join(DATA_FOLDER, file)
            try:
                # Using openpyxl for engine compatibility
                df = pd.read_excel(file_path, engine='openpyxl')
                df["source_file"] = file  # Track file origin
                all_data.append(df)
            except Exception as e:
                st.error(f"Error reading {file}: {e}")

        if not all_data:
            st.error("No valid Excel data loaded into the system.")
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # 1. Convert DataFrame rows to Document objects
        documents = create_documents(combined_df)

        # 2. Initialize the Vector Store (ChromaDB/FAISS)
        vector_store = create_vector_store(documents)

        # 3. Initialize the Hugging Face Pipeline Agent
        rag_agent = create_rag_agent(vector_store)

        return rag_agent

    except Exception as e:
        st.error(f"Setup failed during initialization: {e}")
        return None

# -----------------------------
# Initialize RAG
# -----------------------------
# FIXED: The function returns only rag_agent, so we only assign one variable
rag_agent = setup_rag()

# -----------------------------
# UI - Query Section
# -----------------------------
st.subheader("Ask Your Question")

with st.form("query_form"):
    query = st.text_input("Enter your question about the Excel data:")
    submit_button = st.form_submit_button("Ask AI")

if submit_button:
    if not query:
        st.warning("Please enter a question first.")
    elif rag_agent is None:
        st.error("RAG system is not initialized. Check the errors in the Debug section.")
    else:
        with st.spinner("Analyzing documents and generating answer..."):
            try:
                # Handling different possible agent structures
                if hasattr(rag_agent, "invoke"):
                    response = rag_agent.invoke(query)
                else:
                    response = rag_agent(query)
                
                st.success("AI Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error during query processing: {e}")

# -----------------------------
# Debug Section
# -----------------------------
with st.expander("System Debug Info"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Working Dir:** `{os.getcwd()}`")
        st.write(f"**Data Folder:** `{DATA_FOLDER}`")
    with col2:
        if os.path.exists(DATA_FOLDER):
            st.write("**Detected Files:**")
            st.json(os.listdir(DATA_FOLDER))
        else:
            st.write("❌ Folder not found")
