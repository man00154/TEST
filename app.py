import streamlit as st
import os
import pandas as pd

from rag.vector_store import create_documents, create_vector_store
from rag.agent import create_rag_agent


st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG")
st.write("Ask questions from your Excel files using AI")


DATA_FOLDER = "C-Folder"   


@st.cache_data
def load_excel_data():
    all_texts = []

    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder '{DATA_FOLDER}' not found in repo")
        return []

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".xlsx")]

    if not files:
        st.warning("No Excel files found in data folder")
        return []

    for file in files:
        file_path = os.path.join(DATA_FOLDER, file)

        try:
            df = pd.read_excel(file_path)

            # Convert each row into text
            for _, row in df.iterrows():
                text = ", ".join(
                    [f"{col}: {row[col]}" for col in df.columns]
                )
                all_texts.append(text)

        except Exception as e:
            st.error(f" Error reading {file}: {e}")

    return all_texts


@st.cache_resource
def setup_rag():
    texts = load_excel_data()

    if not texts:
        return None, None

    try:
        documents = create_documents(texts)
        vector_store = create_vector_store(documents)

        rag_agent = create_rag_agent(vector_store)

        return vector_store, rag_agent

    except Exception as e:
        st.error(f" RAG Setup Error: {e}")
        return None, None

# Initialize
vector_store, rag_agent = setup_rag()


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
