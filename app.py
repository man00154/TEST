import streamlit as st
import os
import pandas as pd

from rag.vector_store import create_documents, create_vector_store
from rag.agent import create_rag_agent


st.set_page_config(page_title="Excel RAG App", layout="wide")

st.title("Excel RAG")
st.write("Ask questions from your Excel files using AI")


DATA_FOLDER = "C-Folder"   



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
