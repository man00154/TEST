import streamlit as st
from rag.vector_store import create_documents, create_vector_store
from rag.agent import create_rag_agent

st.set_page_config(page_title="Excel AI Agent", layout="wide")

st.title("📊 Excel RAG AI Agent Test")

# Load data
@st.cache_data
def load_data():
    return load_excel_files("C-Folder")


df = load_data()

st.subheader("📁 Data Preview")
st.dataframe(df.head())

# Build RAG
@st.cache_resource
def build_rag():
    docs = create_documents(df)
    vs = create_vector_store(docs)
    agent = create_rag_agent(vs)
    return agent

agent = build_rag()

# Query UI
st.subheader("🔍 Ask Questions")
query = st.text_input("Ask about any row, column, customer, capacity, etc.")

if st.button("Run Query"):
    if query:
        with st.spinner("Thinking..."):
            result = agent(query)

            st.success("Answer:")
            st.write(result['result'])

            st.subheader("📄 Source Data")
            for doc in result['source_documents'][:3]:
                st.write(doc.page_content)
