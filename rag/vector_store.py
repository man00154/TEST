from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


def create_documents(df):
    docs = []

    for _, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=content))

    return docs


def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
