from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def create_documents(texts):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(texts)

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)
