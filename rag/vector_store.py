from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def create_documents(df):
    documents = []

    # Convert each row into readable text
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=text))

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    return splitter.split_documents(documents)


def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)
