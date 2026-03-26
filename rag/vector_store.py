from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

def create_vector_store(texts):
    docs = [Document(page_content=t) for t in texts]

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectordb
