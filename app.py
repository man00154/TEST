from fastapi import FastAPI
from pydantic import BaseModel
import os

from rag.excel_loader import load_excel_files
from rag.vector_store import create_vector_store
from rag.agent import create_rag_agent

app = FastAPI()

DATA_FOLDER = "C-Folder"

# ----------------------------
# Load + Create RAG once
# ----------------------------
documents = load_excel_files(DATA_FOLDER)
vector_store = create_vector_store(documents)
rag_agent = create_rag_agent(vector_store)

# ----------------------------
# Request Model
# ----------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/query")
def query_excel(data: QueryRequest):
    try:
        response = rag_agent({"query": data.query})

        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }

    except Exception as e:
        return {"error": str(e)}
