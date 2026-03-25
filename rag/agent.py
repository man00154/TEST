from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import HuggingFaceHub

def create_rag_agent(vector_store):
    retriever = vector_store.as_retriever()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa
