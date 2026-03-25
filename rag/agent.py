from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def create_rag_agent(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain
