from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def create_rag_agent(vector_store):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
