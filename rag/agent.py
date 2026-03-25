from langchain_openai import ChatOpenAI

def create_rag_agent(vector_store):
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # fast + cheap
        temperature=0
    )

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        return llm.invoke(prompt).content

    return rag_chain
