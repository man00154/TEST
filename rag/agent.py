from langchain_openai import ChatOpenAI
import streamlit as st

def create_rag_agent(vector_store):
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]   # secure
    )

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an intelligent data assistant.

        Answer ONLY from the context below.
        If answer is not present, say "Not found".

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        return response.content

    return rag_chain
