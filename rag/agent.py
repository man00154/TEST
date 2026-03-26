from langchain_openai import ChatOpenAI
import streamlit as st

def create_rag_agent(vector_store):
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    def rag_chain(query):
        # ✅ ONLY THIS (no old method anywhere)
        docs = retriever.invoke(query)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer ONLY using the EXCEL files from C-Folder path.
        If not found, say "Not found".

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        return response.content

    return rag_chain
