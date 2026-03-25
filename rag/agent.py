from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def create_rag_agent(vector_store):
    retriever = vector_store.as_retriever()

    # Local model (no API key needed)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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

        return llm.invoke(prompt)

    return rag_chain
