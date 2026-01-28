from langchain_core.runnables import RunnablePassthrough

from prompt import get_rag_prompt
from client import model



def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = get_rag_prompt()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
    )

    return chain
