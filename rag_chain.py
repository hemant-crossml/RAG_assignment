"""
This module defines the construction of the Retrieval-Augmented Generation
(RAG) inference chain using LangChain Expression Language (LCEL).

It combines:
- A vector store retriever for fetching relevant document chunks
- A structured prompt template enforcing context-grounded responses
- A generative language model for answer synthesis

The resulting runnable forms an end-to-end RAG pipeline that maps user
questions to retrieved context and generates responses based strictly
on that context. All construction steps are logged, and errors are
propagated to prevent silent failures.
"""
from langchain_core.runnables import RunnablePassthrough

from prompt import get_rag_prompt
from client import model
from logger_config import get_logger

logger = get_logger(__name__)


def build_rag_chain(vectorstore):
    """
    Summary:
        Build a Retrieval-Augmented Generation (RAG) chain using LCEL.

    Args:
        vectorstore: Vector store instance used for document retrieval.

    Returns:
        Runnable: Configured LangChain runnable representing the RAG pipeline.
    """
    try:
        logger.info("Building RAG chain")

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        prompt = get_rag_prompt()

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
        )

        logger.info("RAG chain built successfully")
        return chain

    except Exception:
        logger.exception("Failed to build RAG chain")
        raise
