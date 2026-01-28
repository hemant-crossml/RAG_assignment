"""
This module provides utilities for creating, persisting, and loading
FAISS-based vector indexes using LangChain.

It enables:
- Building a FAISS vector store from a collection of documents and
  application-configured embeddings
- Saving the vector index to local storage
- Reloading an existing FAISS index for downstream retrieval or search

All operations are logged for observability, and exceptions are propagated
to ensure indexing or loading failures are not silently ignored.
"""

from langchain_community.vectorstores import FAISS

from client import embeddings
from config import FAISS_PATH
from logger_config import get_logger

logger = get_logger(__name__)


def create_faiss_index(documents):
    """
    Summary:
        Create and persist a FAISS vector index from provided documents.

    Args:
        documents (list): List of LangChain Document objects to index.

    Returns:
        None
    """
    try:
        logger.info(
            "Creating FAISS index | documents=%d | path=%s",
            len(documents),
            FAISS_PATH,
        )

        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_PATH)

        logger.info("FAISS index created and saved successfully")

    except Exception:
        logger.exception("Failed to create FAISS index")
        raise


def load_faiss_index():
    """
    Summary:
        Load an existing FAISS vector index from local storage.

    Args:
        None

    Returns:
        FAISS: Loaded FAISS vector store instance.
    """
    try:
        logger.info(
            "Loading FAISS index from path: %s",
            FAISS_PATH,
        )

        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        logger.info("FAISS index loaded successfully")
        return vectorstore

    except Exception:
        logger.exception("Failed to load FAISS index")
        raise
