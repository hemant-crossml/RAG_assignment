"""
This module provides utilities for splitting documents into smaller,
overlapping text chunks for downstream embedding and retrieval.

It uses LangChain’s `RecursiveCharacterTextSplitter` with application-
configured chunk size and overlap parameters to preserve semantic
continuity across chunks. All splitting operations are logged, and
errors are propagated to ensure failures are visible to the caller.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger_config import get_logger

logger = get_logger(__name__)


def split_documents(documents):
    """
    Summary:
        Split documents into smaller overlapping text chunks for embedding.

    Args:
        documents (list): List of LangChain Document objects to be split.

    Returns:
        list: List of chunked LangChain Document objects.
    """
    try:
        logger.info(
            "Splitting %d documents | chunk_size=%d, chunk_overlap=%d",
            len(documents),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        chunks = splitter.split_documents(documents)

        logger.info(
            "Document splitting completed | generated %d chunks",
            len(chunks),
        )

        return chunks

    except Exception:
        logger.exception("Failed to split documents")
        raise
