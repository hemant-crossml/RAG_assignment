"""
This module provides utilities for loading PDF documents from a directory
using LangChain document loaders.

It supports recursive discovery of PDF files and lazy loading via
`DirectoryLoader` and `PyPDFLoader`, enabling efficient ingestion of large
document collections. All document loading operations are logged, and
errors are propagated to ensure failures are visible to the caller.
"""
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)

from logger_config import get_logger

logger = get_logger(__name__)


def load_documents(data_dir: str):
    """
    Summary:
        Lazily load PDF documents from the specified directory.

    Args:
        data_dir (str): Path to the directory containing PDF files.

    Returns:
        list: List of loaded LangChain Document objects.
    """
    try:
        logger.info(
            "Lazily loading PDF documents from directory: %s",
            data_dir,
        )

        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )

        documents = list(loader.lazy_load())

        logger.info(
            "Successfully loaded %d documents", len(documents)
        )
        return documents

    except Exception:
        logger.exception(
            "Failed to load documents from directory: %s",
            data_dir,
        )
        raise
