import os

from docs_loader import load_documents
from text_splitter import split_documents
from faiss_store import create_faiss_index, load_faiss_index
from rag_chain import build_rag_chain
from config import DATA_DIR, FAISS_PATH
from logger_config import get_logger

logger = get_logger(__name__)


def main():
    """
    Summary:
        Entry point for the stateless RAG chatbot application.

    Args:
        None

    Returns:
        None
    """
    try:
        logger.info("Starting RAG chatbot application")

        # Build FAISS index if it does not exist
        if not os.path.exists(FAISS_PATH):
            logger.info("FAISS index not found. Creating new index.")
            print("📚 Building vector store...")

            docs = load_documents(DATA_DIR)
            chunks = split_documents(docs)
            create_faiss_index(chunks)

        logger.info("Loading FAISS index")
        vectorstore = load_faiss_index()

        logger.info("Building RAG chain")
        rag_chain = build_rag_chain(vectorstore)

        logger.info("RAG chatbot ready for interaction")
        print("\n🤖 Stateless RAG Chatbot is ready! Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ")

                if user_input.lower() in {"exit", "quit", "q"}:
                    logger.info("User requested shutdown")
                    print("👋 Goodbye!")
                    break

                response = rag_chain.invoke(user_input)
                print("\nAssistant:", response.content, "\n")

            except KeyboardInterrupt:
                logger.info("Chat interrupted by user (KeyboardInterrupt)")
                print("\n👋 Goodbye!")
                break

            except Exception:
                logger.exception("Error during chat interaction")
                print("⚠️ An error occurred while processing your request.")

        logger.info("RAG chatbot application stopped")

    except Exception:
        logger.critical(
            "Fatal error occurred during application startup",
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
