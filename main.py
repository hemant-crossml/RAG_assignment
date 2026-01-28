import os
from docs_loader import load_documents
from text_splitter import split_documents
from faiss_store import create_faiss_index, load_faiss_index
from rag_chain import build_rag_chain
from config import DATA_DIR, FAISS_PATH


def main():
    # Build FAISS index once
    if not os.path.exists(FAISS_PATH):
        print("📚 Building vector store...")
        docs = load_documents(DATA_DIR)
        chunks = split_documents(docs)
        create_faiss_index(chunks)

    vectorstore = load_faiss_index()
    rag_chain = build_rag_chain(vectorstore)

    print("\n🤖 Stateless RAG Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = rag_chain.invoke(user_input)
        print("\nAssistant:", response.content, "\n")


if __name__ == "__main__":
    main()
