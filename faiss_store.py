from langchain_community.vectorstores import FAISS
from client import embeddings
from config import FAISS_PATH


def create_faiss_index(documents):
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_PATH)


def load_faiss_index():

    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
