from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="data/",                 # folder containing PDFs
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

vectorstore.save_local("faiss_index")

# Later...
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "What is the main topic discussed in the documents?"

response = qa_chain(query)

print("Answer:\n", response["result"])
print("\nSources:")
for doc in response["source_documents"]:
    print(doc.metadata)