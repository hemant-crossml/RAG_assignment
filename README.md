# RAG_assignment

# Stateless RAG Chatbot with LangChain & Gemini

A production-ready, stateless Retrieval-Augmented Generation (RAG) chatbot built using LangChain LCEL, Gemini LLM, FAISS vector store, and structured logging.
The system is designed with modular architecture, fail-fast behavior, and strong guardrails against hallucinations.

---

## Features

- Stateless interactive CLI chatbot
- Retrieval-Augmented Generation (RAG) using FAISS
- Gemini LLM and Gemini Embedding models
- LangChain LCEL-based explicit RAG pipeline
- PDF document ingestion using DirectoryLoader and PyPDFLoader
- Configurable recursive text splitting
- Structured, hallucination-safe prompt design
- Centralized logging with rotating file handler
- Fail-fast startup and robust error handling
- Clean, modular, production-style codebase

---

## Installation

### Clone the repository
```bash
git clone https://github.com/hemant-crossml/RAG_assignment
cd RAG_assignment
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Application Configuration
Update `config.py` to adjust:
- Model IDs (LLM & embeddings)
- Chunk size and overlap
- Retrieval parameters
- FAISS index path
- Data directory location

---

## Usage

### Add documents
Place your PDF documents inside the configured data directory:

```text
data/
 ├── document1.pdf
 ├── document2.pdf
```

### Run the chatbot
```bash
python main.py
```

### Interact with the chatbot
```text
You: What is the refund policy?
Assistant: Customers can request a refund within 30 days of purchase.
```

Exit with:
```text
You: exit
```

---

## Tools & Capabilities

- Document ingestion via DirectoryLoader and PyPDFLoader
- Recursive text splitting for optimized embeddings
- FAISS vector similarity search
- Gemini-powered reasoning with strict context grounding
- Centralized logging and observability
- Stateless design for scalability and reliability

---

## Notes

- The chatbot does not store conversation history.
- If an answer is not found in the documents, the assistant responds with:
  "I don't know."