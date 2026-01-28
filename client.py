"""
This module initializes Google Gemini language and embedding models using
LangChain's Google Generative AI integrations.

It configures and instantiates:
- A Gemini chat-based large language model (LLM)
- A Gemini embeddings model

Configuration values such as model IDs, generation parameters, and token
limits are loaded from the application's configuration modules, while the
API key is sourced from secured credentials.

Initialization steps are logged, and any failures during setup will be
logged with full exception details and re-raised to prevent silent errors.
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 

from config import MODEL_ID, TEMPERATURE, TOP_P, TOP_K, MAX_TOKEN, EMBEDDING_ID
from cred import gemini_api_key
from logger_config import get_logger

logger = get_logger(__name__)

try:
    logger.info("Initializing Gemini LLM | model=%s", MODEL_ID)

    model = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        api_key=gemini_api_key,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_output_tokens=MAX_TOKEN,
    )

    logger.info("Gemini LLM initialized successfully")

except Exception:
    logger.exception(
        "Failed to initialize Gemini LLM | model=%s", MODEL_ID
    )
    raise


try:
    logger.info(
        "Initializing Gemini Embeddings | model=%s", EMBEDDING_ID
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_ID,
        api_key=gemini_api_key,
    )

    logger.info("Gemini Embeddings initialized successfully")

except Exception:
    logger.exception(
        "Failed to initialize Gemini Embeddings | model=%s",
        EMBEDDING_ID,
    )
    raise