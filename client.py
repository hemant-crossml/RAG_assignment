from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 

from config import MODEL_ID, TEMPERATURE, TOP_P, TOP_K, MAX_TOKEN, EMBEDDING_ID
from cred import gemini_api_key

model = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        api_key=gemini_api_key,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_output_tokens=MAX_TOKEN,   # output length cap
    )


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_ID, 
    api_key=gemini_api_key
    )