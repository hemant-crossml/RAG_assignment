import os 

from dotenv import load_dotenv

load_dotenv()


# Read Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file")
