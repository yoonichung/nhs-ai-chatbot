import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_embedding_function():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set in environment or secrets")
        raise RuntimeError("GEMINI_API_KEY environment variable is missing")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
