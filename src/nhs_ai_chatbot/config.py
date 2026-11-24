from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


CONFIG_FILE = Path(__file__).resolve()
BASE_DIR = CONFIG_FILE.parent.parent.parent

env_file = BASE_DIR / "dev.env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
    logger.info("Loaded environment variables from dev.env")

CHROMA_PATH = str(BASE_DIR / "chroma_db")
DATA_PATH = str(BASE_DIR / "sample_pdfs")

EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
