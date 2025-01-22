from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_pdf(pdf_path_str: str):
    try:
        pdf_path = validate_pdf_path(pdf_path_str)

        if pdf_path.is_file():
            logger.info(f"Loading PDF file: {pdf_path}")
            pdf_loader = PyPDFLoader(pdf_path)
        else:
            logger.info(f"Loading PDF files from directory: {pdf_path}")
            pdf_loader = PyPDFDirectoryLoader(pdf_path, glob="*.pdf")

        documents = pdf_loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading PDF: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading PDFs: {e}")
        return None


def validate_pdf_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File or directory not found at {path_str}")
    if path.is_file():
        if path.suffix != ".pdf":
            raise ValueError(f"Path {path_str} is not a PDF file")
    elif path.is_dir():
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory {path_str}")
    else:
        f"The path {path_str} is neither a file nor a directory"
    return path


if __name__ == "__main__":
    print("!!!!!!")
    pdf_path_str = "/Users/yooni/Repos/nhs-ai-chatbot/sample_pdfs/VV open Surgery.pdf"
    doc = load_pdf(pdf_path_str)
    # print(doc)

# TODO: group pdfs according to files if multiple pages
