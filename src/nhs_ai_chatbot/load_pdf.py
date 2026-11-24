import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_pdfs(pdf_path_str: str) -> list[Document]:
    """
    Parses a single PDF or a directory of PDFs
    and returns a list of LangChain Document objects.
    """
    try:
        pdf_path = validate_pdf_path(pdf_path_str)
        documents = []

        files_to_process = (
            [pdf_path] if pdf_path.is_file() else list(pdf_path.glob("*.pdf"))
        )

        for file_path in files_to_process:
            logger.info(f"Loading PDF file: {file_path}")
            reader = PdfReader(file_path)

            # Extract text per page to preserve page indexing metadata
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_num + 1,  # Human-readable 1-based page index
                        },
                    )
                    documents.append(doc)
        logger.info(
            f"Loaded {len(documents)} pages from {pdf_path.name if pdf_path.is_file() else pdf_path}"
        )
        return documents
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading PDF: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading PDFs: {e}")
        return []


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


def split_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return splitter.split_documents(documents)
