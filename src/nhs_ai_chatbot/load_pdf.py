from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_pdfs(pdf_path_str: str) -> list[Document]:
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


def split_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    # TODO (later): need to think about token limits (1 token = ~4 characters so 500 tokens = 125 characters) but needto think about e.g. response too
    # all sample pdfs = 37508 characters so need ~9352 tokes just for reading in
    split_documents = []
    for doc in documents:
        split_documents.extend(splitter.split_text(doc.page_content))
        # TODO 1. need to split document instead so that we have the chunk - figure out why it wasnt working
    return split_documents


if __name__ == "__main__":
    print("!!!!!!")

    from embedding_function import get_embedding_function

    pdf_path_str = "/Users/yooni/Repos/nhs-ai-chatbot/sample_pdfs/VV open Surgery.pdf"
    docs = load_pdfs(pdf_path_str)
    chunks = split_documents(docs)
    print(len(chunks))
    print(chunks[0])
    embeddings = get_embedding_function()
    print(embeddings)
    print(type(embeddings))
    # print(len(docs))
