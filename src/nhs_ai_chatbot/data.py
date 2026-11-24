import config
from embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_core.documents import Document


def add_to_chroma_db(chunks: list[Document]):
    # Initialise DB connection
    db = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    # Assign unique IDs to chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing document IDs in DB to prevent duplicates
    existing_items = db.get(
        include=[]
    )  # get ID list only - not documents, metadata etc.
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter for new chunks only
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Add new documents if present
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Assigns a unique identifier to each chunk metadata in the format:
    "source:page:chunk_index" (e.g. "data/brochure.pdf:2:1")
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", "unknown")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the chunk index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Save ID to metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_chroma_db():
    db = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    db.reset_collection()
