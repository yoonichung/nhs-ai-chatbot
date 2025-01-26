from embedding_function import get_embedding_function
from langchain.vectorestores.chroma import Chroma
from langchain.schema.document import Document


def add_to_chroma(chunks: list[Document]):
    # TODO 2. figure out chroma
    db = Chroma(
        persis_directory=config.CHROMA_PATH, embedding_function=get_embedding_function()
    )
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()

    # TODO 3. put this into a helper function
    # essentially getting a unique id for each chunk that is source:page:chunk_index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = chunk_id
