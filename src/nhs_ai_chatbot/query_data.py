from pathlib import Path
import config
from embedding_function import get_embedding_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from get_llm import get_llm

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
If you cannot find the answer in the context, state clearly that the brochure does not provide this information.

Context:
{context}

---

Question: {question}
"""


def query_rag(query_text: str) -> tuple[str, list[str]]:

    # 1. Prepare the DB and load embedding function
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # 2. Search DB for top k relevant chunks (Similarity Search)
    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        return "No relevant brochure context found.", []

    # 3. Combine context text from matching chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 4. Format prompt template with retrieved context and user query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 5. Send prompt to Gemini LLM (Uses GEMINI_API_KEY environment variable)
    model = get_llm()
    response = model.invoke(prompt)
    response_text = response.text

    # 6. Extract source metadata citations
    sources = []
    for doc, _score in results:
        page = doc.metadata.get("page", "unknown")

        source_file = Path(doc.metadata.get("source", "")).name

        formatted_source = f"{source_file} (Page {page})"
        sources.append(formatted_source)

    # Deduplicate while preserving order
    sources = list(dict.fromkeys(sources))
    return response_text, sources
