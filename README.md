# NHS Brochure AI Assistant

A web interface for querying medical PDF brochures by question rather than reading full leaflets end-to-end. It uses retrieval-augmented generation (RAG) to find relevant passages from uploaded PDFs and passes them to an LLM to answer questions, showing the source filenames and page numbers alongside the response.

Built as a proof of concept using Streamlit, LangChain and Chroma. It uses Gemini for both embeddings and chat, chosen for simplicity rather than as an architectural requirement. Switching providers requires changes to the model clients.

**Use public, non-sensitive brochures only.** Questions and brochure text are sent to the model provider. This app is not clinically validated and should not be used for personal medical advice.

## How it works

```text
Indexing:  PDFs → Extract text → Split into chunks → Embed → Chroma

Questions: Question → Embed → Retrieve matching chunks from Chroma
                                   ↓
                        Question + context → LLM → Answer
                                                  + source pages
```

1. **Index brochures.** `pypdf` extracts text page by page. The text is split into chunks of up to 500 characters with a 50 character-overlap, embedded and stored in a local Chroma database.
2. **Retrieve context.** The question is embedded with the same model, and matched against Chroma to retrieve the top 5 chunks.
3. **Generate an answer.** The LLM receives the question and retrieved text, with instructions to answer only from that context or say the information is missing. A fallback model is used if the primary model call fails.
4. **Show sources.** The app displays the answer with the filenames and page numbers of the retrieved passages. These identify the context used; they do not verify every claim in the answer.

Each question is independent; the app does not maintain conversation history.

## Getting started

Requirements
- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- a Gemini API key with access to the models configured in `src/nhs_ai_chatbot/config.py`

From the repository root, install the dependencies:
```sh
uv sync --locked
```

Create `dev.env` in the repository root (and make sure it's in `.gitignore`):
```dotenv
GEMINI_API_KEY=your-api-key
```

The app loads `dev.env` automatically, or you can set `GEMINI_API_KEY` as an environment variable.

Start the app:
```sh
uv run --locked streamlit run src/nhs_ai_chatbot/app.py
```

Open the URL printed in the terminal, usually `http://localhost:8501`.

1. Upload PDFs in the sidebar. Example brochures are available in `sample_pdfs/`.
2. Click **Process & Index** and check that they appear under **Active Brochures**.
3. Enter a question and click **Ask Assistant**.

The sample brochures are not indexed automatically. The index is saved in `chroma_db/` and persists across local restarts. **Clear Database** removes all indexed content for the app instance.

### Model configuration

Model names are set in `src/nhs_ai_chatbot/config.py`. The current implementation uses a Gemini embedding model and primary/fallback chat models. Your API key needs to have access to the configured models.

If you change the embedding model or its dimensions, clear the database and re-index the PDFs.

## Tests

The test suite checks answers about the sample varicose vein surgery brochure using an LLM judge, including one intentionally incorrect expected answer.

Index the sample brochures, including `VV open Surgery.pdf`, set your API key, then run:
```sh
uv run --locked pytest -v
```

**Tests call live embedding and LLM APIs by default**, use quota and may incur charges. Judge results can vary, and quota errors cause tests to be skipped. The suite does not yet verify source accuracy or clinical safety.

To check test collection without making API calls:
```sh
uv run --locked pytest --collect-only -q
```

## Project structure

```text
src/nhs_ai_chatbot/
├── app.py                 # Streamlit interface
├── config.py              # Paths, env loading and model names
├── load_pdf.py            # PDF extraction and chunking 
├── embedding_function.py  # Embedding client setup
├── data.py                # Chroma indexing and reset helpers
├── query_data.py          # Retrieval, prompting and source labels
└── get_llm.py             # Primary and fallback LLM clients
tests/                     # Live answer evaluations
sample_pdfs/               # Example brochures
```

## Limitations and production considerations
The PoC keeps ingestion, retrieval and the UI in one application. Before using it as a production service, the main areas to address are:

- **Access control:** all visitors share the same database and can upload PDFs or clear it. Restrict these actions to administrators and enforce document access rules where needed.
- **Document storage:** original uploads are deleted after indexing, and the index lives on local disk. Use durable storage and stable document IDs; temporary upload paths currently make duplicate detection unreliable.
- **Retrieval quality:** scanned PDFs need OCR, and the app uses fixed chunk sizes and the top five matches without a relevance threshold. Test against a larger, clinically reviewed question set.
- **Networking & TLS:** clients currently use default network settings. Configure approved proxy URLs or model gateway endpoints and trusted CA certificates where required, for both embeddings and generation. Keep TLS verification enabled.
- **Operations and safety:** add rate limits, logging/monitoring, clearer error handling and automated checks. Review privacy, security and clinical suitability before patient-facing use.
