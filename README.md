# DocuAgent Terminal CLI

DocuAgent now includes a standalone terminal-first interface in [docu_agent_cli.py](docu_agent_cli.py). It can ingest a new document into a local Chroma database or open a direct chat session against an existing knowledge base, all from one interactive console flow.

## What The CLI Does

The CLI reads local `.pdf`, `.txt`, and `.md` files, splits them into chunks, stores the vectors in Chroma using Google Gemini embeddings, and then starts an interactive retrieval-augmented chat loop.

If the local context does not contain the answer, the assistant responds with:

```text
I cannot find this information in the provided documentation.
```

## Project Structure

```text
core_engine/
├── .env
├── .gitignore
├── README.md
├── docu_agent_cli.py
├── ingestion.py
├── main.py
├── retrieval.py
├── requirements.txt
└── data_source/
    └── sample.txt
```

## Project Structure

```text
core_engine/
├── .env
├── .gitignore
├── README.md
├── main.py
├── ingestion.py
├── retrieval.py
├── requirements.txt
└── data_source/
    └── sample.txt
```

## Tools And Dependencies

The CLI uses these packages:

- LangChain for document loading, splitting, and retrieval orchestration
- Chroma for local vector storage
- Google GenAI embeddings for semantic indexing
- Gemini chat models for grounded answer generation
- python-dotenv for environment variable loading
- pypdf for PDF reading support

## Environment Setup

1. Add your Google AI Studio key to [.env](.env) using the `GEMINI_API_KEY` variable.
2. Keep the real key private. Do not commit `.env` to source control.
3. Put your source documents inside [data_source/](data_source/).

Example:

```env
GEMINI_API_KEY=...
```

## CLI Flow

The pipeline in [docu_agent_cli.py](docu_agent_cli.py) follows this path:

1. Load environment variables from `.env`
2. Check that `GEMINI_API_KEY` is available
3. Ask whether to ingest a new document or chat with existing vectors
4. Load the selected `.pdf`, `.txt`, or `.md` file
5. Split the document into overlapping chunks with `RecursiveCharacterTextSplitter`
6. Convert chunks into embeddings with Google Gemini
7. Store the vectors locally in `local_chroma_db/`
8. Start an infinite chat loop with conversational memory

The legacy `ingestion.py`, `retrieval.py`, and `main.py` files are still present, but the new CLI is the recommended entry point for terminal use.

## How To Run

From the `core_engine` folder:

```powershell
pip install -r requirements.txt
python docu_agent_cli.py
```

When the CLI starts:

1. Press `1` to ingest a new document.
2. Drag and drop a file path, or paste the absolute path to a `.pdf`, `.txt`, or `.md` file.
3. After ingestion, the chat loop opens automatically.
4. Press `2` to chat immediately with an existing `local_chroma_db/`.
5. Type `exit` or `quit` to close the session.

If you already have a local Chroma database, you can launch the CLI and choose option `2` directly.

## Expected Output

When everything is ready, the CLI should:

- detect the selected document
- split it into chunks
- create or refresh the local Chroma database
- open an interactive terminal conversation loop
- return grounded answers from local documentation only

If `GEMINI_API_KEY` is missing, the script stops early and asks you to add it to `.env`.

## Notes

- The current environment on this machine uses Python 3.13 x86, so package installation may need a compatible Python build on some systems.
- The CLI is the main supported entry point for this workspace.
- `local_chroma_db/` is the vector store used by the standalone terminal tool.