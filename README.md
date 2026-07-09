# DocuAgent RAGs and LangChain: Day 1 and Day 2

Day 1 was all about getting the project onto a clean base so we could turn local documentation into a searchable vector knowledge base. Day 2 builds on that foundation with a stricter retrieval engine that expands each question into multiple semantic variants before searching the local Chroma store.

## What This Day 1 Setup Does

This starter setup reads local files, splits them into meaningful chunks, and stores the results in a Chroma vector database using Google Gemini embeddings. In plain English, it converts your documents into something the app can search semantically.

## What Day 2 Adds

Day 2 introduces [retrieval.py](retrieval.py), which changes the query flow from single-shot retrieval to a multi-query pipeline:

1. Expand the user question into 3 different technical search variants.
2. Search the local Chroma database for each variant in parallel.
3. De-duplicate retrieved chunks before building the final context.
4. Feed the context into a strict prompt that only allows grounded answers from local documentation.

If the local context does not contain the answer, the chain returns:

```text
Error: Requested technical mapping absent within local knowledge base vectors.
```

## Project Structure

```text
core_engine/
├── .env
├── .gitignore
├── README.md
├── ingestion.py
├── retrieval.py
├── requirements.txt
└── data_source/
    └── sample.txt
```

## Tools And Dependencies

The Day 1 and Day 2 stack uses these packages:

- FastAPI and Uvicorn for future API work
- LangChain for document loading, splitting, and vector orchestration
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

## Ingestion Flow

The pipeline in [ingestion.py](ingestion.py) follows this path:

1. Load environment variables from `.env`
2. Check that `GEMINI_API_KEY` is available
3. Read `.md`, `.txt`, and `.pdf` files from `data_source/`
4. Split documents into overlapping chunks with `RecursiveCharacterTextSplitter`
5. Convert chunks into embeddings with Google Gemini
6. Store the vectors locally in `chroma_db/`

## Retrieval Flow

The pipeline in [retrieval.py](retrieval.py) follows this path:

1. Load environment variables from `.env`
2. Confirm that `GEMINI_API_KEY` is available
3. Open the local Chroma database at `chroma_db/`
4. Generate 3 semantic query variants for the user question
5. Search Chroma in parallel for each query variant
6. De-duplicate the retrieved chunks
7. Build a strict prompt with the retrieved context only
8. Send the prompt to `gemini-2.5-flash`

## How To Run

From the `core_engine` folder:

```powershell
pip install -r requirements.txt
python ingestion.py
python retrieval.py
```

If the folder is empty, add a file like `data_source/sample.txt` first and run the script again.

For Day 2, make sure `chroma_db/` already exists before running `python retrieval.py`.

## Expected Output

When everything is ready, the script should:

- detect your documents
- split them into chunks
- create or refresh the local Chroma database
- print a success message for the knowledge base
- expand a question into multiple search variants
- retrieve relevant chunks from the local database
- return a grounded answer or the explicit missing-context error

If `GEMINI_API_KEY` is missing, the script will stop early and ask you to add it.

## Notes For Day 1

- The current environment on this machine uses Python 3.13 x86, so package installation may need a compatible Python build on some systems.
- The code is structured and compiles cleanly.
- The repo is ready for the next phase, where we can add an API layer or a query interface on top of the vector store.

## Notes For Day 2

- `retrieval.py` is designed to fail fast if `chroma_db/` is missing.
- The retrieval engine is intentionally strict: it only answers from local context.
- On this machine, the pinned LangChain stack may still require a compatible Python environment before it can run end to end.