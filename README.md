# DocuAgent RAGs and LangChain: Day 1

Day 1 was all about getting the project onto a clean base so we could turn local documentation into a searchable vector knowledge base. The focus stayed simple: set up the folder structure, define the dependency contract, wire the environment variables, and keep the ingestion pipeline runnable.

## What This Day 1 Setup Does

This starter setup reads local files, splits them into meaningful chunks, and stores the results in a Chroma vector database using Google Gemini embeddings. In plain English, it converts your documents into something the app can search semantically.

## Project Structure

```text
core_engine/
├── .env
├── .gitignore
├── README.md
├── ingestion.py
├── requirements.txt
└── data_source/
    └── sample.txt
```

## Tools And Dependencies

The Day 1 stack uses these packages:

- FastAPI and Uvicorn for future API work
- LangChain for document loading, splitting, and vector orchestration
- Chroma for local vector storage
- Google GenAI embeddings for semantic indexing
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

## How To Run

From the `core_engine` folder:

```powershell
pip install -r requirements.txt
python ingestion.py
```

If the folder is empty, add a file like `data_source/sample.txt` first and run the script again.

## Expected Output

When everything is ready, the script should:

- detect your documents
- split them into chunks
- create or refresh the local Chroma database
- print a success message for the knowledge base

If `GEMINI_API_KEY` is missing, the script will stop early and ask you to add it.

## Notes For Day 1

- The current environment on this machine uses Python 3.13 x86, so package installation may need a compatible Python build on some systems.
- The code is structured and compiles cleanly.
- The repo is ready for the next phase, where we can add an API layer or a query interface on top of the vector store.