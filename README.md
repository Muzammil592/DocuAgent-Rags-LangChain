# DocuAgent-Rags and LangChain Day 1

Day 1 ka kaam yeh tha ke project ko bilkul clean base par khara kiya jaye, jahan documentation files ko ingest karke local vector knowledge base banaya ja sake. Is phase mein focus simple rakha gaya: folder structure, dependency contract, secure environment variable setup, aur ek runnable ingestion pipeline.

## What This Day 1 Setup Does

Yeh starter setup local files ko read karta hai, unko meaningful chunks mein split karta hai, aur phir Google Gemini embeddings ke through Chroma vector database mein save karta hai. Simple alfaaz mein: aapki docs ko searchable knowledge base mein convert karta hai.

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

The Day 1 stack is based on these packages:

- FastAPI and Uvicorn for future API work
- LangChain for document loading, splitting, and vector orchestration
- Chroma for local vector storage
- Google GenAI embeddings for semantic indexing
- python-dotenv for environment variable loading
- pypdf for PDF reading support

## Environment Setup

1. Place your Google AI Studio key inside [.env](.env) using the `GEMINI_API_KEY` variable.
2. Keep your actual key private. Do not commit `.env` to source control.
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
- The code itself is structured and compiles cleanly.
- The repo is ready for the next phase, where we can add an API layer or a query interface on top of the vector store.