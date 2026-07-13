from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "local_chroma_db"

load_dotenv(BASE_DIR / ".env")


def verify_environment() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print()
        print("CRITICAL ERROR: GEMINI_API_KEY is missing in your .env file.")
        print("Add GEMINI_API_KEY=your_key_here inside core_engine/.env and try again.")
        sys.exit(1)


def _load_document(file_path: Path):
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix.lower() in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError("Unsupported format. Use a .pdf, .txt, or .md file.")

    return loader.load()


def ingest_source_document(file_path: str) -> bool:
    """Load, chunk, vectorize, and persist a single document into local Chroma."""
    source_path = Path(file_path.strip().strip('"\''))

    if not source_path.exists():
        print(f"\nFile not found: {source_path}")
        return False

    print(f"\nProcessing: {source_path.name}")

    try:
        documents = _load_document(source_path)
    except Exception as exc:
        print(f"Unable to load document: {exc}")
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    DB_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating embeddings and persisting local vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    print("Knowledge base ready.")
    return True


def _db_is_ready() -> bool:
    return DB_DIR.exists() and any(DB_DIR.iterdir())


def build_rag_chain():
    if not _db_is_ready():
        raise FileNotFoundError(
            f"No local vector store found at [{DB_DIR}]. Run ingestion first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer it, just rephrase it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    system_prompt = (
        "You are DocuAgent, an expert AI specialized for this product documentation.\n"
        "Answer the questions using ONLY the provided documentation context below.\n\n"
        "STRICT BOUNDARY:\n"
        "If the context does not contain the answer, reply strictly with: 'I cannot find this information in the provided documentation.' Do not hallucinate.\n\n"
        "CONTEXT:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def start_interactive_chat() -> None:
    print("\nLoading local knowledge vectors...")

    rag_chain = build_rag_chain()
    chat_history = ChatMessageHistory()
    session_id = uuid.uuid4().hex

    print()
    print("==========================================================")
    print("DOCUAGENT CLI IS LIVE. Type 'exit' or 'quit' to stop.")
    print("==========================================================")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            print("Thinking...", end="\r")
            response = rag_chain.invoke(
                {"input": user_input, "chat_history": chat_history.messages},
                config={"configurable": {"session_id": session_id}},
            )

            answer = response["answer"]
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(answer)

            print(" " * 40, end="\r")
            print(f"Agent: {answer}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye.")
            break
        except Exception as exc:
            print(" " * 40, end="\r")
            print(f"Error during chat execution: {exc}")


def main() -> None:
    verify_environment()

    print("======================================================")
    print("DOCUAGENT TERMINAL INTERACTION INTERFACE")
    print("======================================================")
    print("1. Ingest a new document (PDF, TXT, MD)")
    print("2. Chat with the existing local knowledge base")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        file_path = input("\nDrag and drop a file path here: ").strip()
        if ingest_source_document(file_path):
            start_interactive_chat()
        return

    if choice == "2":
        if not _db_is_ready():
            print()
            print(f"No local vector store found at [{DB_DIR}]. Run option 1 first.")
            return

        start_interactive_chat()
        return

    print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()