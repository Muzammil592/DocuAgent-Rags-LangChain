from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_source"
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")


def _load_documents() -> list:
    loaders = [
        (
            "Markdown Files",
            DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                use_multithreading=True,
            ),
        ),
        (
            "Plain Text Docs",
            DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                use_multithreading=True,
            ),
        ),
        (
            "Enterprise PDFs",
            DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                use_multithreading=True,
            ),
        ),
    ]

    documents = []
    for description, loader in loaders:
        try:
            extracted_docs = loader.load()
            if extracted_docs:
                documents.extend(extracted_docs)
                print(f"   ✔ [{description}] mapped successfully. Node Count: {len(extracted_docs)}")
        except Exception as err:
            print(f"   ⚠️  Skipping context vector processing block for [{description}]. Cause: {err}")

    return documents


def run_ingestion_pipeline() -> bool:
    """
    Executes end-to-end vector space synthesis for local documentation.
    """
    print("\n==========================================================================")
    print("🛰️  ENGINE ACTIVATION: Executing Documentation Vector Ingress Pipeline...")
    print("==========================================================================\n")

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ CRITICAL REJECTION: Core execution environment variable [GEMINI_API_KEY] is absent.")
        print(f"💡 Action Required: Please append your authorization key within [{BASE_DIR / '.env'}]")
        return False

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📂 Pipeline Alert: Generated empty data matrix folder at [{DATA_DIR}].")
        print("👉 Action Required: Place your target text, markdown, or PDF developer documentations inside it.")
        return False

    print("📋 Structural Ingestion Matrix initialized. Parsing disk assets...")
    documents = _load_documents()

    if not documents:
        print("\n❌ PIPELINE SHUTDOWN: Target tracking canvas directory has zero extractable payloads.")
        print("👉 Please populate data_source folder with real data before execution.")
        return False

    print(f"\n📊 Extraction phase finalized. Aggregated independent text nodes: {len(documents)}")
    print("\n✂️  Executing multi-layered string partitioning arrays...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
    )
    tokenized_chunks = text_splitter.split_documents(documents)
    print(f"   ✔ Fragmentation sequence complete. Generated total matrix segments: {len(tokenized_chunks)}")

    if DB_DIR.exists():
        print(f"🧹 Clearing stale database tracking structures localized at: [{DB_DIR}]")
        shutil.rmtree(DB_DIR)

    print("\n🧠 Connecting to Google AI Studio Embedding Clusters via LangChain SDK...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        print(f"💾 Persisting multidimensional structural indices onto disk at: [{DB_DIR}]...")
        vector_storage = Chroma.from_documents(
            documents=tokenized_chunks,
            embedding=embeddings_model,
            persist_directory=str(DB_DIR),
        )
        if hasattr(vector_storage, "persist"):
            vector_storage.persist()

        print("\n🔥 SUCCESS MATRIX ACHIEVED: Complete isolated RAG Vector Knowledge Base is live!")
        return True
    except Exception as network_err:
        print(f"\n❌ PIPELINE EXCEPTION: Vector hydration sequence collapsed. Details: {network_err}")
        return False


if __name__ == "__main__":
    run_ingestion_pipeline()
