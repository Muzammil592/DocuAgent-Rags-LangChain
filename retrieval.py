from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


def format_docs(docs: Iterable) -> str:
    """Aggregate retrieved documents into a single context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def _normalize_query_lines(raw_text: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    for line in raw_text.splitlines():
        candidate = re.sub(r"^[\s\-\*\d\.)]+", "", line).strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(candidate)

    return queries


def _fallback_query_variants(question: str) -> list[str]:
    return [
        f"Steps to configure {question}",
        f"Documentation for {question}",
        f"Technical setup parameters for {question}",
    ]


def initialize_qa_chain():
    print("\nStarting Day 2 retrieval engine...")

    try:
        from dotenv import load_dotenv
        from langchain_community.vectorstores import Chroma
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
    except ImportError as import_err:
        raise EnvironmentError(
            "Required runtime packages are missing. Install the project dependencies before running retrieval.py."
        ) from import_err

    load_dotenv(BASE_DIR / ".env")

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Database directory missing at [{DB_DIR}]. Run ingestion.py first."
        )

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to core_engine/.env before running retrieval.py."
        )

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_storage = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings_model,
    )
    base_retriever = vector_storage.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate exactly 3 semantically different technical search queries for the same user question. "
                "Return one query per line only, without numbering, bullets, or commentary.",
            ),
            ("human", "{question}"),
        ]
    )
    query_expander = query_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_template(
        """
You are an elite software architecture copilot named DocuAgent.
Use ONLY the retrieved technical context below.

STRICT OPERATIONAL BOUNDARIES:
1. If the provided context does not contain the answer, state exactly: "Error: Requested technical mapping absent within local knowledge base vectors."
2. Do not use external knowledge.
3. Keep the answer concise, technical, and structured.

Retrieved context:
{context}

User question: {question}

DocuAgent response:
""".strip()
    )

    def _retrieve_docs(question: str):
        try:
            raw_variants = query_expander.invoke({"question": question})
            variants = _normalize_query_lines(raw_variants)
        except Exception:
            variants = []

        if len(variants) < 3:
            variants.extend(_fallback_query_variants(question))

        ordered_queries = [question, *variants[:3]]

        def _search(query: str):
            return base_retriever.invoke(query)

        deduped_docs = []
        seen_keys = set()

        with ThreadPoolExecutor(max_workers=len(ordered_queries)) as executor:
            for docs in executor.map(_search, ordered_queries):
                for doc in docs:
                    metadata = getattr(doc, "metadata", {}) or {}
                    key = (
                        metadata.get("source"),
                        metadata.get("chunk_id"),
                        doc.page_content,
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped_docs.append(doc)

        return deduped_docs

    docu_agent_chain = (
        {
            "context": RunnableLambda(lambda question: format_docs(_retrieve_docs(question))),
            "question": RunnablePassthrough(),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    print("Day 2 retrieval chain assembled successfully.")
    return docu_agent_chain


if __name__ == "__main__":
    try:
        chain = initialize_qa_chain()
        print("\nRunning pipeline test trace...")
        test_query = "What is the core setup instruction?"
        response = chain.invoke(test_query)
        print(f"\nAgent output:\n{response}")
    except Exception as err:
        print(f"\nPipeline test failed: {err}")