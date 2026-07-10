from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="DocuAgent Core AI Engine",
    description="Production-grade isolated RAG microservice for automated documentation synthesis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not DB_DIR.exists():
    raise FileNotFoundError(
        f"Core database missing at [{DB_DIR}]. Run ingestion.py first."
    )

if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError(
        "GEMINI_API_KEY is missing. Add it to core_engine/.env before starting the API."
    )

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_storage = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings_model)
base_retriever = vector_storage.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference prior context,
formulate a standalone question that can be understood without the chat history.
Do NOT answer the question. Return only the reformulated question when needed.
""".strip()

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    base_retriever,
    contextualize_q_prompt,
)

system_prompt = """
You are an elite software architecture copilot named DocuAgent.
Answer the user's technical question using ONLY the provided documentation context blocks.

STRICT OPERATIONAL BOUNDARIES:
1. If the provided context does not explicitly contain the answer, reply with:
   "Error: Requested technical mapping absent within local knowledge base vectors."
2. Do not invent facts or use external knowledge.
3. Keep outputs developer-centric, crisp, and structured.

RETRIEVED DOCUMENTATION CONTEXT:
{context}
""".strip()

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

state_store: dict[str, ChatMessageHistory] = {}
state_lock = Lock()


def get_session_history(session_id: str) -> ChatMessageHistory:
    with state_lock:
        if session_id not in state_store:
            state_store[session_id] = ChatMessageHistory()
        return state_store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


class QueryRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "DocuAgent RAG Core Vector Mesh",
        "database": str(DB_DIR),
    }


@app.post("/api/v1/query")
async def execute_query(payload: QueryRequest):
    try:
        response = conversational_rag_chain.invoke(
            {"input": payload.query},
            config={"configurable": {"session_id": payload.session_id}},
        )
        return {
            "status": "success",
            "session_id": payload.session_id,
            "answer": response["answer"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Core Engine Execution Exception: {exc}",
        ) from exc


@app.get("/")
async def root():
    return {"status": "ok", "service": "DocuAgent Core AI Engine"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)