# api/prompt.py
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

router = APIRouter()

# ---------- Request/Response schemas ----------
class PromptIn(BaseModel):
    question: str

# ---------- Helpers ----------
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def _build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(
            f"[{i}] talk_id={c.get('talk_id','')} | title={c.get('title','')} | score={c.get('score',0.0):.4f}\n"
            f"{c.get('chunk','')}\n"
        )
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{''.join(lines)}\n\n"
        f"Answer ONLY using the retrieved context."
    )

@router.post("/prompt")
def prompt_endpoint(payload: PromptIn):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # ---- Env (set these in Vercel dashboard) ----
    OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
    LLMOD_BASE_URL = _require_env("LLMOD_BASE_URL")

    PINECONE_API_KEY = _require_env("PINECONE_API_KEY")
    INDEX_NAME = _require_env("VECTOR_DB_INDEX_NAME")

    EMBEDDING_MODEL = _require_env("EMBEDDING_MODEL")
    GENERATION_MODEL = _require_env("GENERATION_MODEL")

    TOP_K = int(_require_env("TOP_K"))
    SYSTEM_PROMPT = _require_env("SYSTEM_PROMPT")

    # ---- Clients ----
    emb = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=LLMOD_BASE_URL,   # key point for llmod.ai
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # ---- Embed query + retrieve ----
    qvec = emb.embed_query(question)

    res = index.query(
        vector=qvec,
        top_k=TOP_K,
        include_metadata=True,
    )

    matches = (res.get("matches") or [])
    context_chunks: List[Dict[str, Any]] = []
    for m in matches:
        md = (m.get("metadata") or {})
        context_chunks.append({
            "talk_id": md.get("talk_id") or "",
            "title": md.get("title", "") or "",
            "chunk": md.get("chunk_text", "") or "",
            "score": float(m.get("score") or 0.0),
        })

    # ---- Augmented prompt ----
    user_prompt = _build_user_prompt(question, context_chunks)

    # ---- Generate answer (LangChain interface) ----
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=LLMOD_BASE_URL
    )

    messages = []
    if SYSTEM_PROMPT:
        messages.append(("system", SYSTEM_PROMPT))
    messages.append(("user", user_prompt))

    answer = llm.invoke(messages).content

    return {
        "response": answer,
        "context": context_chunks,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
