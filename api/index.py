import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings

# -------------------------
# FastAPI app (Vercel looks for `app`)
# -------------------------
app = FastAPI()

# CORS (optional but convenient)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# Models
# -------------------------
class PromptRequest(BaseModel):
    question: str

# -------------------------
# Helpers
# -------------------------
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(
            f"[{i}] talk_id={c.get('talk_id','')} | title={c.get('title','')} | score={c.get('score',0):.4f}\n"
            f"{c.get('chunk','')}\n"
        )
    context_block = "\n".join(blocks).strip()

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Answer using ONLY the retrieved context. "
        "If the context is insufficient, say you don't know based on the provided TED data."
    )

def get_clients():
    # --- env ---
    llmod_api_key = require_env("OPENAI_API_KEY")
    llmod_base_url = require_env("LLMOD_BASE_URL")

    pinecone_api_key = require_env("PINECONE_API_KEY")
    index_name = require_env("VECTOR_DB_INDEX_NAME")

    embedding_model = os.getenv("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")
    generation_model = os.getenv("GENERATION_MODEL", "RPRTHPB-gpt-5-mini")

    top_k = int(os.getenv("TOP_K", "10"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "2048"))
    overlap_ratio = float(os.getenv("OVERLAP_RATIO", "0.1"))

    system_prompt = os.getenv("SYSTEM_PROMPT", "")

    # --- clients ---
    # LangChain embeddings wrapper (OpenAI-compatible endpoint)
    emb = OpenAIEmbeddings(
        model=embedding_model,
        api_key=llmod_api_key,
        base_url=llmod_base_url,
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # OpenAI SDK client for chat (also OpenAI-compatible endpoint)
    llm_client = OpenAI(api_key=llmod_api_key, base_url=llmod_base_url)

    return {
        "emb": emb,
        "index": index,
        "llm_client": llm_client,
        "embedding_model": embedding_model,
        "generation_model": generation_model,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "system_prompt": system_prompt,
    }

# -------------------------
# Routes
# -------------------------

@app.get("/")
def health():
    return {"ok": True, "service": "ted-rag"}

@app.get("/api/stats")
def stats():
    # strict JSON field names per your spec
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "2048")),
        "overlap_ratio": float(os.getenv("OVERLAP_RATIO", "0.1")),
        "top_k": int(os.getenv("TOP_K", "10")),
    }

@app.post("/api/prompt")
def prompt(req: PromptRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'.")

    try:
        ctx = get_clients()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    emb = ctx["emb"]
    index = ctx["index"]
    llm_client = ctx["llm_client"]

    top_k = ctx["top_k"]
    system_prompt = ctx["system_prompt"]
    model = ctx["generation_model"]

    # 1) embed query
    qvec = emb.embed_query(question)

    # 2) retrieve
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    matches = (res.get("matches") or [])

    contexts: List[Dict[str, Any]] = []
    for m in matches:
        md = m.get("metadata") or {}
        chunk_text = md.get("chunk_text") or md.get("chunk") or ""  # your stored field
        contexts.append(
            {
                "talk_id": md.get("talk_id") or md.get("row_id") or m.get("id"),
                "title": md.get("title") or "",
                "chunk": chunk_text,
                "score": float(m.get("score") or 0.0),
            }
        )

    # 3) build augmented prompt
    user_prompt = build_user_prompt(question, contexts)

    # 4) generate
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    chat = llm_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    answer = (chat.choices[0].message.content or "").strip()

    return {
        "response": answer,
        "context": contexts,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt,
        },
    }
