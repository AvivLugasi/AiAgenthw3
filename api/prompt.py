# api/prompt.py
import json

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from openai import OpenAI
from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

# APIs
os.environ["OPENAI_API_KEY"] = "sk-05hY9VmXtb8jonQshatE0Q"
os.environ["Pinecone_API_KEY"] = "pcsk_4kvKQd_36TWp5QQpxjoF7SasYzsE5ja3gKBG5pjRcsAihBz3g9RFuzNJgtbxrkDGaNaLvV"
os.environ["LLMOD_BASE_URL"] = "https://api.llmod.ai/v1"

# models
os.environ["EMBEDDING_MODEL"] = "RPRTHPB-text-embedding-3-small"
os.environ["GENERATION_MODEL"] = "RPRTHPB-gpt-5-mini"

# embedding parameters
os.environ["TEXT_EMBEDDING_DIM"] = "1536"
os.environ["VECTOR_DB_INDEX_NAME"] = "ted-talks-embeddings"
os.environ["VECTOR_DB_SIMILARITY_METRIC"] = "cosine"
os.environ["VECTOR_DB_CLOUD"] = "aws"
os.environ["VECTOR_DB_REGION"] = "us-east-1"
os.environ["EMBEDDING_BATCH_SIZE"] = "100"

# RAG parameters
os.environ["OVERLAP_RATIO"] = "0.1" # range: 0-0.3
os.environ["TOP_K"]  = "10" # range: 1-30
os.environ["CHUNK_SIZE"] = "2048" # limit 2048

# DataSet parameters
META_COLS = [
    "talk_id",
    "title",
    "speaker_1",
    "all_speakers",
    "occupations",
    "about_speakers",
    "views",
    "recorded_date",
    "published_date",
    "event",
    "native_lang",
    "available_lang",
    "comments",
    "duration",
    "topics",
    "related_talks",
    "url",
    "description",
]

# Generation model parameters
os.environ["SYSTEM_PROMPT"] = "You are a TED Talk assistant that answers questions strictly and \
                               only based on the TED dataset context provided to you metadata \
                               and transcript passages. You must not use any external \
                               knowledge, the open internet, or information that is not explicitly \
                               contained in the retrieved context. If the answer cannot be \
                               determined from the provided context, respond: “I don’t know \
                               based on the provided TED data.” Always explain your answer \
                               using the given context, quoting or paraphrasing the relevant \
                               transcript or metadata when helpful."

# -------------------------
# Helpers
# -------------------------

def _json_response(status: int, payload: dict):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }

def _parse_body(event) -> dict:
    body = event.get("body")
    if not body:
        return {}
    if isinstance(body, dict):
        return body
    try:
        return json.loads(body)
    except Exception:
        return {}

def _build_user_prompt(question: str, contexts: list[dict]) -> str:
    # Simple RAG prompt formatting (you can change this freely)
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(
            f"[{i}] talk_id={c.get('talk_id')} | title={c.get('title')} | score={c.get('score')}\n"
            f"{c.get('chunk', '')}\n"
        )
    context_block = "\n".join(ctx_lines).strip()

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Answer using the context above. If the context is insufficient, say so."
    )

# -------------------------
# Main Vercel handler
# -------------------------

def handler(event, context):
    # CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return _json_response(200, {"ok": True})

    if event.get("httpMethod") != "POST":
        return _json_response(405, {"error": "Method not allowed. Use POST."})

    body = _parse_body(event)
    question = (body.get("question") or "").strip()
    if not question:
        return _json_response(400, {"error": "Missing 'question' in JSON body."})

    # ---- Config from env ----
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_INDEX = os.environ["VECTOR_DB_INDEX_NAME"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")
    CHAT_MODEL = os.environ.get("GENERATION_MODEL", "RPRTHPB-gpt-5-mini")
    TOP_K = int(os.environ.get("TOP_K", "10"))
    SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "")

    # ---- Clients ----
    emb_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # ---- Embed query ----
    query_vec = emb_model.embed_query(question)

    # ---- Retrieve from Pinecone ----
    res = index.query(
        vector=query_vec,
        top_k=TOP_K,
        include_metadata=True,
    )

    matches = res.get("matches", []) or []
    context_chunks = []
    for m in matches:
        md = (m.get("metadata") or {})
        chunk_text = md.get("chunk_text") or md.get("chunk") or ""
        context_chunks.append({
            "talk_id": md.get("talk_id") or md.get("row_id") or m.get("id"),
            "title": md.get("title"),
            "chunk": chunk_text,
            "score": float(m.get("score") or 0.0),
        })

    # ---- Build augmented prompt ----
    user_prompt = _build_user_prompt(question, context_chunks)

    # ---- Call GPT-5-mini ----
    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_prompt})

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    answer = chat.choices[0].message.content or ""

    return _json_response(200, {
        "response": answer,
        "context": context_chunks,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        }
    })