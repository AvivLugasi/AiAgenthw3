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
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }

def _parse_body(event):
    try:
        return json.loads(event.get("body") or "{}")
    except Exception:
        return {}

def _build_user_prompt(question: str, contexts: list[dict]) -> str:
    ctx = []
    for i, c in enumerate(contexts, 1):
        ctx.append(
            f"[{i}] {c['title']} (score={c['score']:.4f})\n{c['chunk']}\n"
        )
    return f"Question:\n{question}\n\nContext:\n{''.join(ctx)}"

# -------------------------
# Main handler
# -------------------------

def handler(event, context):
    if event.get("httpMethod") != "POST":
        return _json_response(405, {"error": "POST only"})

    body = _parse_body(event)
    question = body.get("question", "").strip()
    if not question:
        return _json_response(400, {"error": "Missing question"})

    # ---- ENV ----
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    INDEX_NAME = os.environ["VECTOR_DB_INDEX_NAME"]
    TOP_K = int(os.environ.get("TOP_K", "10"))

    # ---- Clients ----
    emb = OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=OPENAI_API_KEY,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # ---- Embed query ----
    qvec = emb.embed_query(question)

    # ---- Retrieve ----
    res = index.query(
        vector=qvec,
        top_k=TOP_K,
        include_metadata=True,
    )

    contexts = []
    for m in res["matches"]:
        md = m["metadata"]
        contexts.append({
            "talk_id": md.get("talk_id"),
            "title": md.get("title"),
            "chunk": md.get("chunk_text", ""),
            "score": m["score"],
        })

    # ---- Prompt ----
    user_prompt = _build_user_prompt(question, contexts)

    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=os.environ["GENERATION_MODEL"],
        messages=[
            {"role": "system", "content": os.environ["SYSTEM_PROMPT"]},
            {"role": "user", "content": user_prompt},
        ],
    )

    return _json_response(200, {
        "response": completion.choices[0].message.content,
        "context": contexts,
        "Augmented_prompt": {
            "System": os.environ["SYSTEM_PROMPT"],
            "User": user_prompt,
        }
    })