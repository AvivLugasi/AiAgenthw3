# api/stats.py
import os
from fastapi import APIRouter

router = APIRouter()

@router.get("/stats")
def stats_endpoint():
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE")),
        "overlap_ratio": float(os.getenv("OVERLAP_RATIO")),
        "top_k": int(os.getenv("TOP_K")),
    }
