# api/stats.py
import os
from fastapi import APIRouter

router = APIRouter()

@router.get("/stats")
def stats_endpoint():
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "2048")),
        "overlap_ratio": float(os.getenv("OVERLAP_RATIO", "0.1")),
        "top_k": int(os.getenv("TOP_K", "10")),
    }
