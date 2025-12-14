# api/stats.py
import os
from fastapi import APIRouter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/stats")
def stats_endpoint():
    logger.info(f"check logging {os.getenv("CHUNK_SIZE", "2048")}")
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "2048")),
        "overlap_ratio": float(os.getenv("OVERLAP_RATIO", "0.1")),
        "top_k": int(os.getenv("TOP_K", "10")),
    }
