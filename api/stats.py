# api/stats.py
import os
import json

def _json_response(status: int, payload: dict):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }

def handler(event, context):
    if event.get("httpMethod") != "GET":
        return _json_response(405, {"error": "Method not allowed"})

    return _json_response(200, {
        "chunk_size": int(os.environ.get("CHUNK_SIZE", "2048")),
        "overlap_ratio": float(os.environ.get("OVERLAP_RATIO", "0.1")),
        "top_k": int(os.environ.get("TOP_K", "10")),
    })
