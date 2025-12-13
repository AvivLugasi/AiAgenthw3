# api/stats.py
import os
from starlette.responses import JSONResponse, Response

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

async def handler(request):
    # CORS preflight
    if request.method == "OPTIONS":
        return Response(status_code=200, headers=CORS_HEADERS)

    if request.method != "GET":
        return JSONResponse(
            {"error": "Method not allowed. Use GET."},
            status_code=405,
            headers=CORS_HEADERS,
        )

    chunk_size = int(os.environ.get("CHUNK_SIZE", "2048"))
    overlap_ratio = float(os.environ.get("OVERLAP_RATIO", "0.1"))
    top_k = int(os.environ.get("TOP_K", "10"))

    return JSONResponse(
        {
            "chunk_size": chunk_size,
            "overlap_ratio": overlap_ratio,
            "top_k": top_k,
        },
        status_code=200,
        headers=CORS_HEADERS,
    )