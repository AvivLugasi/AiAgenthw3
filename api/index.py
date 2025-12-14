# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .prompt import router as prompt_router
from .stats import router as stats_router

app = FastAPI()

# CORS (optional but useful for browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your required endpoints
app.include_router(prompt_router, prefix="/api")
app.include_router(stats_router, prefix="/api")