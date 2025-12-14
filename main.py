from fastapi import FastAPI
from api.prompt import router as prompt_router
from api.stats import router as stats_router

app = FastAPI(title="TED Talks RAG API")

app.include_router(prompt_router, prefix="/api")
app.include_router(stats_router, prefix="/api")