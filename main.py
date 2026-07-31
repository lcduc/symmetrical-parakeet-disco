from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from api.candidates import router as candidates_router
from api.interview import router as interview_router
from api.jobs import router as jobs_router
from api.matches import router as matches_router
from config.settings import settings
from services.ocr_router_service import OCRRouterService
from services.rate_limiter_service import InMemoryRateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and construct shared singletons once at startup."""
    app.state.embedding_model = SentenceTransformer(settings.embedding_model_name)
    app.state.ocr_router_service = OCRRouterService()
    app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
    yield


app = FastAPI(title="HR CV Matcher", lifespan=lifespan)

_cors_origins = [origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()]
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(candidates_router)
app.include_router(interview_router)
app.include_router(jobs_router)
app.include_router(matches_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness endpoint for the Docker healthcheck; intentionally unauthenticated."""
    return {"status": "ok"}
