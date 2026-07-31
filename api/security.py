import logging
import secrets

from fastapi import Header, HTTPException, Request, status

from config.settings import settings

logger = logging.getLogger(__name__)


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency gating a route behind the shared `settings.api_key`.

    A no-op when `settings.api_key` is unset, so local/demo use stays
    frictionless; set `API_KEY` before exposing the app beyond localhost.
    """
    if not settings.api_key:
        return
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


async def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency limiting a route to `settings.rate_limit_per_minute` calls per client IP.

    Applied only to endpoints that fan out to paid LLM calls
    (`/candidates/upload`, `/matches/run`) rather than every route.
    """
    limiter = request.app.state.rate_limiter
    client_key = request.client.host if request.client else "unknown"
    if not limiter.allow(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded; please slow down.",
        )
