import logging
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_dsn: str
    llm_provider: str = "openai"
    llm_api_key: str
    llm_model: str | None = None  # overrides the provider's default model (e.g. "gpt-4.1", "claude-opus-4-6")
    # Overrides llm_model for resume extraction specifically (same llm_provider) — the one
    # LLM-driven task most worth pinning to a stronger model. Falls back to llm_model, then
    # the provider's own default, when unset.
    extraction_llm_model: str | None = None
    embedding_model_name: str
    vlm_device: str
    max_extraction_retries: int = 3
    hard_filter_soft_cap: float = 0.4

    ocr_vlm_backend: str = "paddle_vlm"  # one of: paddle_vlm, datalab, qwen2_vl
    datalab_api_key: str | None = None
    # auto = native text layer -> local fast OCR -> escalate to ocr_vlm_backend on low quality (default).
    # cloud_only = skip native text/local OCR entirely; always route straight through ocr_vlm_backend
    # (pair with OCR_VLM_BACKEND=datalab for a fully cloud-based, no-local-model deployment).
    ocr_mode: Literal["auto", "cloud_only"] = "auto"
    ocr_quality_threshold: float = 0.85  # min fast-OCR quality_score (0-1) before escalating to the VLM tier

    company_name: str = ""  # watermarked onto rendered resume templates when set

    api_key: str = ""  # shared secret gating every API route except /health; empty disables auth
    max_upload_size_mb: int = 20
    rate_limit_per_minute: int = 20

    log_level: str = "INFO"


def configure_logging(level_name: str = "INFO") -> None:
    """Configure the root logger once for the whole application.

    Every module should log via `logging.getLogger(__name__)` rather than
    configuring its own handlers; this is the single place that sets format
    and level. Falls back to INFO (with a warning) for an unrecognized level name.
    """
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        level = logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        logging.getLogger(__name__).warning("Unknown LOG_LEVEL %r; defaulting to INFO", level_name)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


settings = Settings()
configure_logging(settings.log_level)
