import logging
from pathlib import Path
from threading import Lock

from config.settings import settings

logger = logging.getLogger(__name__)


class DatalabOCRClient:
    """Cloud MVP OCR tier backed by Datalab's hosted Surya OCR API.

    Intended to be constructed once and reused (see OCRRouterService); `_load`
    is safe to call from concurrent threads (`asyncio.to_thread` workers).
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.datalab_api_key
        if not self.api_key:
            raise ValueError("DATALAB_API_KEY is not set")
        self._client = None
        self._load_lock = Lock()

    def _load(self) -> None:
        """Lazily construct the Datalab SDK client on first use."""
        if self._client is not None:
            return
        with self._load_lock:
            if self._client is not None:
                return

            logger.info("Initializing Datalab client (api_key=%s***)", self.api_key[:4])
            from datalab_sdk import DatalabClient

            self._client = DatalabClient(api_key=self.api_key)

    def extract_text(self, file_path: str | Path) -> str:
        """Run Datalab's hosted OCR on `file_path` and return the recognized text."""
        self._load()

        result = self._client.ocr(str(file_path))
        if not result.success:
            logger.error("Datalab OCR failed for %s: %s", file_path, result.error)
            raise RuntimeError(f"Datalab OCR failed: {result.error}")
        return result.get_text()
