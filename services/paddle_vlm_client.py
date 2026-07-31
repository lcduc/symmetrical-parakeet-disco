import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class PaddleVLMClient:
    """GPU-accelerated OCR/document-understanding tier using PaddleOCR-VL (pipeline v1.6).

    Intended to be constructed once and reused (see OCRRouterService); `_load`
    is safe to call from concurrent threads (`asyncio.to_thread` workers).
    """

    PIPELINE_VERSION = "v1.6"

    def __init__(self):
        self._pipeline = None
        self._load_lock = Lock()

    def _load(self) -> None:
        """Lazily construct the PaddleOCR-VL pipeline on first use."""
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return

            logger.info("Loading PaddleOCR-VL pipeline (version=%s)", self.PIPELINE_VERSION)
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(pipeline_version=self.PIPELINE_VERSION)

    def extract_text(self, file_path: str | Path) -> str:
        """Extract all recognized text blocks from `file_path`, in document order."""
        self._load()

        results = self._pipeline.predict(str(file_path))
        texts = []
        for result in results:
            # Read every recognized block directly rather than result.markdown:
            # the markdown builder silently drops blocks labeled header/footer/etc,
            # which discards real text on non-standard-document layouts.
            for block in result["parsing_res_list"]:
                if block.content:
                    texts.append(block.content)

        if not texts:
            logger.warning("PaddleOCR-VL returned no text blocks for %s", file_path)
        return "\n".join(texts).strip()
