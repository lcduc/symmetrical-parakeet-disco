import logging
from pathlib import Path

import fitz

from config.settings import settings
from services.ocr_router_service import OCRRouterService

logger = logging.getLogger(__name__)


class IngestionService:
    """Extracts raw text from a candidate/job file, preferring native PDF text over OCR.

    `settings.ocr_mode` selects the flow: "auto" (default) prefers the native
    PDF text layer, then local fast OCR, escalating to the configured VLM
    tier only on low quality; "cloud_only" skips straight to the VLM tier
    (pair with OCR_VLM_BACKEND=datalab for a fully cloud-based pipeline).
    """

    NATIVE_TEXT_MIN_CHARS = 200

    def __init__(self, ocr_router: OCRRouterService | None = None):
        self._ocr_router = ocr_router or OCRRouterService()

    def extract_text(self, file_path: str | Path) -> str:
        """Return the best-effort raw text for `file_path`.

        PDFs with a usable native text layer skip OCR entirely; everything
        else (and PDFs without enough native text) goes through the OCR
        pipeline. Bypassed entirely when `settings.ocr_mode` is "cloud_only".
        """
        logger.info("Extracting text from %s (ocr_mode=%s)", file_path, settings.ocr_mode)
        path = Path(file_path)
        if settings.ocr_mode == "cloud_only":
            return self._ocr_router.run_vlm_ocr(path).text
        if path.suffix.lower() == ".pdf":
            native_text = self._try_native_text_layer(path)
            if native_text is not None:
                logger.info("Using native PDF text layer for %s", path)
                return native_text
        return self._run_ocr_pipeline(path)

    def _try_native_text_layer(self, file_path: Path) -> str | None:
        """Return the PDF's embedded text layer, or None if too short to trust."""
        with fitz.open(file_path) as document:
            text = "".join(page.get_text() for page in document)

        if len(text.strip()) >= self.NATIVE_TEXT_MIN_CHARS:
            return text
        logger.debug(
            "Native text layer for %s is too short (%d chars); falling back to OCR",
            file_path,
            len(text.strip()),
        )
        return None

    def _run_ocr_pipeline(self, file_path: Path) -> str:
        """Run fast OCR, escalating to the configured VLM tier if quality is too low."""
        fast_result = self._ocr_router.run_fast_ocr(file_path)
        score = self._ocr_router.quality_score(fast_result)
        if score >= settings.ocr_quality_threshold:
            return fast_result.text

        logger.warning(
            "Fast OCR quality_score=%.3f below threshold=%.2f for %s; falling back to VLM OCR",
            score,
            settings.ocr_quality_threshold,
            file_path,
        )
        vlm_result = self._ocr_router.run_vlm_ocr(file_path)
        return vlm_result.text
