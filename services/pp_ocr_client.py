import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class PPOCRClient:
    """CPU-friendly fast OCR tier using PaddleOCR's PP-OCRv6 small detection/recognition models.

    Intended to be constructed once and reused (see OCRRouterService); `_load`
    is safe to call from concurrent threads (`asyncio.to_thread` workers).
    """

    DET_MODEL_NAME = "PP-OCRv6_small_det"
    REC_MODEL_NAME = "PP-OCRv6_small_rec"

    def __init__(self):
        self._engine = None
        self._load_lock = Lock()

    def _load(self) -> None:
        """Lazily construct the PaddleOCR engine on first use."""
        if self._engine is not None:
            return
        with self._load_lock:
            if self._engine is not None:
                return

            logger.info("Loading PP-OCRv6 small models (%s, %s)", self.DET_MODEL_NAME, self.REC_MODEL_NAME)
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                text_detection_model_name=self.DET_MODEL_NAME,
                text_recognition_model_name=self.REC_MODEL_NAME,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # oneDNN's PIR executor hits an unimplemented op conversion on this stack
                # (ConvertPirAttribute2RuntimeAttribute) for the text detection model.
                enable_mkldnn=False,
            )

    def run(self, file_path: str | Path) -> dict:
        """Run PP-OCRv6 on a single-page image and return the raw result dict for that page."""
        self._load()
        results = self._engine.predict(str(file_path))
        if not results:
            logger.warning("PP-OCRv6 returned no results for %s", file_path)
            return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
        return results[0].json.get("res", results[0].json)
