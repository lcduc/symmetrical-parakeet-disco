import logging
import statistics
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import ClassVar

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    layout_complexity: float


def _boxes_from_polys(polys) -> list[tuple[float, float]]:
    """Reduce OCR quadrilateral polygons to (left_x, width) tuples."""
    boxes = []
    for poly in polys:
        xs = [point[0] for point in poly]
        left = min(xs)
        boxes.append((left, max(xs) - left))
    return boxes


def _estimate_layout_complexity(boxes: list[tuple[float, float]]) -> float:
    """Heuristic based on the spread of distinct x-positions of text boxes.

    A wide spread of left-edge x-positions relative to the page's horizontal
    extent indicates a multi-column or otherwise complex layout.
    """
    if len(boxes) < 2:
        return 0.0

    xs = [left for left, _ in boxes]
    page_extent = max(left + width for left, width in boxes) - min(xs)
    if page_extent <= 0:
        return 0.0

    stdev = statistics.pstdev(xs)
    normalized = (stdev / page_extent) * 2
    return max(0.0, min(1.0, normalized))


class OCRRouterService:
    """Routes OCR requests to the fast (PP-OCRv6) tier or the configured VLM/cloud tier.

    Both the fast-tier and VLM-tier clients are constructed once here rather
    than per call, so their internal lazy-loaded models are loaded at most
    once per process — build one `OCRRouterService` at app startup (see
    `main.py`'s lifespan) and reuse it, mirroring the embedding-model pattern.
    """

    VLM_BACKENDS: ClassVar[dict[str, tuple[str, str]]] = {
        "paddle_vlm": ("services.paddle_vlm_client", "PaddleVLMClient"),
        "datalab": ("services.datalab_client", "DatalabOCRClient"),
        "qwen2_vl": ("services.vlm_client", "VLMClient"),
    }

    def __init__(self):
        from services.pp_ocr_client import PPOCRClient

        self._fast_client = PPOCRClient()

        backend = settings.ocr_vlm_backend
        if backend not in self.VLM_BACKENDS:
            raise ValueError(
                f"Unknown ocr_vlm_backend '{backend}'. Supported values: {sorted(self.VLM_BACKENDS)}"
            )
        module_name, class_name = self.VLM_BACKENDS[backend]
        client_cls = getattr(import_module(module_name), class_name)
        # Only VLMClient (qwen2_vl) takes a device knob; the others don't accept one.
        self._vlm_client = client_cls(device=settings.vlm_device) if backend == "qwen2_vl" else client_cls()

    def run_fast_ocr(self, file_path: str | Path) -> OCRResult:
        """Run the CPU-friendly PP-OCRv6 tier on a single-page image."""
        logger.info("Running fast OCR (PP-OCRv6) on %s", file_path)
        try:
            ocr_data = self._fast_client.run(file_path)
        except Exception:
            logger.exception("Fast OCR failed for %s", file_path)
            raise

        scores = [float(score) for score in ocr_data.get("rec_scores", [])]
        mean_confidence = (sum(scores) / len(scores)) if scores else 0.0

        boxes = _boxes_from_polys(ocr_data.get("rec_polys", []))
        layout_complexity = _estimate_layout_complexity(boxes)

        text = " ".join(word for word in ocr_data.get("rec_texts", []) if word.strip())

        logger.debug(
            "Fast OCR result: mean_confidence=%.3f layout_complexity=%.3f chars=%d",
            mean_confidence,
            layout_complexity,
            len(text),
        )
        return OCRResult(text=text, mean_confidence=mean_confidence, layout_complexity=layout_complexity)

    def quality_score(self, result: OCRResult) -> float:
        """Combine OCR confidence and layout complexity into a single 0-1 score."""
        return result.mean_confidence * (1 - 0.5 * result.layout_complexity)

    def run_vlm_ocr(self, file_path: str | Path) -> OCRResult:
        """Run the configured VLM/cloud OCR tier on a single-page file."""
        backend = settings.ocr_vlm_backend
        logger.info("Running VLM OCR (backend=%s) on %s", backend, file_path)
        try:
            text = self._vlm_client.extract_text(file_path)
        except Exception:
            logger.exception("VLM OCR (backend=%s) failed for %s", backend, file_path)
            raise

        return OCRResult(text=text, mean_confidence=1.0, layout_complexity=0.0)
