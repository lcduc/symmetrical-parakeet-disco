from unittest.mock import MagicMock, patch

import fitz

from services.ingestion_service import IngestionService
from services.ocr_router_service import OCRResult


def _make_native_pdf(path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(36, 36, page.rect.width - 36, page.rect.height - 36), text)
    document.save(path)
    document.close()


def test_extract_text_uses_native_layer_without_ocr(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    long_text = "Experienced software engineer. " * 20
    assert len(long_text) >= 200
    _make_native_pdf(pdf_path, long_text)

    mock_router = MagicMock()
    text = IngestionService(ocr_router=mock_router).extract_text(pdf_path)

    mock_router.run_fast_ocr.assert_not_called()
    mock_router.run_vlm_ocr.assert_not_called()
    assert "software engineer" in text.lower()


def test_extract_text_falls_back_to_vlm_on_low_quality(tmp_path):
    image_path = tmp_path / "scanned.png"
    image_path.write_bytes(b"not a real image, just needs to exist")

    low_quality_result = OCRResult(text="garbled text", mean_confidence=0.5, layout_complexity=0.8)
    vlm_result = OCRResult(text="clean vlm transcription", mean_confidence=1.0, layout_complexity=0.0)

    mock_router = MagicMock()
    mock_router.run_fast_ocr.return_value = low_quality_result
    mock_router.quality_score.return_value = 0.5 * (1 - 0.5 * 0.8)
    mock_router.run_vlm_ocr.return_value = vlm_result

    text = IngestionService(ocr_router=mock_router).extract_text(image_path)

    mock_router.run_fast_ocr.assert_called_once()
    mock_router.run_vlm_ocr.assert_called_once()
    assert text == "clean vlm transcription"


def test_extract_text_keeps_fast_ocr_when_quality_is_high(tmp_path):
    image_path = tmp_path / "scanned.png"
    image_path.write_bytes(b"not a real image, just needs to exist")

    high_quality_result = OCRResult(text="crisp text", mean_confidence=0.95, layout_complexity=0.1)

    mock_router = MagicMock()
    mock_router.run_fast_ocr.return_value = high_quality_result
    mock_router.quality_score.return_value = 0.95 * (1 - 0.5 * 0.1)

    text = IngestionService(ocr_router=mock_router).extract_text(image_path)

    mock_router.run_fast_ocr.assert_called_once()
    mock_router.run_vlm_ocr.assert_not_called()
    assert text == "crisp text"


def test_extract_text_cloud_only_mode_skips_native_and_fast_ocr(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    long_text = "Experienced software engineer. " * 20
    _make_native_pdf(pdf_path, long_text)

    mock_router = MagicMock()
    mock_router.run_vlm_ocr.return_value = OCRResult(
        text="cloud transcription", mean_confidence=1.0, layout_complexity=0.0
    )

    with patch("services.ingestion_service.settings.ocr_mode", "cloud_only"):
        text = IngestionService(ocr_router=mock_router).extract_text(pdf_path)

    mock_router.run_fast_ocr.assert_not_called()
    mock_router.run_vlm_ocr.assert_called_once()
    assert text == "cloud transcription"


def test_extract_text_respects_configurable_quality_threshold(tmp_path):
    image_path = tmp_path / "scanned.png"
    image_path.write_bytes(b"not a real image, just needs to exist")

    borderline_result = OCRResult(text="borderline text", mean_confidence=0.7, layout_complexity=0.0)

    mock_router = MagicMock()
    mock_router.run_fast_ocr.return_value = borderline_result
    mock_router.quality_score.return_value = 0.7
    mock_router.run_vlm_ocr.return_value = OCRResult(text="vlm text", mean_confidence=1.0, layout_complexity=0.0)

    # Default threshold (0.85) would escalate a 0.7 score; lowering it should keep the fast-OCR result.
    with patch("services.ingestion_service.settings.ocr_quality_threshold", 0.6):
        text = IngestionService(ocr_router=mock_router).extract_text(image_path)

    mock_router.run_vlm_ocr.assert_not_called()
    assert text == "borderline text"
