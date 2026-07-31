from unittest.mock import patch

import pytest

from services.ocr_router_service import (
    OCRResult,
    OCRRouterService,
    _boxes_from_polys,
    _estimate_layout_complexity,
)


def test_quality_score_combines_confidence_and_layout():
    result = OCRResult(text="x", mean_confidence=0.9, layout_complexity=0.2)
    assert OCRRouterService().quality_score(result) == 0.9 * (1 - 0.5 * 0.2)


def test_quality_score_is_lower_for_complex_layouts():
    router = OCRRouterService()
    simple = OCRResult(text="x", mean_confidence=0.9, layout_complexity=0.0)
    complex_ = OCRResult(text="x", mean_confidence=0.9, layout_complexity=1.0)
    assert router.quality_score(complex_) < router.quality_score(simple)


def test_layout_complexity_is_zero_for_single_column():
    boxes = [(100, 50), (102, 48), (101, 52), (99, 51)]
    assert _estimate_layout_complexity(boxes) < 0.1


def test_layout_complexity_is_higher_for_wide_spread_columns():
    boxes = [(10, 50), (900, 50), (15, 50), (895, 50)]
    assert _estimate_layout_complexity(boxes) > 0.5


def test_layout_complexity_needs_at_least_two_boxes():
    assert _estimate_layout_complexity([(10, 50)]) == 0.0
    assert _estimate_layout_complexity([]) == 0.0


def test_boxes_from_polys_extracts_left_and_width():
    polys = [
        [[10, 0], [60, 0], [60, 20], [10, 20]],
        [[100, 0], [130, 0], [130, 20], [100, 20]],
    ]
    boxes = _boxes_from_polys(polys)
    assert boxes == [(10, 50), (100, 30)]


def test_run_fast_ocr_computes_confidence_and_text_from_pp_ocr():
    pp_ocr_result = {
        "rec_texts": ["Hello", "World"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": [
            [[10, 0], [60, 0], [60, 20], [10, 20]],
            [[15, 30], [65, 30], [65, 50], [15, 50]],
        ],
    }
    with patch("services.pp_ocr_client.PPOCRClient.run", return_value=pp_ocr_result):
        result = OCRRouterService().run_fast_ocr("dummy.png")

    assert result.text == "Hello World"
    assert result.mean_confidence == pytest.approx(0.85)
    assert 0.0 <= result.layout_complexity <= 1.0


def test_run_vlm_ocr_dispatches_to_configured_backend():
    with patch(
        "services.paddle_vlm_client.PaddleVLMClient.extract_text", return_value="paddle text"
    ) as mock_extract, patch("services.ocr_router_service.settings.ocr_vlm_backend", "paddle_vlm"):
        result = OCRRouterService().run_vlm_ocr("dummy.png")

    mock_extract.assert_called_once()
    assert result.text == "paddle text"


def test_run_vlm_ocr_rejects_unknown_backend():
    with (
        patch("services.ocr_router_service.settings.ocr_vlm_backend", "not_a_real_backend"),
        pytest.raises(ValueError),
    ):
        OCRRouterService().run_vlm_ocr("dummy.png")
