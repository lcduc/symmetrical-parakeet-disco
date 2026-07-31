from utils.pii_scrub import scrub_free_text


def test_scrub_free_text_removes_marital_status_mention():
    text = "Built backend services. Marital Status: Single. Led a team of 4."

    scrubbed = scrub_free_text(text)

    assert "marital status" not in scrubbed.lower()
    assert "Built backend services." in scrubbed
    assert "Led a team of 4." in scrubbed
