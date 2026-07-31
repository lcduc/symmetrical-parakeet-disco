import re

_PII_PATTERNS = [
    re.compile(r"date of birth", re.IGNORECASE),
    re.compile(r"age:\s*\d+", re.IGNORECASE),
    re.compile(r"marital status", re.IGNORECASE),
    re.compile(r"nationality", re.IGNORECASE),
]


def scrub_free_text(text: str) -> str:
    """Strip mentions of protected-characteristic terms from free text.

    Defense-in-depth against date of birth, age, marital status, or
    nationality leaking into a free-text field even though the
    CandidateProfile schema excludes them structurally.
    """
    scrubbed = text
    for pattern in _PII_PATTERNS:
        scrubbed = pattern.sub("", scrubbed)
    return scrubbed
