import re

_DEGREE_LEVEL_KEYWORDS: list[tuple[str, str]] = [
    ("phd", "phd"),
    ("ph.d", "phd"),
    ("doctor", "phd"),
    ("master", "master"),
    ("msc", "master"),
    ("m.sc", "master"),
    ("mba", "master"),
    ("bachelor", "bachelor"),
    ("bsc", "bachelor"),
    ("b.sc", "bachelor"),
    ("beng", "bachelor"),
    ("b.eng", "bachelor"),
    ("associate", "associate"),
    ("high school", "high_school"),
    ("diploma", "high_school"),
]


def infer_education_level(degree: str | None) -> str | None:
    """Infer a ranked education level from a free-text degree string.

    Matches against common degree abbreviations (BSc, MSc, PhD, MBA, ...)
    so MatchingService's EDUCATION_RANK lookup has something to compare
    against even when an LLM extraction leaves `level` null but populates
    `degree`. Returns None if no keyword matches, rather than guessing.
    """
    if not degree:
        return None
    normalized = degree.lower()
    for keyword, level in _DEGREE_LEVEL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return level
    return None
