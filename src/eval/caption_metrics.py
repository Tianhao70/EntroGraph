from __future__ import annotations

import re
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text)))


def average_word_count(texts: Iterable[str]) -> float:
    values = [word_count(text) for text in texts]
    return sum(values) / len(values) if values else 0.0

