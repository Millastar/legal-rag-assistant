"""Serializable application-domain schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Citation:
    """A legal provision used to support an answer."""

    full_title: str
    law_name: str
    article: str
    source_file: str
    content: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Citation":
        return cls(**value)


@dataclass(frozen=True)
class AnswerResult:
    """A source-grounded answer returned by the application service."""

    answer: str
    citations: tuple[Citation, ...]
    matched: bool
    elapsed_seconds: float

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "matched": self.matched,
            "elapsed_seconds": self.elapsed_seconds,
        }

