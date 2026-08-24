"""Load and validate the structured labor-law knowledge base."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    """Raised when a knowledge-base file does not match the expected schema."""


@dataclass(frozen=True)
class LegalArticle:
    """Normalized representation of one legal provision."""

    full_title: str
    content: str
    source_file: str
    law_name: str
    article: str

    @property
    def node_id(self) -> str:
        digest = hashlib.sha256(self.full_title.encode("utf-8")).hexdigest()[:20]
        return f"legal-article-{digest}"

    @classmethod
    def from_raw(cls, full_title: str, content: str, source_file: str) -> "LegalArticle":
        law_name, separator, article = full_title.partition(" ")
        if not separator or not law_name.strip() or not article.strip():
            raise DataValidationError(
                f"{source_file}: title {full_title!r} must contain a law name and article"
            )
        return cls(
            full_title=full_title.strip(),
            content=content.strip(),
            source_file=source_file,
            law_name=law_name.strip(),
            article=article.strip(),
        )


def load_legal_articles(data_dir: str | Path) -> list[LegalArticle]:
    """Load every JSON file in *data_dir* and return validated, unique provisions."""

    directory = Path(data_dir)
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        raise DataValidationError(f"no JSON knowledge-base files found in {directory}")

    articles: list[LegalArticle] = []
    seen_titles: set[str] = set()

    for json_file in json_files:
        try:
            raw_data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"failed to read {json_file.name}: {exc}") from exc

        if not isinstance(raw_data, list) or not raw_data:
            raise DataValidationError(f"{json_file.name}: root value must be a non-empty list")

        for item_index, item in enumerate(raw_data):
            if not isinstance(item, dict) or not item:
                raise DataValidationError(
                    f"{json_file.name}: item {item_index} must be a non-empty object"
                )
            for full_title, content in item.items():
                if not isinstance(full_title, str) or not isinstance(content, str):
                    raise DataValidationError(
                        f"{json_file.name}: titles and contents must both be strings"
                    )
                if not full_title.strip() or not content.strip():
                    raise DataValidationError(
                        f"{json_file.name}: titles and contents cannot be empty"
                    )
                normalized_title = full_title.strip()
                if normalized_title in seen_titles:
                    raise DataValidationError(
                        f"{json_file.name}: duplicate title {normalized_title!r}"
                    )
                seen_titles.add(normalized_title)
                articles.append(
                    LegalArticle.from_raw(normalized_title, content, json_file.name)
                )

    return articles


def create_text_nodes(
    articles: Iterable[LegalArticle],
    node_factory: Callable[..., Any] | None = None,
) -> list[Any]:
    """Convert normalized provisions into LlamaIndex TextNode-compatible objects."""

    if node_factory is None:
        from llama_index.core.schema import TextNode

        node_factory = TextNode

    nodes: list[Any] = []
    for article in articles:
        nodes.append(
            node_factory(
                text=article.content,
                id_=article.node_id,
                metadata={
                    "law_name": article.law_name,
                    "article": article.article,
                    "full_title": article.full_title,
                    "source_file": article.source_file,
                    "content_type": "legal_article",
                },
            )
        )
    return nodes

