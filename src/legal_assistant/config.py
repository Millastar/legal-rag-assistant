"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in production
    load_dotenv = None


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw_value!r}") from exc


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw_value!r}") from exc


def _resolve_path(root: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class AppSettings:
    """Validated runtime settings for the application."""

    project_root: Path
    data_dir: Path
    cache_dir: Path
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    llm_context_window: int
    embedding_model: str
    rerank_model: str
    retrieval_top_k: int
    rerank_top_k: int
    rerank_score_threshold: float
    collection_name: str

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AppSettings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        if load_dotenv is not None:
            load_dotenv(root / ".env", override=False)

        settings = cls(
            project_root=root,
            data_dir=_resolve_path(root, os.getenv("DATA_DIR", "data")),
            cache_dir=_resolve_path(root, os.getenv("CACHE_DIR", ".cache")),
            llm_api_base=os.getenv("LLM_API_BASE", "http://localhost:23333/v1").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "EMPTY").strip(),
            llm_model=os.getenv("LLM_MODEL", "Qwen--Qwen2.5-7B-Instruct-law").strip(),
            llm_context_window=_read_int("LLM_CONTEXT_WINDOW", 4096),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sungw111/text2vec-base-chinese-sentence"
            ).strip(),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large").strip(),
            retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 10),
            rerank_top_k=_read_int("RERANK_TOP_K", 3),
            rerank_score_threshold=_read_float("RERANK_SCORE_THRESHOLD", 0.4),
            collection_name=os.getenv("COLLECTION_NAME", "chinese_labor_laws").strip(),
        )
        settings.validate()
        return settings

    @property
    def chroma_dir(self) -> Path:
        return self.cache_dir / "chroma"

    def validate(self) -> None:
        required_strings = {
            "LLM_API_BASE": self.llm_api_base,
            "LLM_MODEL": self.llm_model,
            "EMBEDDING_MODEL": self.embedding_model,
            "RERANK_MODEL": self.rerank_model,
            "COLLECTION_NAME": self.collection_name,
        }
        for name, value in required_strings.items():
            if not value:
                raise ConfigError(f"{name} cannot be empty")

        if self.llm_context_window <= 0:
            raise ConfigError("LLM_CONTEXT_WINDOW must be greater than zero")
        if self.retrieval_top_k <= 0 or self.rerank_top_k <= 0:
            raise ConfigError("retrieval and rerank top-k values must be greater than zero")
        if self.rerank_top_k > self.retrieval_top_k:
            raise ConfigError("RERANK_TOP_K cannot exceed RETRIEVAL_TOP_K")
        if not 0.0 <= self.rerank_score_threshold <= 1.0:
            raise ConfigError("RERANK_SCORE_THRESHOLD must be between 0 and 1")

