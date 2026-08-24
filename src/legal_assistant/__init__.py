"""Legal RAG assistant package."""

from .config import AppSettings, ConfigError
from .schemas import AnswerResult, Citation
from .service import LegalAssistant

__all__ = [
    "AnswerResult",
    "AppSettings",
    "Citation",
    "ConfigError",
    "LegalAssistant",
]

__version__ = "1.0.0"

