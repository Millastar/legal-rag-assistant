from pathlib import Path

import pytest

from legal_assistant.config import AppSettings, ConfigError


def test_relative_paths_resolve_from_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_DIR", "custom-data")
    monkeypatch.setenv("CACHE_DIR", "runtime-cache")
    settings = AppSettings.from_env(tmp_path)

    assert settings.data_dir == (tmp_path / "custom-data").resolve()
    assert settings.cache_dir == (tmp_path / "runtime-cache").resolve()


def test_environment_overrides_retrieval_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "20")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("RERANK_SCORE_THRESHOLD", "0.75")
    settings = AppSettings.from_env(tmp_path)

    assert settings.retrieval_top_k == 20
    assert settings.rerank_top_k == 5
    assert settings.rerank_score_threshold == 0.75


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_MODEL", ""),
        ("RERANK_SCORE_THRESHOLD", "1.2"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RERANK_TOP_K", "999"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        AppSettings.from_env(tmp_path)

