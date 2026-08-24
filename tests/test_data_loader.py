import json
from collections import Counter
from pathlib import Path

import pytest

from legal_assistant.data_loader import (
    DataValidationError,
    create_text_nodes,
    load_legal_articles,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_packaged_dataset_contains_expected_unique_articles():
    data_dir = Path(__file__).parents[1] / "data"
    articles = load_legal_articles(data_dir)

    assert len(articles) == 205
    assert len({article.full_title for article in articles}) == 205
    assert {article.law_name for article in articles} == {
        "中华人民共和国劳动法",
        "中华人民共和国劳动合同法",
    }
    assert Counter(article.law_name for article in articles) == {
        "中华人民共和国劳动法": 107,
        "中华人民共和国劳动合同法": 98,
    }


def test_nodes_have_stable_ids_and_traceable_metadata(tmp_path: Path):
    _write_json(tmp_path / "laws.json", [{"中华人民共和国劳动法 第一条": "示例条文"}])
    articles = load_legal_articles(tmp_path)
    nodes = create_text_nodes(articles, node_factory=lambda **kwargs: kwargs)

    assert nodes[0]["id_"] == articles[0].node_id
    assert nodes[0]["metadata"]["full_title"] == "中华人民共和国劳动法 第一条"
    assert nodes[0]["metadata"]["source_file"] == "laws.json"


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        ["not-an-object"],
        [{}],
        [{"中华人民共和国劳动法 第一条": 123}],
        [{"": "正文"}],
        [{"中华人民共和国劳动法 第一条": ""}],
    ],
)
def test_invalid_shapes_are_rejected(tmp_path: Path, value: object):
    _write_json(tmp_path / "invalid.json", value)
    with pytest.raises(DataValidationError):
        load_legal_articles(tmp_path)


def test_duplicate_titles_are_rejected(tmp_path: Path):
    duplicate = "中华人民共和国劳动法 第一条"
    _write_json(tmp_path / "one.json", [{duplicate: "正文一"}])
    _write_json(tmp_path / "two.json", [{duplicate: "正文二"}])

    with pytest.raises(DataValidationError, match="duplicate title"):
        load_legal_articles(tmp_path)


def test_empty_file_is_rejected(tmp_path: Path):
    (tmp_path / "empty.json").write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="failed to read"):
        load_legal_articles(tmp_path)
