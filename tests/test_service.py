from types import SimpleNamespace

import pytest

from legal_assistant.service import (
    NO_MATCH_ANSWER,
    LegalAssistant,
    filter_by_score,
    strip_reasoning_blocks,
)


def _node(title: str, score: float):
    law_name, article = title.split(" ", 1)
    return SimpleNamespace(
        score=score,
        node=SimpleNamespace(
            text=f"{title}的正文",
            metadata={
                "full_title": title,
                "law_name": law_name,
                "article": article,
                "source_file": "labor_laws.json",
            },
        ),
    )


class DummyRetriever:
    def __init__(self, nodes):
        self.nodes = nodes

    def retrieve(self, query):
        return list(self.nodes)


class DummyReranker:
    def postprocess_nodes(self, nodes, query_str):
        return sorted(nodes, key=lambda node: node.score, reverse=True)


class DummySynthesizer:
    def synthesize(self, query, nodes):
        return SimpleNamespace(response="<think>内部推理\n不得展示</think>根据第三十八条，可以解除合同。")


def test_threshold_boundary_is_inclusive():
    nodes = [_node("中华人民共和国劳动法 第一条", 0.4), _node("中华人民共和国劳动法 第二条", 0.399)]
    assert filter_by_score(nodes, 0.4) == [nodes[0]]


def test_no_match_returns_safe_fallback_without_citations():
    assistant = LegalAssistant(
        DummyRetriever([_node("中华人民共和国劳动法 第一条", 0.2)]),
        DummyReranker(),
        DummySynthesizer(),
        score_threshold=0.4,
    )
    result = assistant.answer("一个无匹配问题")

    assert result.answer == NO_MATCH_ANSWER
    assert result.citations == ()
    assert result.matched is False


def test_answer_keeps_serializable_citations_and_removes_reasoning():
    title = "中华人民共和国劳动合同法 第三十八条"
    assistant = LegalAssistant(
        DummyRetriever([_node(title, 0.91)]),
        DummyReranker(),
        DummySynthesizer(),
        score_threshold=0.4,
    )
    result = assistant.answer("劳动者何时可以解除劳动合同？")
    message = result.to_message()

    assert "<think>" not in result.answer
    assert "内部推理" not in result.answer
    assert result.matched is True
    assert message["citations"][0]["full_title"] == title
    assert "think" not in message


def test_empty_question_is_rejected():
    assistant = LegalAssistant(DummyRetriever([]), DummyReranker(), DummySynthesizer(), 0.4)
    with pytest.raises(ValueError, match="cannot be empty"):
        assistant.answer("   ")


def test_reasoning_cleaner_handles_multiline_and_case():
    raw = "前言<THINK data-x='1'>第一行\n第二行</think>最终答案"
    assert strip_reasoning_blocks(raw) == "前言最终答案"

