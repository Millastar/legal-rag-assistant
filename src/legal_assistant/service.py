"""Application service for retrieval, reranking, filtering, and synthesis."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any

from .schemas import AnswerResult, Citation


NO_MATCH_ANSWER = (
    "未在当前知识库中检索到足够相关的法律条文。请调整问题描述，"
    "或向具备资质的专业人士咨询。"
)

_REASONING_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_reasoning_blocks(text: str) -> str:
    """Remove model-specific private reasoning blocks from displayable output."""

    cleaned = _REASONING_BLOCK.sub("", text)
    return cleaned.strip()


def filter_by_score(nodes: Iterable[Any], threshold: float) -> list[Any]:
    """Keep reranked nodes whose score meets the configured threshold."""

    return [node for node in nodes if float(getattr(node, "score", 0.0) or 0.0) >= threshold]


def _citation_from_node(node_with_score: Any) -> Citation:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = getattr(node, "metadata", {}) or {}
    return Citation(
        full_title=str(metadata.get("full_title", "未知条款")),
        law_name=str(metadata.get("law_name", "未知法律")),
        article=str(metadata.get("article", "未知条款")),
        source_file=str(metadata.get("source_file", "未知来源")),
        content=str(getattr(node, "text", "")),
        score=float(getattr(node_with_score, "score", 0.0) or 0.0),
    )


class LegalAssistant:
    """Orchestrate the complete, source-grounded legal QA workflow."""

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        response_synthesizer: Any,
        score_threshold: float,
        citation_limit: int = 3,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.response_synthesizer = response_synthesizer
        self.score_threshold = score_threshold
        self.citation_limit = citation_limit

    def answer(self, question: str) -> AnswerResult:
        query = question.strip()
        if not query:
            raise ValueError("question cannot be empty")

        started_at = time.perf_counter()
        retrieved_nodes = self.retriever.retrieve(query)
        reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_str=query)
        filtered_nodes = filter_by_score(reranked_nodes, self.score_threshold)

        if not filtered_nodes:
            return AnswerResult(
                answer=NO_MATCH_ANSWER,
                citations=(),
                matched=False,
                elapsed_seconds=time.perf_counter() - started_at,
            )

        response = self.response_synthesizer.synthesize(query, nodes=filtered_nodes)
        raw_answer = getattr(response, "response", str(response))
        clean_answer = strip_reasoning_blocks(str(raw_answer))
        if not clean_answer:
            clean_answer = "模型未返回可展示的答复，请稍后重试。"

        citations = tuple(
            _citation_from_node(node) for node in filtered_nodes[: self.citation_limit]
        )
        return AnswerResult(
            answer=clean_answer,
            citations=citations,
            matched=True,
            elapsed_seconds=time.perf_counter() - started_at,
        )

