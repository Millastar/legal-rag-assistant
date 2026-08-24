"""Model and response-synthesizer initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppSettings


LEGAL_QA_TEMPLATE = """\
你是一名劳动法信息检索助手。请严格遵守以下要求：
1. 只能依据“参考法律条文”作答，不得补充条文中不存在的事实。
2. 明确区分劳动者、用人单位及其他主体，避免混淆权利义务和责任归属。
3. 在回答中写明所依据的法律名称和条款；若资料不足，明确说明无法依据当前知识库回答。
4. 使用准确、克制的中文，不将信息检索结果表述为正式法律意见。

参考法律条文：
{context_str}

用户问题：{query_str}

回答：
"""


@dataclass(frozen=True)
class ModelBundle:
    embedding: Any
    llm: Any
    reranker: Any


def create_model_bundle(settings: AppSettings) -> ModelBundle:
    """Initialize embedding, generation, and reranking models."""

    from llama_index.core import Settings
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    embedding = HuggingFaceEmbedding(model_name=settings.embedding_model)
    llm = OpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        context_window=settings.llm_context_window,
        is_chat_model=True,
        is_function_calling_model=False,
    )
    reranker = SentenceTransformerRerank(
        model=settings.rerank_model,
        top_n=settings.rerank_top_k,
    )

    Settings.embed_model = embedding
    Settings.llm = llm
    return ModelBundle(embedding=embedding, llm=llm, reranker=reranker)


def create_response_synthesizer(models: ModelBundle) -> Any:
    """Create a source-constrained LlamaIndex response synthesizer."""

    from llama_index.core import PromptTemplate, get_response_synthesizer

    return get_response_synthesizer(
        llm=models.llm,
        text_qa_template=PromptTemplate(LEGAL_QA_TEMPLATE),
        response_mode="compact",
        verbose=False,
    )

