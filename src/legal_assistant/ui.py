"""Streamlit user interface."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from .config import AppSettings
from .knowledge_base import LegalKnowledgeBase
from .models import create_model_bundle, create_response_synthesizer
from .schemas import Citation
from .service import LegalAssistant

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner="正在加载模型与法律知识库……")
def get_assistant() -> tuple[LegalAssistant, AppSettings, int]:
    settings = AppSettings.from_env()
    models = create_model_bundle(settings)
    knowledge_base = LegalKnowledgeBase(settings, models.embedding).build_or_load()
    assistant = LegalAssistant(
        retriever=knowledge_base.create_retriever(),
        reranker=models.reranker,
        response_synthesizer=create_response_synthesizer(models),
        score_threshold=settings.rerank_score_threshold,
        citation_limit=settings.rerank_top_k,
    )
    return assistant, settings, knowledge_base.article_count


def _show_citations(raw_citations: list[dict[str, Any]]) -> None:
    if not raw_citations:
        return
    with st.expander("查看支持依据"):
        for index, raw_citation in enumerate(raw_citations, start=1):
            citation = Citation.from_dict(raw_citation)
            st.markdown(f"**[{index}] {citation.full_title}**")
            st.caption(
                f"来源：{citation.source_file} · 法律：{citation.law_name} · "
                f"重排得分：{citation.score:.4f}"
            )
            st.info(citation.content)


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_citations(message.get("citations", []))
                elapsed = message.get("elapsed_seconds")
                if elapsed is not None:
                    st.caption(f"处理耗时：{elapsed:.2f} 秒")


def main() -> None:
    st.set_page_config(
        page_title="智能劳动法咨询助手",
        page_icon="⚖️",
        layout="centered",
    )
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("基于本地劳动法律条文进行检索、重排序和答案生成。")
    st.warning("本工具仅用于信息检索与技术演示，不构成正式法律意见。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        assistant, settings, article_count = get_assistant()
    except Exception:
        LOGGER.exception("Failed to initialize the legal assistant")
        st.error("系统初始化失败，请检查模型服务、模型文件和环境配置。")
        st.stop()

    with st.sidebar:
        st.subheader("运行信息")
        st.caption(f"知识库条款：{article_count}")
        st.caption(f"生成模型：{settings.llm_model}")
        st.caption(
            f"召回 {settings.retrieval_top_k} 条 · 重排 {settings.rerank_top_k} 条 · "
            f"阈值 {settings.rerank_score_threshold:.2f}"
        )
        if st.button("清空会话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    _render_history()

    if question := st.chat_input("请输入劳动法相关问题"):
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索相关法律条文……"):
                try:
                    result = assistant.answer(question)
                except Exception:
                    LOGGER.exception("Failed to answer question")
                    st.error("本次请求处理失败，请检查模型服务后重试。")
                    return
            st.markdown(result.answer)
            _show_citations([citation.to_dict() for citation in result.citations])
            st.caption(f"处理耗时：{result.elapsed_seconds:.2f} 秒")

        st.session_state.messages.append(result.to_message())

