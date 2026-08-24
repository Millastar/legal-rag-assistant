"""Persistent Chroma knowledge-base lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppSettings
from .data_loader import create_text_nodes, load_legal_articles


@dataclass
class LegalKnowledgeBase:
    """Build or load the dense-vector index used by the application."""

    settings: AppSettings
    embedding_model: Any
    index: Any | None = None
    article_count: int = 0

    def build_or_load(self) -> "LegalKnowledgeBase":
        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        collection = client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)

        if collection.count() == 0:
            articles = load_legal_articles(self.settings.data_dir)
            nodes = create_text_nodes(articles)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )
            self.article_count = len(articles)
        else:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embedding_model,
            )
            self.article_count = collection.count()

        return self

    def create_retriever(self) -> Any:
        if self.index is None:
            raise RuntimeError("knowledge base must be built or loaded before creating a retriever")
        return self.index.as_retriever(similarity_top_k=self.settings.retrieval_top_k)

