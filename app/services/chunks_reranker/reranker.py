"""Service for reranking document chunks using OpenAI GPT models."""

from typing import (
    Any,
    Dict,
    List,
)

from langchain.docstore.document import Document
from structlog import get_logger

from app.services.chunks_reranker.openai_reranker import OpenAIChunksReRanker

logger = get_logger(__name__)


class ChunksReRanker:
    """Reranks document chunks using OpenAI GPT models (backward compatibility wrapper)."""

    def __init__(self, batch_size: int = None):
        """Initialize the reranker with specified batch size.

        Args:
            batch_size: Number of documents to process in each batch.
        """
        self.openai_reranker = OpenAIChunksReRanker(batch_size=batch_size)
        logger.info("Initialized ChunksReRanker (using OpenAI)", batch_size=batch_size)

    @classmethod
    def rerank(cls, query: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """Sync class method to rerank documents using OpenAI GPT model.

        Args:
            query: The search query
            docs: List of documents to rerank

        Returns:
            List of ranked documents with scores
        """
        return OpenAIChunksReRanker.rerank(query, docs)

    def _rerank(self, query: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """Sync internal method to rerank documents using OpenAI GPT model.

        Args:
            query: The search query
            docs: List of documents to rerank

        Returns:
            List of ranked documents with scores
        """
        return self.openai_reranker._rerank(query, docs)
