"""Service for reranking document chunks using OpenAI GPT models."""

import asyncio
from typing import Any, Dict, List

import structlog
from django.conf import settings
from langchain.docstore.document import Document
from openai import OpenAI

logger = structlog.get_logger(__name__)


class OpenAIChunksReRanker:
    """Reranks document chunks using OpenAI GPT models for relevance scoring."""

    def __init__(self, batch_size: int = None):
        """Initialize the reranker with specified batch size.

        Args:
            batch_size: Number of documents to process in each batch.
                Defaults to settings.RERANKING_BATCH_SIZE if not provided.
        """
        self.batch_size = batch_size or settings.RERANKING_BATCH_SIZE
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

        logger.info("Initialized OpenAIChunksReRanker", batch_size=self.batch_size, model=self.model)

    @classmethod
    def rerank(cls, query: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """Sync class method to rerank documents using OpenAI GPT model.

        Args:
            query: The search query
            docs: List of documents to rerank

        Returns:
            List of ranked documents with scores
        """
        reranker = cls()
        return reranker._rerank(query, docs)

    def _rerank(self, query: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """Sync internal method to rerank documents using OpenAI GPT model.

        Args:
            query: The search query
            docs: List of documents to rerank

        Returns:
            List of ranked documents with scores
        """
        if not docs:
            logger.warning("No documents provided for reranking")
            return []

        logger.info("Starting OpenAI reranking", num_docs=len(docs))

        # Process documents in batches to manage API costs
        doc_chunks = self._chunk_docs(docs, self.batch_size)
        all_ranked_docs = []

        for i, chunk in enumerate(doc_chunks):
            try:
                logger.debug(f"Processing batch {i+1}/{len(doc_chunks)}", batch_size=len(chunk))
                ranked_chunk = self._process_chunk(query, chunk)
                all_ranked_docs.extend(ranked_chunk)
            except Exception as e:
                logger.error("Error processing chunk", chunk_index=i, error=str(e))
                # Return documents with zero score in case of error
                all_ranked_docs.extend([
                    {"score": 0.0, **doc.metadata, "text": doc.page_content}
                    for doc in chunk
                ])

        # Sort all ranked documents by score from highest to lowest
        all_ranked_docs.sort(key=lambda x: x["score"], reverse=True)

        logger.info("Completed OpenAI reranking", num_docs=len(all_ranked_docs))
        return all_ranked_docs

    @staticmethod
    def _chunk_docs(docs: List[Document], size: int) -> List[List[Document]]:
        """Divide the documents into chunks of the given size.

        Args:
            docs: List of documents to chunk
            size: Size of each chunk

        Returns:
            List of document chunks
        """
        return [docs[i : i + size] for i in range(0, len(docs), size)]

    def _process_chunk(self, query: str, chunk: List[Document]) -> List[Dict[str, Any]]:
        """Process a single chunk of documents using OpenAI API.

        Args:
            query: Search query
            chunk: List of documents to process

        Returns:
            List of ranked documents with scores
        """
        try:
            # Create prompt for GPT model
            prompt = self._create_reranking_prompt(query, chunk)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a relevance scoring expert. Score how relevant each text segment is to the query on a scale of 0.0 to 1.0, where 1.0 is highly relevant and 0.0 is not relevant at all."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Deterministic output
                max_tokens=500,
            )

            # Parse the response
            scores = self._parse_scores(response.choices[0].message.content, len(chunk))

            # Create ranked documents with scores
            ranked_docs = []
            for doc, score in zip(chunk, scores, strict=False):
                ranked_docs.append({
                    "score": float(score),
                    **doc.metadata,
                    "text": doc.page_content
                })

            logger.debug("Successfully processed chunk with OpenAI", num_docs=len(chunk))
            return ranked_docs

        except Exception as e:
            logger.error("Error processing chunk with OpenAI", error=str(e))
            # Return documents with zero score in case of error
            return [{"score": 0.0, **doc.metadata, "text": doc.page_content} for doc in chunk]

    def _create_reranking_prompt(self, query: str, docs: List[Document]) -> str:
        """Create prompt for reranking task.

        Args:
            query: Search query
            docs: List of documents

        Returns:
            Formatted prompt string
        """
        docs_text = ""
        for i, doc in enumerate(docs):
            # Truncate long documents to save tokens
            text = doc.page_content[:500] if len(doc.page_content) > 500 else doc.page_content
            docs_text += f"\n[{i}] {text}\n"

        prompt = f"""Query: {query}

Text Segments:
{docs_text}

Rate the relevance of each text segment to the query. Return only the scores in this exact format:
0: [score]
1: [score]
2: [score]
...

Where each score is a number between 0.0 and 1.0."""

        return prompt

    def _parse_scores(self, response_text: str, expected_count: int) -> List[float]:
        """Parse scores from GPT response.

        Args:
            response_text: Response text from GPT
            expected_count: Expected number of scores

        Returns:
            List of scores (defaults to 0.0 if parsing fails)
        """
        scores = []
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        try:
                            score = float(parts[1].strip())
                            # Ensure score is between 0 and 1
                            score = max(0.0, min(1.0, score))
                            scores.append(score)
                        except ValueError:
                            scores.append(0.0)

            # Ensure we have the right number of scores
            while len(scores) < expected_count:
                scores.append(0.0)

            return scores[:expected_count]

        except Exception as e:
            logger.error("Error parsing scores from OpenAI response", error=str(e))
            return [0.0] * expected_count
