"""Utility functions for vector database operations.

This module provides utility functions for generating unique IDs for documents,
calculating average scores, and converting chunks to minimal representations.
"""

import hashlib
import json
from typing import List

from langchain_core.documents.base import Document

from app.schemas import ChunkSchema


def get_chunk_id(chunk: Document) -> str:
    """Generate a unique ID for a document chunk."""
    serialized = json.dumps(
        {"content": chunk.page_content, "metadata": {k: v for k, v in chunk.metadata.items() if v is not None}}
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def get_avg_score(chunks: List[ChunkSchema], video_id: str) -> float:
    """Calculate the average score for a video."""
    relevant_chunks = [r for r in chunks if r.videoId == video_id]
    if not relevant_chunks:
        return 0.0

    # Extract scores, handling both numeric types
    scores = []
    for r in relevant_chunks:
        score = r.score
        # Convert to float if needed
        if hasattr(score, 'item'):
            # Handle tensor-like objects (if torch is used elsewhere)
            score = score.item()
        scores.append(float(score))

    if not scores:
        return 0.0  # Safeguard against division by zero

    return sum(scores) / len(scores)


def minimise_chunks(chunks: List[dict]) -> List[ChunkSchema]:
    """Convert chunks to a minimal representation."""
    return [
        ChunkSchema(
            text=r["text"],
            start=str(r["start"]),
            end=str(r["end"]),
            videoId=r["video_id"],
            score=r["score"],
        )
        for r in chunks
        if all(key in r for key in ["text", "start", "end", "video_id", "score"])
    ]
