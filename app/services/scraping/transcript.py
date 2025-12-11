"""Transcript-related functionality for YouTube scraping."""

from typing import (
    Dict,
    List,
)

import structlog

from app.services.scraping.whisper_transcript import WhisperTranscriptScraper

logger = structlog.get_logger(__name__)


class TranscriptScraper:
    """Transcript-related functionality for YouTube scraping."""

    def __init__(self, max_transcript_segment_duration: int = 40):
        """Initialize the transcript scraper.

        Args:
            max_transcript_segment_duration: Maximum duration for transcript segments in seconds
        """
        self.max_transcript_segment_duration = max_transcript_segment_duration
        self.whisper_scraper = WhisperTranscriptScraper(max_transcript_segment_duration)

    def get_video_transcript(self, video_metadata: Dict) -> List[Dict]:
        """Fetches and formats the transcript of a YouTube video using OpenAI Whisper.

        Args:
            video_metadata: Dictionary containing video metadata

        Returns:
            List[Dict]: List of transcript segments
        """
        video_id = video_metadata["videoId"]
        
        try:
            logger.info("Attempting transcription with Whisper API", video_id=video_id)
            return self.whisper_scraper.get_video_transcript(video_metadata)

        except Exception as e:
            logger.error(
                f"Failed to fetch transcript for https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                error=e,
            )
            return []
