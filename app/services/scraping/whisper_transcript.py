"""OpenAI Whisper-based transcript extraction for YouTube videos."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import structlog
import yt_dlp
from openai import OpenAI

from app.helpers import convert_seconds_to_timestamp

logger = structlog.get_logger(__name__)


class WhisperTranscriptScraper:
    """Transcript extraction using OpenAI Whisper API for videos without subtitles."""

    def __init__(self, max_transcript_segment_duration: int = 40):
        """Initialize the Whisper transcript scraper.

        Args:
            max_transcript_segment_duration: Maximum duration for transcript segments in seconds
        """
        self.max_transcript_segment_duration = max_transcript_segment_duration
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_video_transcript(self, video_metadata: Dict) -> List[Dict]:
        """Fetches and formats the transcript of a YouTube video using Whisper API.

        Args:
            video_metadata: Dictionary containing video metadata

        Returns:
            List[Dict]: List of transcript segments
        """
        video_id = video_metadata["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            logger.info("Starting Whisper transcription", video_id=video_id)

            # Download audio from YouTube
            audio_path = self._download_audio(video_url, video_id)

            if not audio_path:
                logger.error("Failed to download audio", video_id=video_id)
                return []

            try:
                # Transcribe using OpenAI Whisper API
                transcript_data = self._transcribe_with_whisper(audio_path, video_id)

                if not transcript_data:
                    logger.error("Failed to transcribe audio", video_id=video_id)
                    return []

                # Format the transcript into segments
                formatted_transcript = self._format_whisper_transcript(
                    transcript_data, video_metadata
                )

                logger.info(
                    "Successfully transcribed video",
                    video_id=video_id,
                    segments=len(formatted_transcript),
                )

                return formatted_transcript

            finally:
                # Clean up the audio file
                self._cleanup_audio(audio_path)

        except Exception as e:
            logger.error(
                "Failed to process video transcript",
                video_id=video_id,
                error=str(e),
            )
            return []

    def _download_audio(self, video_url: str, video_id: str) -> Optional[str]:
        """Download audio from YouTube video.

        Args:
            video_url: YouTube video URL
            video_id: YouTube video ID

        Returns:
            Optional[str]: Path to downloaded audio file, or None if failed
        """
        try:
            # Create temporary directory for audio file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"yt_audio_{video_id}.mp3")

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '96',
                }],
                'outtmpl': output_path.replace('.mp3', ''),
                'quiet': True,
                'no_warnings': True,
            }

            logger.info("Attempting to download audio", video_url=video_url, output_path=output_path)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            logger.info("yt-dlp download finished", output_path=output_path)

            if os.path.exists(output_path):
                logger.debug("Audio downloaded successfully", path=output_path)
                return output_path

            logger.error("Audio file not found after download", expected_path=output_path)
            return None

        except Exception as e:
            logger.error("Error downloading audio", video_id=video_id, error=str(e))
            return None

    def _transcribe_with_whisper(self, audio_path: str, video_id: str) -> Optional[Dict]:
        """Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            video_id: Video ID for logging

        Returns:
            Optional[Dict]: Transcript data with timestamps, or None if failed
        """
        try:
            # Check file size (OpenAI limit is 25MB)
            file_size = os.path.getsize(audio_path)
            if file_size > 25 * 1024 * 1024:
                logger.warning(
                    "Audio file exceeds OpenAI limit",
                    size_mb=file_size / (1024 * 1024),
                )
                return None

            with open(audio_path, "rb") as audio_file:
                # Request transcript with timestamps
                transcript = self.client.audio.transcriptions.create(
                    # model="whisper-1",
                    model = "gpt-4o-transcribe",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
                logger.info("OpenAI API response received", video_id=video_id if 'video_id' in locals() else 'unknown')

            logger.debug("Whisper transcription completed")
            return transcript.model_dump()

        except Exception as e:
            logger.error("Error transcribing with Whisper", error=str(e))
            return None

    def _format_whisper_transcript(
        self, transcript_data: Dict, video_metadata: Dict
    ) -> List[Dict]:
        """Format Whisper transcript data into standardized segments.

        Args:
            transcript_data: Raw transcript data from Whisper API
            video_metadata: Dictionary containing video metadata

        Returns:
            List[Dict]: List of formatted transcript segments
        """
        formatted_transcript = []
        video_id = video_metadata["videoId"]

        try:
            segments = transcript_data.get("segments", [])
            logger.info("Whisper raw segments found", video_id=video_id, count=len(segments))

            if not segments:
                # Fallback: create single segment with full text
                text = transcript_data.get("text", "")
                logger.info("No segments found, checking for full text", video_id=video_id, text_length=len(text))
                if text:
                    formatted_transcript.append({
                        "video_id": video_id,
                        "start_time": 0,
                        "timestamp": "00:00:00",
                        "text": text.strip(),
                        "duration": 0,
                    })
                return formatted_transcript

            # Combine segments that are too short
            current_segment = None

            for segment in segments:
                segment_text = segment.get("text", "").strip()
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", segment_start)
                segment_duration = segment_end - segment_start

                if current_segment is None:
                    # Start new segment
                    current_segment = {
                        "video_id": video_id,
                        "start_time": segment_start,
                        "timestamp": convert_seconds_to_timestamp(segment_start),
                        "text": segment_text,
                        "duration": segment_duration,
                    }
                else:
                    # Check if we should continue current segment or start new one
                    total_duration = current_segment["duration"] + segment_duration

                    if total_duration <= self.max_transcript_segment_duration:
                        # Continue current segment
                        current_segment["text"] += " " + segment_text
                        current_segment["duration"] = total_duration
                    else:
                        # Save current segment and start new one
                        formatted_transcript.append(current_segment)
                        current_segment = {
                            "video_id": video_id,
                            "start_time": segment_start,
                            "timestamp": convert_seconds_to_timestamp(segment_start),
                            "text": segment_text,
                            "duration": segment_duration,
                        }

            # Add the last segment
            if current_segment:
                formatted_transcript.append(current_segment)

            return formatted_transcript

        except Exception as e:
            logger.error("Error formatting Whisper transcript", video_id=video_id, error=str(e))
            return []

    def _cleanup_audio(self, audio_path: str) -> None:
        """Remove temporary audio file.

        Args:
            audio_path: Path to audio file to remove
        """
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug("Audio file cleaned up", path=audio_path)
        except Exception as e:
            logger.warning("Failed to cleanup audio file", path=audio_path, error=str(e))
