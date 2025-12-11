"""Utility functions for YouTube scraping."""

import itertools
import re
from typing import (
    Any,
    List,
    Optional,
)

import scrapetube
import structlog

logger = structlog.get_logger(__name__)


def get_channel_username(channel_link: str) -> str:
    """Extract the username/ID from a YouTube channel link.

    Supports multiple formats:
    - https://www.youtube.com/@username -> username
    - https://www.youtube.com/channel/CHANNEL_ID -> CHANNEL_ID
    - https://www.youtube.com/c/customname -> customname

    Args:
        channel_link: The full YouTube channel URL

    Returns:
        str: The extracted channel username or ID
    """
    channel_link = channel_link.strip()

    # Try different URL formats
    if "/@" in channel_link:
        username = channel_link.split("https://www.youtube.com/@")[1].strip()
    elif "/channel/" in channel_link:
        username = channel_link.split("https://www.youtube.com/channel/")[1].strip()
    elif "/c/" in channel_link:
        username = channel_link.split("https://www.youtube.com/c/")[1].strip()
    else:
        username = channel_link

    logger.debug("Extracted channel identifier", username=username)
    return username


def validate_channel_link(channel_link: str) -> str:
    """Validate the YouTube channel link format and existence.

    Supports multiple YouTube channel URL formats:
    - https://www.youtube.com/@username
    - https://www.youtube.com/channel/CHANNEL_ID
    - https://www.youtube.com/c/customname

    Args:
        channel_link: The YouTube channel link to validate

    Returns:
        str: The validated channel username or ID

    Raises:
        ValueError: If the channel link is invalid or the channel doesn't exist
    """
    if not channel_link:
        error_msg = "Channel link cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    channel_link = channel_link.strip()

    # Match different YouTube channel URL formats
    handle_match = re.match(r"https://www\.youtube\.com/@(.+)", channel_link)
    channel_id_match = re.match(r"https://www\.youtube\.com/channel/(.+)", channel_link)
    custom_match = re.match(r"https://www\.youtube\.com/c/(.+)", channel_link)

    if handle_match:
        # Handle format: @username
        channel_identifier = handle_match.group(1)
        logger.debug("Detected handle format", identifier=channel_identifier)
    elif channel_id_match:
        # Channel ID format: channel/UC...
        channel_identifier = channel_id_match.group(1)
        logger.debug("Detected channel ID format", identifier=channel_identifier)
    elif custom_match:
        # Custom URL format: /c/customname
        channel_identifier = custom_match.group(1)
        logger.debug("Detected custom URL format", identifier=channel_identifier)
    else:
        error_msg = "Invalid YouTube channel link format. Supported formats: @username, /channel/ID, /c/customname"
        logger.error(error_msg, channel_link=channel_link)
        raise ValueError(error_msg)

    try:
        # Try to fetch channel to validate it exists
        scrapetube.get_channel(channel_identifier)
        logger.info("Channel validated successfully", channel_identifier=channel_identifier)
        return channel_identifier
    except Exception as e:
        logger.error("Channel validation failed", channel_identifier=channel_identifier, error=e)
        raise ValueError(f"Invalid YouTube channel: {str(e)}") from e


def chunk_generator(items: List[Any], chunk_size: int):
    """Generate chunks of items with a specified size.

    Args:
        items: List of items to chunk
        chunk_size: Size of each chunk

    Yields:
        List[Any]: Chunks of items
    """
    iterator = iter(items)
    chunk_count = 0

    while chunk := list(itertools.islice(iterator, chunk_size)):
        chunk_count += 1
        logger.debug(f"Processing chunk {chunk_count}", chunk_size=len(chunk))
        yield chunk
