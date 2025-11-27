"""
Tests for BinaryHandler.
"""

import pytest
import asyncio
from pathlib import Path

from pydantic_ai_telegram.binary_handler import BinaryHandler


@pytest.mark.asyncio
async def test_binary_handler_init():
    """Test BinaryHandler initialization."""
    handler = BinaryHandler()
    assert handler.temp_dir.exists()


@pytest.mark.asyncio
async def test_save_and_delete_file():
    """Test saving and deleting a file."""
    handler = BinaryHandler()
    
    # Save a test file
    content = b"Test content"
    file_path = await handler.save_file(content, suffix=".txt", prefix="test_")
    
    assert file_path.exists()
    assert file_path.read_bytes() == content
    
    # Delete the file
    await handler.delete_file(file_path)
    
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_get_file_extension():
    """Test getting file extension from MIME type."""
    handler = BinaryHandler()
    
    # Test with MIME type
    assert handler.get_file_extension("image/jpeg", None) == ".jpg"
    assert handler.get_file_extension("audio/ogg", None) == ".ogg"
    
    # Test with filename
    assert handler.get_file_extension(None, "test.pdf") == ".pdf"
    assert handler.get_file_extension("image/png", "test.jpg") == ".jpg"  # filename takes priority
    
    # Test fallback
    assert handler.get_file_extension(None, None) == ".bin"


@pytest.mark.asyncio
async def test_cleanup_old_files():
    """Test cleaning up old files."""
    handler = BinaryHandler()
    
    # Create a test file
    content = b"Test"
    file_path = await handler.save_file(content, prefix="telegram_bot_cleanup_")
    
    # Cleanup with 0 seconds max age (should delete everything)
    deleted = await handler.cleanup_old_files(max_age_seconds=0)
    
    assert deleted >= 1
    assert not file_path.exists()

