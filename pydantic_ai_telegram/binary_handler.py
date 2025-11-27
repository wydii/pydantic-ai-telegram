"""
Temporary file management and cleanup for binary content.
Handles downloading, processing, and automatic cleanup of files.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger(__name__)


class BinaryHandler:
    """
    Manages temporary files for binary content processing.
    Ensures proper cleanup after processing.
    """
    
    def __init__(self, temp_dir: Optional[str] = None) -> None:
        """
        Initialize binary handler.
        
        Args:
            temp_dir: Custom temporary directory (uses system default if None)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_temp_file(
        self,
        suffix: str = "",
        prefix: str = "telegram_bot_",
    ) -> Path:
        """
        Create a temporary file path.
        
        Args:
            suffix: File suffix (e.g., ".jpg", ".mp3")
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        fd, path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(self.temp_dir),
        )
        # Close the file descriptor
        os.close(fd)
        
        return Path(path)
    
    async def save_file(
        self,
        content: bytes,
        suffix: str = "",
        prefix: str = "telegram_bot_",
    ) -> Path:
        """
        Save binary content to a temporary file.
        
        Args:
            content: File content as bytes
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to saved file
        """
        file_path = self.create_temp_file(suffix=suffix, prefix=prefix)
        
        # Write file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._write_file_sync,
            file_path,
            content,
        )
        
        logger.debug(f"Saved temporary file: {file_path}")
        return file_path
    
    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """
        Write file synchronously (runs in thread pool).
        
        Args:
            file_path: Path to write to
            content: Content to write
        """
        with open(file_path, "wb") as f:
            f.write(content)
    
    async def delete_file(self, file_path: str | Path) -> None:
        """
        Delete a temporary file.
        
        Args:
            file_path: Path to file to delete
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.debug(f"File already deleted or doesn't exist: {path}")
            return
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, path.unlink)
            logger.debug(f"Deleted temporary file: {path}")
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
    
    async def cleanup_old_files(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_seconds: Maximum age of files to keep (default: 1 hour)
            
        Returns:
            Number of files deleted
        """
        import time
        
        deleted_count = 0
        current_time = time.time()
        
        try:
            for file_path in self.temp_dir.glob("telegram_bot_*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        await self.delete_file(file_path)
                        deleted_count += 1
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old temporary files")
        
        return deleted_count
    
    def get_file_extension(self, mime_type: Optional[str], filename: Optional[str]) -> str:
        """
        Determine file extension from MIME type or filename.
        
        Args:
            mime_type: MIME type
            filename: Original filename
            
        Returns:
            File extension (with dot)
        """
        # Try to get extension from filename first
        if filename:
            ext = Path(filename).suffix
            if ext:
                return ext
        
        # Common MIME type mappings
        mime_extensions = {
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
            "application/zip": ".zip",
            "application/json": ".json",
            "text/plain": ".txt",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
        }
        
        if mime_type and mime_type in mime_extensions:
            return mime_extensions[mime_type]
        
        return ".bin"  # Default binary extension

