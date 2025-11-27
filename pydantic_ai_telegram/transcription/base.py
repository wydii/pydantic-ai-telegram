"""
Abstract base class for transcription services.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class TranscriptionService(ABC):
    """
    Abstract base class for audio transcription services.
    Allows pluggable implementations (Local Whisper, OpenAI API, etc.)
    """
    
    @abstractmethod
    async def transcribe(self, audio_file_path: str | Path) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Clean up any resources (models, HTTP clients, etc.)
        """
        pass

