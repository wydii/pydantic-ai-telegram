"""
OpenAI API transcription service using Whisper API.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic_ai_telegram.transcription.base import TranscriptionService

logger = logging.getLogger(__name__)


class OpenAITranscription(TranscriptionService):
    """
    Transcription service using OpenAI's Whisper API.
    Requires openai package and API key.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI transcription service.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (whisper-1)
            language: Language code for transcription (None for auto-detection)
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        self.client: Optional[Any] = None
        
        try:
            from openai import AsyncOpenAI
            self.AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )
    
    def _get_client(self) -> Any:
        """Get or create OpenAI client (lazy loading)."""
        if self.client is None:
            self.client = self.AsyncOpenAI(api_key=self.api_key)
        return self.client
    
    async def transcribe(self, audio_file_path: str | Path) -> str:
        """
        Transcribe an audio file using OpenAI Whisper API.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        audio_path = Path(audio_file_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        client = self._get_client()
        
        logger.info(f"Transcribing audio file with OpenAI: {audio_path}")
        
        # Open file and send to OpenAI
        with open(audio_path, "rb") as audio_file:
            kwargs = {"file": audio_file, "model": self.model}
            
            if self.language:
                kwargs["language"] = self.language
            
            transcript = await client.audio.transcriptions.create(**kwargs)
        
        logger.info("Transcription completed")
        
        return transcript.text.strip()
    
    async def close(self) -> None:
        """Clean up resources (close HTTP client)."""
        if self.client is not None:
            await self.client.close()
            self.client = None
            logger.info("OpenAI client closed")

