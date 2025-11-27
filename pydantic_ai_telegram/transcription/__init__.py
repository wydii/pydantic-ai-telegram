"""
Transcription services for audio processing.
"""

from pydantic_ai_telegram.transcription.base import TranscriptionService

# Import implementations if available
try:
    from pydantic_ai_telegram.transcription.whisper_local import (
        LocalWhisperTranscription,
        check_ffmpeg_installed,
    )
    __all__ = ["TranscriptionService", "LocalWhisperTranscription", "check_ffmpeg_installed"]
except ImportError:
    __all__ = ["TranscriptionService"]

try:
    from pydantic_ai_telegram.transcription.openai_api import OpenAITranscription  # noqa: F401
    __all__.append("OpenAITranscription")
except ImportError:
    pass

