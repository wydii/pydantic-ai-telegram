"""
Pydantic AI Telegram - A library for integrating Pydantic AI agents with Telegram bots.
"""

from pydantic_ai_telegram.bot import TelegramAgent
from pydantic_ai_telegram.models import BotConfig
from pydantic_ai_telegram.transcription.base import TranscriptionService
from pydantic_ai_telegram.transcription.whisper_local import LocalWhisperTranscription

__version__ = "0.1.0"
__all__ = ["TelegramAgent", "BotConfig", "TranscriptionService", "LocalWhisperTranscription"]

