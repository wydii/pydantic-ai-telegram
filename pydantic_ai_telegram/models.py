"""
Pydantic v2 models for Telegram data structures and internal state management.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# Telegram API Models
class TelegramUser(BaseModel):
    """Represents a Telegram user or bot."""
    
    model_config = ConfigDict(extra='ignore')
    
    id: int
    is_bot: bool
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None


class TelegramChat(BaseModel):
    """Represents a Telegram chat."""
    
    model_config = ConfigDict(extra='ignore')
    
    id: int
    type: str  # "private", "group", "supergroup", "channel"
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class PhotoSize(BaseModel):
    """Represents one size of a photo or a file/sticker thumbnail."""
    
    model_config = ConfigDict(extra='ignore')
    
    file_id: str
    file_unique_id: str
    width: int
    height: int
    file_size: Optional[int] = None


class TelegramDocument(BaseModel):
    """Represents a general file (document)."""
    
    model_config = ConfigDict(extra='ignore')
    
    file_id: str
    file_unique_id: str
    thumb: Optional[PhotoSize] = None
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramVoice(BaseModel):
    """Represents a voice note."""
    
    model_config = ConfigDict(extra='ignore')
    
    file_id: str
    file_unique_id: str
    duration: int
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramAudio(BaseModel):
    """Represents an audio file."""
    
    model_config = ConfigDict(extra='ignore')
    
    file_id: str
    file_unique_id: str
    duration: int
    performer: Optional[str] = None
    title: Optional[str] = None
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramMessage(BaseModel):
    """Represents a Telegram message."""
    
    model_config = ConfigDict(extra='ignore')
    
    message_id: int
    date: int
    chat: TelegramChat
    from_user: Optional[TelegramUser] = Field(None, alias='from')
    text: Optional[str] = None
    voice: Optional[TelegramVoice] = None
    audio: Optional[TelegramAudio] = None
    document: Optional[TelegramDocument] = None
    photo: Optional[list[PhotoSize]] = None
    caption: Optional[str] = None
    reply_to_message: Optional['TelegramMessage'] = None


class TelegramUpdate(BaseModel):
    """Represents an incoming update."""
    
    model_config = ConfigDict(extra='ignore')
    
    update_id: int
    message: Optional[TelegramMessage] = None
    edited_message: Optional[TelegramMessage] = None
    channel_post: Optional[TelegramMessage] = None
    edited_channel_post: Optional[TelegramMessage] = None


class TelegramFile(BaseModel):
    """Represents a file ready to be downloaded."""
    
    model_config = ConfigDict(extra='ignore')
    
    file_id: str
    file_unique_id: str
    file_size: Optional[int] = None
    file_path: Optional[str] = None


# Internal Models
class BotConfig(BaseModel):
    """Configuration for the Telegram bot."""
    
    model_config = ConfigDict(extra='ignore')
    
    bot_token: str
    allowed_chat_ids: Optional[list[int]] = None
    allowed_usernames: Optional[list[str]] = None
    openai_api_key: Optional[str] = None
    transcription_service: str = "local"  # "local" or "openai"
    whisper_model: str = "turbo"  # For local whisper
    max_history_messages: int = 50
    polling_timeout: int = 30


class ConversationMessage(BaseModel):
    """Represents a single message in a conversation."""
    
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    
    role: str  # "user" or "assistant"
    content: str | list[Any] | Any  # Can be text, multimodal content, or pydantic-ai message
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: Optional[int] = None
    tokens: Optional[int] = None


class ConversationState(BaseModel):
    """Represents the state of a conversation."""
    
    model_config = ConfigDict(extra='ignore')
    
    chat_id: int
    messages: list[ConversationMessage] = Field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class MessageContent(BaseModel):
    """Standardized message content for agent processing."""
    
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    
    text: Optional[str] = None
    file_path: Optional[str] = None
    file_data: Optional[bytes] = None  # Binary data for the file
    file_type: Optional[str] = None  # "image", "document", "audio", "voice"
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None


class APIResponse(BaseModel):
    """Standard response from Telegram API."""
    
    model_config = ConfigDict(extra='ignore')
    
    ok: bool
    result: Optional[Any] = None
    error_code: Optional[int] = None
    description: Optional[str] = None

