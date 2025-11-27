"""
Message handlers for different content types (text, voice, photos, documents).
All binaries are downloaded, processed, passed to agent, then deleted.
"""

import logging
from pathlib import Path
from typing import Optional

from pydantic_ai_telegram.models import TelegramMessage, MessageContent
from pydantic_ai_telegram.api import TelegramAPI
from pydantic_ai_telegram.binary_handler import BinaryHandler
from pydantic_ai_telegram.transcription.base import TranscriptionService

logger = logging.getLogger(__name__)


def get_media_type_from_mime(mime_type: Optional[str]) -> str:
    """
    Convert MIME type to media type for BinaryContent.
    
    Args:
        mime_type: MIME type
        
    Returns:
        Media type string (image/jpeg, image/png, etc.)
    """
    if not mime_type:
        return "application/octet-stream"
    return mime_type


class MessageHandler:
    """
    Base message handler with common functionality.
    """
    
    def __init__(
        self,
        api: TelegramAPI,
        binary_handler: BinaryHandler,
        transcription_service: Optional[TranscriptionService] = None,
    ) -> None:
        """
        Initialize message handler.
        
        Args:
            api: Telegram API client
            binary_handler: Binary file handler
            transcription_service: Optional transcription service for audio
        """
        self.api = api
        self.binary_handler = binary_handler
        self.transcription_service = transcription_service


class TextHandler(MessageHandler):
    """
    Handler for text messages.
    """
    
    async def handle(self, message: TelegramMessage) -> MessageContent:
        """
        Process text message.
        
        Args:
            message: Telegram message
            
        Returns:
            Message content for agent
        """
        text = message.text or message.caption or ""
        
        return MessageContent(
            text=text,
            file_type="text",
        )


class VoiceHandler(MessageHandler):
    """
    Handler for voice messages.
    Downloads, transcribes, then deletes the audio file.
    """
    
    async def handle(self, message: TelegramMessage) -> MessageContent:
        """
        Process voice message.
        
        Args:
            message: Telegram message with voice
            
        Returns:
            Message content with transcribed text
            
        Raises:
            ValueError: If transcription service is not available
        """
        if not message.voice:
            raise ValueError("Message does not contain voice")
        
        if not self.transcription_service:
            raise ValueError("Transcription service is not configured")
        
        temp_file: Optional[Path] = None
        
        try:
            # Get file info
            file_info = await self.api.get_file(message.voice.file_id)
            
            if not file_info.file_path:
                raise ValueError("File path not available")
            
            # Download audio file
            audio_data = await self.api.download_file(file_info.file_path)
            
            # Save to temporary file
            extension = self.binary_handler.get_file_extension(
                message.voice.mime_type,
                None,
            )
            temp_file = await self.binary_handler.save_file(
                audio_data,
                suffix=extension,
                prefix="voice_",
            )
            
            logger.info(f"Transcribing voice message from chat {message.chat.id}")
            
            # Transcribe
            transcribed_text = await self.transcription_service.transcribe(temp_file)
            
            # Add caption if present
            full_text = transcribed_text
            if message.caption:
                full_text = f"{message.caption}\n\n[Voice transcription]: {transcribed_text}"
            
            return MessageContent(
                text=full_text,
                file_type="voice",
            )
            
        finally:
            # Always clean up the temporary file
            if temp_file:
                await self.binary_handler.delete_file(temp_file)


class AudioHandler(MessageHandler):
    """
    Handler for audio files.
    Similar to voice handler but for audio file attachments.
    """
    
    async def handle(self, message: TelegramMessage) -> MessageContent:
        """
        Process audio message.
        
        Args:
            message: Telegram message with audio
            
        Returns:
            Message content with transcribed text
            
        Raises:
            ValueError: If transcription service is not available
        """
        if not message.audio:
            raise ValueError("Message does not contain audio")
        
        if not self.transcription_service:
            raise ValueError("Transcription service is not configured")
        
        temp_file: Optional[Path] = None
        
        try:
            # Get file info
            file_info = await self.api.get_file(message.audio.file_id)
            
            if not file_info.file_path:
                raise ValueError("File path not available")
            
            # Download audio file
            audio_data = await self.api.download_file(file_info.file_path)
            
            # Save to temporary file
            extension = self.binary_handler.get_file_extension(
                message.audio.mime_type,
                message.audio.file_name,
            )
            temp_file = await self.binary_handler.save_file(
                audio_data,
                suffix=extension,
                prefix="audio_",
            )
            
            logger.info(f"Transcribing audio file from chat {message.chat.id}")
            
            # Transcribe
            transcribed_text = await self.transcription_service.transcribe(temp_file)
            
            # Add caption and metadata if present
            parts = []
            if message.caption:
                parts.append(message.caption)
            
            if message.audio.title or message.audio.performer:
                metadata = []
                if message.audio.performer:
                    metadata.append(f"Artist: {message.audio.performer}")
                if message.audio.title:
                    metadata.append(f"Title: {message.audio.title}")
                parts.append(f"[Audio metadata]: {', '.join(metadata)}")
            
            parts.append(f"[Audio transcription]: {transcribed_text}")
            
            full_text = "\n\n".join(parts)
            
            return MessageContent(
                text=full_text,
                file_type="audio",
                original_filename=message.audio.file_name,
            )
            
        finally:
            # Always clean up the temporary file
            if temp_file:
                await self.binary_handler.delete_file(temp_file)


class PhotoHandler(MessageHandler):
    """
    Handler for photo messages.
    Downloads photo, passes to agent as binary content, then deletes.
    """
    
    async def handle(self, message: TelegramMessage) -> MessageContent:
        """
        Process photo message.
        
        Args:
            message: Telegram message with photo
            
        Returns:
            Message content with photo binary data
        """
        if not message.photo:
            raise ValueError("Message does not contain photo")
        
        # Get the largest photo size
        photo = max(message.photo, key=lambda p: p.file_size or 0)
        
        temp_file: Optional[Path] = None
        
        try:
            # Get file info
            file_info = await self.api.get_file(photo.file_id)
            
            if not file_info.file_path:
                raise ValueError("File path not available")
            
            # Download photo
            photo_data = await self.api.download_file(file_info.file_path)
            
            # Save to temporary file for backup/reference
            temp_file = await self.binary_handler.save_file(
                photo_data,
                suffix=".jpg",
                prefix="photo_",
            )
            
            logger.info(f"Received photo from chat {message.chat.id}")
            
            return MessageContent(
                text=message.caption,
                file_path=str(temp_file),
                file_data=photo_data,  # Include binary data
                file_type="image",
                mime_type="image/jpeg",
            )
            
        except Exception:
            # Clean up on error
            if temp_file:
                await self.binary_handler.delete_file(temp_file)
            raise


class DocumentHandler(MessageHandler):
    """
    Handler for document messages.
    Downloads document, passes to agent as binary content, then deletes.
    """
    
    async def handle(self, message: TelegramMessage) -> MessageContent:
        """
        Process document message.
        
        Args:
            message: Telegram message with document
            
        Returns:
            Message content with document binary data
        """
        if not message.document:
            raise ValueError("Message does not contain document")
        
        temp_file: Optional[Path] = None
        
        try:
            # Get file info
            file_info = await self.api.get_file(message.document.file_id)
            
            if not file_info.file_path:
                raise ValueError("File path not available")
            
            # Download document
            doc_data = await self.api.download_file(file_info.file_path)
            
            # Save to temporary file
            extension = self.binary_handler.get_file_extension(
                message.document.mime_type,
                message.document.file_name,
            )
            temp_file = await self.binary_handler.save_file(
                doc_data,
                suffix=extension,
                prefix="doc_",
            )
            
            logger.info(f"Received document from chat {message.chat.id}")
            
            return MessageContent(
                text=message.caption,
                file_path=str(temp_file),
                file_data=doc_data,  # Include binary data
                file_type="document",
                original_filename=message.document.file_name,
                mime_type=message.document.mime_type,
            )
            
        except Exception:
            # Clean up on error
            if temp_file:
                await self.binary_handler.delete_file(temp_file)
            raise

