"""
Main TelegramAgent class that wraps Pydantic AI agents with Telegram bot capabilities.
"""

import asyncio
import logging
from typing import Any, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from pydantic_ai_telegram.api import TelegramAPI, TelegramAPIError
from pydantic_ai_telegram.models import TelegramMessage, TelegramUpdate, MessageContent
from pydantic_ai_telegram.conversation import ConversationManager
from pydantic_ai_telegram.binary_handler import BinaryHandler
from pydantic_ai_telegram.handlers import (
    TextHandler,
    VoiceHandler,
    AudioHandler,
    PhotoHandler,
    DocumentHandler,
    get_media_type_from_mime,
)
from pydantic_ai_telegram.transcription.base import TranscriptionService

logger = logging.getLogger(__name__)


class TelegramAgent:
    """
    Wraps a Pydantic AI agent with Telegram bot capabilities.
    Handles multimodal interactions including text, voice, images, and documents.
    """
    
    def __init__(
        self,
        bot_token: str,
        agent: Agent[Any, Any],
        transcription_service: Optional[TranscriptionService | str] = None,
        whisper_model: Optional[str] = None,
        allowed_chat_ids: Optional[list[int] | str] = None,
        allowed_usernames: Optional[list[str] | str] = None,
        max_history: Optional[int | str] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the Telegram bot agent.
        
        Args:
            bot_token: Telegram bot token from BotFather
            agent: Pydantic AI agent to wrap
            transcription_service: TranscriptionService instance, or "local"/"openai"/"none" string
            whisper_model: Whisper model name (if using local transcription)
            allowed_chat_ids: List of allowed chat IDs, comma-separated string, or None (allow all)
            allowed_usernames: List of allowed usernames, comma-separated string, or None (allow all)
            max_history: Maximum conversation history to maintain (default: 50)
            temp_dir: Directory for temporary files
        """
        self.bot_token = bot_token
        self.agent = agent
        
        # Parse and setup transcription service
        self.transcription_service = self._setup_transcription(
            transcription_service, whisper_model
        )
        
        # Parse allowed_chat_ids
        self.allowed_chat_ids = self._parse_chat_ids(allowed_chat_ids)
        
        # Parse allowed_usernames
        self.allowed_usernames = self._parse_usernames(allowed_usernames)
        
        # Parse max_history
        if max_history is None:
            max_history = 50
        elif isinstance(max_history, str):
            try:
                max_history = int(max_history)
            except ValueError:
                logger.warning(f"Invalid max_history value: {max_history}, using default 50")
                max_history = 50
        
        # Initialize components
        self.api = TelegramAPI(bot_token)
        self.conversation_manager = ConversationManager(max_history=max_history)
        self.binary_handler = BinaryHandler(temp_dir=temp_dir)
        
        # Initialize message handlers
        self.text_handler = TextHandler(self.api, self.binary_handler)
        self.voice_handler = VoiceHandler(
            self.api,
            self.binary_handler,
            transcription_service,
        )
        self.audio_handler = AudioHandler(
            self.api,
            self.binary_handler,
            transcription_service,
        )
        self.photo_handler = PhotoHandler(self.api, self.binary_handler)
        self.document_handler = DocumentHandler(self.api, self.binary_handler)
        
        # State
        self.running = False
        self.last_update_id = 0
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task[None]] = None
    
    def _parse_chat_ids(self, chat_ids: Optional[list[int] | str]) -> Optional[list[int]]:
        """Parse chat IDs from string or list."""
        if chat_ids is None or chat_ids == "":
            return None
        
        if isinstance(chat_ids, list):
            return chat_ids
        
        # Parse comma-separated string
        result = []
        for item in str(chat_ids).split(','):
            item = item.strip()
            if item and item.isdigit():
                result.append(int(item))
        
        return result if result else None
    
    def _parse_usernames(self, usernames: Optional[list[str] | str]) -> Optional[list[str]]:
        """Parse usernames from string or list."""
        if usernames is None or usernames == "":
            return None
        
        if isinstance(usernames, list):
            return usernames
        
        # Parse comma-separated string
        result = []
        for item in str(usernames).split(','):
            item = item.strip()
            if item:
                # Remove @ if present
                if item.startswith('@'):
                    item = item[1:]
                result.append(item)
        
        return result if result else None
    
    def _setup_transcription(
        self,
        transcription_service: Optional[TranscriptionService | str],
        whisper_model: Optional[str],
    ) -> Optional[TranscriptionService]:
        """
        Setup transcription service from various input types.
        
        Args:
            transcription_service: Service instance or string ("local", "openai", "none")
            whisper_model: Whisper model name for local transcription
            
        Returns:
            Configured TranscriptionService or None
        """
        # If already a TranscriptionService instance, return it
        if isinstance(transcription_service, TranscriptionService):
            return transcription_service
        
        # If None or "none", no transcription
        if transcription_service is None or transcription_service == "none":
            logger.info("Voice transcription disabled")
            return None
        
        # Setup based on string
        service_type = str(transcription_service).lower()
        
        if service_type == "local":
            try:
                from pydantic_ai_telegram.transcription import (
                    LocalWhisperTranscription,
                    check_ffmpeg_installed,
                )
                
                # Check ffmpeg (with helpful error)
                if not check_ffmpeg_installed():
                    logger.warning(
                        "⚠️  ffmpeg is not installed. Voice transcription will not work.\n"
                        "   Install: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Ubuntu)"
                    )
                    return None
                
                model = whisper_model or "turbo"
                logger.info(f"Setting up local Whisper transcription (model: {model})")
                
                return LocalWhisperTranscription(model_name=model)
                
            except ImportError as e:
                logger.warning(
                    f"Could not setup local Whisper: {e}\n"
                    "Install with: pip install pydantic-ai-telegram[whisper]"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to setup local Whisper: {e}")
                return None
        
        elif service_type == "openai":
            try:
                from pydantic_ai_telegram.transcription import OpenAITranscription
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set, cannot use OpenAI transcription")
                    return None
                
                logger.info("Setting up OpenAI transcription")
                return OpenAITranscription(api_key=api_key)
                
            except ImportError:
                logger.warning(
                    "Could not import OpenAI transcription.\n"
                    "Install with: pip install pydantic-ai-telegram[openai]"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to setup OpenAI transcription: {e}")
                return None
        
        else:
            logger.warning(f"Unknown transcription service: {service_type}")
            return None
    
    def is_authorized(self, message: TelegramMessage) -> bool:
        """
        Check if user is authorized to use the bot.
        
        Args:
            message: Telegram message
            
        Returns:
            True if authorized, False otherwise
        """
        # Check chat ID
        if self.allowed_chat_ids and message.chat.id not in self.allowed_chat_ids:
            return False
        
        # Check username
        if self.allowed_usernames and message.from_user:
            username = message.from_user.username
            if not username or username not in self.allowed_usernames:
                return False
        
        return True
    
    async def handle_command(self, command: str, chat_id: int, message_id: int) -> None:
        """
        Handle bot commands.
        
        Args:
            command: Command text (without /)
            chat_id: Chat ID
            message_id: Message ID to reply to
        """
        if command == "start":
            response = (
                "👋 Hello! I'm your AI assistant.\n\n"
                "You can send me:\n"
                "• Text messages\n"
                "• Voice messages (will be transcribed)\n"
                "• Images\n"
                "• Documents\n\n"
                "Commands:\n"
                "/reset - Clear conversation history\n"
                "/tokens - Show token count\n"
                "/help - Show this message"
            )
            await self.api.send_message(chat_id, response, reply_to_message_id=message_id)
        
        elif command == "help":
            response = (
                "Available commands:\n"
                "/start - Welcome message\n"
                "/reset - Clear conversation history\n"
                "/tokens - Show current token count\n"
                "/help - Show this message\n\n"
                "Send me any message and I'll respond!"
            )
            await self.api.send_message(chat_id, response, reply_to_message_id=message_id)
        
        elif command == "reset":
            self.conversation_manager.reset_conversation(chat_id)
            response = "✅ Conversation history cleared!"
            await self.api.send_message(chat_id, response, reply_to_message_id=message_id)
        
        elif command == "tokens":
            token_count = self.conversation_manager.get_token_count(chat_id)
            message_count = self.conversation_manager.get_message_count(chat_id)
            response = (
                f"📊 Conversation Statistics:\n"
                f"• Messages: {message_count}\n"
                f"• Tokens: {token_count}"
            )
            await self.api.send_message(chat_id, response, reply_to_message_id=message_id)
        
        else:
            response = f"Unknown command: /{command}\nUse /help to see available commands."
            await self.api.send_message(chat_id, response, reply_to_message_id=message_id)
    
    async def process_message(self, message: TelegramMessage) -> None:
        """
        Process an incoming message.
        
        Args:
            message: Telegram message to process
        """
        # Check authorization
        if not self.is_authorized(message):
            logger.warning(f"Unauthorized access attempt from chat {message.chat.id}")
            await self.api.send_message(
                message.chat.id,
                "⛔ You are not authorized to use this bot.",
                reply_to_message_id=message.message_id,
            )
            return
        
        # Handle commands
        if message.text and message.text.startswith("/"):
            command = message.text[1:].split()[0].lower()
            await self.handle_command(command, message.chat.id, message.message_id)
            return
        
        # Show typing indicator
        await self.api.send_chat_action(message.chat.id, "typing")
        
        try:
            # Process message based on type
            message_content = await self.route_message(message)
            
            # Get agent response
            response_text = await self.get_agent_response(
                message.chat.id,
                message_content,
            )
            
            # Send response
            await self.api.send_message(
                message.chat.id,
                response_text,
                reply_to_message_id=message.message_id,
            )
            
            # Clean up any temporary files
            if message_content.file_path:
                await self.binary_handler.delete_file(message_content.file_path)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error_msg = "Sorry, I encountered an error processing your message."
            await self.api.send_message(
                message.chat.id,
                error_msg,
                reply_to_message_id=message.message_id,
            )
    
    async def route_message(self, message: TelegramMessage) -> MessageContent:
        """
        Route message to appropriate handler.
        
        Args:
            message: Telegram message
            
        Returns:
            Processed message content
        """
        if message.voice:
            return await self.voice_handler.handle(message)
        elif message.audio:
            return await self.audio_handler.handle(message)
        elif message.photo:
            return await self.photo_handler.handle(message)
        elif message.document:
            return await self.document_handler.handle(message)
        else:
            return await self.text_handler.handle(message)
    
    async def get_agent_response(
        self,
        chat_id: int,
        message_content: MessageContent,
    ) -> str:
        """
        Get response from Pydantic AI agent.
        
        Args:
            chat_id: Chat ID
            message_content: Processed message content
            
        Returns:
            Agent response text
        """
        # Get conversation history (pydantic-ai message format)
        # This is the complete history from result.all_messages()
        history = self.conversation_manager.get_pydantic_history(chat_id)
        
        # Prepare message for agent (can include text + BinaryContent)
        user_message_parts = self._prepare_agent_message(message_content)
        
        # Run agent with history
        try:
            # Run agent with message history
            # Important: Pass the complete message_history from previous run
            # This maintains context across the conversation
            if history:
                logger.debug(f"Running agent with {len(history)} messages in history")
                result = await self.agent.run(user_message_parts, message_history=history)
            else:
                logger.debug("Running agent without history (first message)")
                result = await self.agent.run(user_message_parts)
            
            response_text = str(result.output)
            
            # Store the complete message history from pydantic-ai
            # This includes system prompt, all user messages, all assistant responses
            # Use result.all_messages() to get the complete history
            all_messages = result.all_messages()
            self.conversation_manager.set_pydantic_history(chat_id, all_messages)
            
            logger.debug(f"Stored {len(all_messages)} messages in history for chat {chat_id}")
            
            return response_text
        
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            raise
    
    def _prepare_agent_message(self, message_content: MessageContent) -> str | list[str | BinaryContent]:
        """
        Prepare message content for the agent.
        
        Args:
            message_content: Message content
            
        Returns:
            Message for agent (text or list with text + BinaryContent)
        """
        # If we have binary data (image, document), create BinaryContent
        if message_content.file_data:
            parts: list[str | BinaryContent] = []
            
            # Add text if present
            if message_content.text:
                parts.append(message_content.text)
            
            # Add binary content
            media_type = get_media_type_from_mime(message_content.mime_type)
            
            binary_content = BinaryContent(
                data=message_content.file_data,
                media_type=media_type,
            )
            parts.append(binary_content)
            
            return parts
        
        # Just text
        return message_content.text or ""
    
    async def process_update(self, update: TelegramUpdate) -> None:
        """
        Process a Telegram update.
        
        Args:
            update: Telegram update
        """
        # Get the message from the update
        message = update.message or update.edited_message
        
        if message:
            await self.process_message(message)
    
    async def _polling_loop(self) -> None:
        """Main polling loop for receiving updates."""
        logger.info("Starting polling loop...")
        
        while self.running:
            try:
                updates = await self.api.get_updates(
                    offset=self.last_update_id + 1,
                    timeout=30,
                )
                
                for update in updates:
                    self.last_update_id = update.update_id
                    await self.process_update(update)
            
            except TelegramAPIError as e:
                logger.error(f"Telegram API error: {e}")
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Unexpected error in polling loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("Polling loop stopped")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of temporary files."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.binary_handler.cleanup_old_files(max_age_seconds=3600)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def start(self) -> None:
        """
        Start the bot (async version).
        Runs until stopped.
        """
        if self.running:
            logger.warning("Bot is already running")
            return
        
        self.running = True
        
        # Get bot info
        try:
            bot_info = await self.api.get_me()
            logger.info(f"Bot started: @{bot_info['result']['username']}")
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        try:
            await self._polling_loop()
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the bot and clean up resources."""
        if not self.running:
            return
        
        logger.info("Stopping bot...")
        self.running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close API client
        await self.api.close()
        
        # Close transcription service
        if self.transcription_service:
            await self.transcription_service.close()
        
        logger.info("Bot stopped")
    
    def run(self) -> None:
        """
        Start the bot (synchronous version).
        Blocks until stopped.
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")

