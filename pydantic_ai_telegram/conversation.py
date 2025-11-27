"""
Conversation history manager for tracking chat context and token usage.
"""

import logging
from datetime import datetime
from typing import Any, Optional
import tiktoken

from pydantic_ai_telegram.models import ConversationState, ConversationMessage

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history per chat/user with token counting.
    Stores conversations in memory by default, easily extensible for persistence.
    """
    
    def __init__(self, max_history: int = 50, encoding_name: str = "cl100k_base") -> None:
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep per conversation
            encoding_name: Tokenizer encoding to use for token counting
        """
        self.max_history = max_history
        self.conversations: dict[int, ConversationState] = {}
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}. Using character-based estimation.")
            self.encoding = None
    
    def get_or_create_conversation(self, chat_id: int) -> ConversationState:
        """
        Get existing conversation or create a new one.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            ConversationState for the chat
        """
        if chat_id not in self.conversations:
            self.conversations[chat_id] = ConversationState(chat_id=chat_id)
        
        return self.conversations[chat_id]
    
    def get_history(self, chat_id: int) -> list[ConversationMessage]:
        """
        Get conversation history for a chat.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            List of conversation messages
        """
        conversation = self.get_or_create_conversation(chat_id)
        return conversation.messages
    
    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str | list[Any],
        message_id: Optional[int] = None,
    ) -> ConversationMessage:
        """
        Add a message to the conversation history.
        
        Args:
            chat_id: Telegram chat ID
            role: Message role ("user" or "assistant")
            content: Message content (text or multimodal)
            message_id: Telegram message ID
            
        Returns:
            The created ConversationMessage
        """
        conversation = self.get_or_create_conversation(chat_id)
        
        # Calculate tokens for this message
        tokens = self._count_tokens(content)
        
        message = ConversationMessage(
            role=role,
            content=content,
            message_id=message_id,
            tokens=tokens,
            timestamp=datetime.now(),
        )
        
        conversation.messages.append(message)
        conversation.total_tokens += tokens
        conversation.last_updated = datetime.now()
        
        # Trim history if it exceeds max_history
        if len(conversation.messages) > self.max_history:
            removed_messages = conversation.messages[:len(conversation.messages) - self.max_history]
            conversation.messages = conversation.messages[-self.max_history:]
            
            # Adjust token count
            removed_tokens = sum(msg.tokens or 0 for msg in removed_messages)
            conversation.total_tokens -= removed_tokens
        
        return message
    
    def reset_conversation(self, chat_id: int) -> None:
        """
        Clear conversation history for a chat.
        
        Args:
            chat_id: Telegram chat ID
        """
        if chat_id in self.conversations:
            self.conversations[chat_id] = ConversationState(chat_id=chat_id)
            logger.info(f"Reset conversation for chat {chat_id}")
    
    def get_pydantic_history(self, chat_id: int) -> Optional[list[Any]]:
        """
        Get pydantic-ai formatted message history for a chat.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            List of pydantic-ai messages or None if no history
        """
        conversation = self.get_or_create_conversation(chat_id)
        
        # Return the raw pydantic-ai message history if available
        if hasattr(conversation, '_pydantic_history') and conversation._pydantic_history:
            return conversation._pydantic_history
        
        return None
    
    def set_pydantic_history(self, chat_id: int, messages: list[Any]) -> None:
        """
        Store pydantic-ai formatted message history from result.all_messages().
        
        This preserves the complete conversation including system prompts,
        user messages, assistant responses, and tool calls.
        
        Automatically limits history to max_history most recent messages
        (excluding system prompt which is always kept).
        
        Args:
            chat_id: Telegram chat ID
            messages: List of pydantic-ai messages from result.all_messages()
        """
        conversation = self.get_or_create_conversation(chat_id)
        
        # Limit history depth if needed
        # Keep system prompt (first message) and limit the rest
        if len(messages) > self.max_history + 1:  # +1 for system prompt
            # Check if first message is system prompt
            system_prompt = None
            other_messages = messages
            
            if messages and hasattr(messages[0], 'kind') and messages[0].kind == 'request':
                # First message might be system prompt
                if hasattr(messages[0], 'parts') and messages[0].parts:
                    first_part = messages[0].parts[0]
                    if hasattr(first_part, 'part_kind') and first_part.part_kind == 'system-prompt':
                        system_prompt = messages[0]
                        other_messages = messages[1:]
            
            # Keep only the most recent messages
            if system_prompt:
                limited_messages = [system_prompt] + other_messages[-(self.max_history):]
            else:
                limited_messages = other_messages[-(self.max_history):]
            
            logger.info(f"Limited history for chat {chat_id} from {len(messages)} to {len(limited_messages)} messages")
            messages = limited_messages
        
        # Store the pydantic-ai message history
        conversation._pydantic_history = messages  # type: ignore
        conversation.last_updated = datetime.now()
        
        # Update statistics
        conversation.total_tokens = 0
        message_count = 0
        
        for msg in messages:
            message_count += 1
            # Estimate tokens for statistics
            try:
                # Try to get content from the message
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            content = str(part.content)
                            tokens = self._count_text_tokens(content)
                            conversation.total_tokens += tokens
                elif hasattr(msg, 'content'):
                    content = str(msg.content)
                    tokens = self._count_text_tokens(content)
                    conversation.total_tokens += tokens
            except Exception as e:
                logger.debug(f"Could not estimate tokens for message: {e}")
                pass
        
        # Store a simple message count for display
        conversation.messages = []  # We don't need the old format anymore
        logger.debug(f"Updated history for chat {chat_id}: {message_count} messages, ~{conversation.total_tokens} tokens")
    
    def get_token_count(self, chat_id: int) -> int:
        """
        Get the total token count for a conversation.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Total number of tokens in the conversation
        """
        conversation = self.get_or_create_conversation(chat_id)
        return conversation.total_tokens
    
    def get_message_count(self, chat_id: int) -> int:
        """
        Get the total message count for a conversation.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Number of messages in the conversation
        """
        conversation = self.get_or_create_conversation(chat_id)
        
        # If we have pydantic history, count those messages
        if hasattr(conversation, '_pydantic_history') and conversation._pydantic_history:
            return len(conversation._pydantic_history)
        
        return len(conversation.messages)
    
    def set_max_history(self, max_history: int) -> None:
        """
        Update the maximum history size.
        
        Args:
            max_history: New maximum number of messages to keep
        """
        self.max_history = max_history
    
    def _count_tokens(self, content: str | list[Any]) -> int:
        """
        Count tokens in content.
        
        Args:
            content: Text content or multimodal content list
            
        Returns:
            Estimated token count
        """
        if isinstance(content, str):
            return self._count_text_tokens(content)
        elif isinstance(content, list):
            # For multimodal content, count all text parts
            total = 0
            for item in content:
                if isinstance(item, str):
                    total += self._count_text_tokens(item)
                elif isinstance(item, dict) and "text" in item:
                    total += self._count_text_tokens(item["text"])
                # Images/files typically count as ~85-170 tokens each
                # This is a rough estimate
                elif isinstance(item, dict) and ("image" in item or "file" in item):
                    total += 100
            return total
        
        return 0
    
    def _count_text_tokens(self, text: str) -> int:
        """
        Count tokens in text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using estimation.")
        
        # Fallback to character-based estimation
        # Rough estimate: ~4 characters per token on average
        return len(text) // 4
    
    def get_conversation_summary(self, chat_id: int) -> dict[str, Any]:
        """
        Get a summary of the conversation state.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Dictionary with conversation statistics
        """
        conversation = self.get_or_create_conversation(chat_id)
        
        return {
            "chat_id": chat_id,
            "message_count": len(conversation.messages),
            "total_tokens": conversation.total_tokens,
            "created_at": conversation.created_at.isoformat(),
            "last_updated": conversation.last_updated.isoformat(),
        }
    
    def list_active_conversations(self) -> list[int]:
        """
        Get list of all active chat IDs.
        
        Returns:
            List of chat IDs with active conversations
        """
        return list(self.conversations.keys())

