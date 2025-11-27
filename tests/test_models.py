"""
Tests for Pydantic models.
"""

import pytest
from datetime import datetime

from pydantic_ai_telegram.models import (
    TelegramUser,
    TelegramChat,
    TelegramMessage,
    TelegramUpdate,
    BotConfig,
    ConversationMessage,
    ConversationState,
    MessageContent,
)


def test_telegram_user():
    """Test TelegramUser model."""
    user = TelegramUser(
        id=123456,
        is_bot=False,
        first_name="John",
        last_name="Doe",
        username="johndoe",
    )
    
    assert user.id == 123456
    assert user.first_name == "John"
    assert user.username == "johndoe"


def test_telegram_chat():
    """Test TelegramChat model."""
    chat = TelegramChat(
        id=987654,
        type="private",
        first_name="Jane",
    )
    
    assert chat.id == 987654
    assert chat.type == "private"


def test_telegram_message():
    """Test TelegramMessage model."""
    message_data = {
        "message_id": 1,
        "date": 1234567890,
        "chat": {
            "id": 123,
            "type": "private",
        },
        "from": {
            "id": 456,
            "is_bot": False,
            "first_name": "User",
        },
        "text": "Hello!",
    }
    
    message = TelegramMessage(**message_data)
    
    assert message.message_id == 1
    assert message.text == "Hello!"
    assert message.chat.id == 123


def test_telegram_update():
    """Test TelegramUpdate model."""
    update_data = {
        "update_id": 100,
        "message": {
            "message_id": 1,
            "date": 1234567890,
            "chat": {"id": 123, "type": "private"},
            "text": "Test",
        },
    }
    
    update = TelegramUpdate(**update_data)
    
    assert update.update_id == 100
    assert update.message is not None
    assert update.message.text == "Test"


def test_bot_config():
    """Test BotConfig model."""
    config = BotConfig(
        bot_token="123:ABC",
        allowed_chat_ids=[123, 456],
        max_history_messages=30,
    )
    
    assert config.bot_token == "123:ABC"
    assert 123 in config.allowed_chat_ids


def test_conversation_message():
    """Test ConversationMessage model."""
    msg = ConversationMessage(
        role="user",
        content="Hello",
        message_id=1,
    )
    
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert isinstance(msg.timestamp, datetime)


def test_conversation_state():
    """Test ConversationState model."""
    state = ConversationState(chat_id=123)
    
    assert state.chat_id == 123
    assert len(state.messages) == 0
    assert state.total_tokens == 0


def test_message_content():
    """Test MessageContent model."""
    content = MessageContent(
        text="Hello",
        file_type="text",
    )
    
    assert content.text == "Hello"
    assert content.file_type == "text"
    
    # Test with file
    content_with_file = MessageContent(
        text="Check this",
        file_path="/tmp/test.jpg",
        file_type="image",
    )
    
    assert content_with_file.file_path == "/tmp/test.jpg"

