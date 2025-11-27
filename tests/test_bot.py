"""
Tests for TelegramAgent.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from pydantic_ai_telegram.bot import TelegramAgent
from pydantic_ai_telegram.models import TelegramMessage, TelegramChat, TelegramUser


@pytest.fixture
def mock_agent():
    """Create a mock Pydantic AI agent."""
    agent = Mock()
    agent.run = AsyncMock(return_value=Mock(output="Test response", all_messages=Mock(return_value=[])))
    return agent


@pytest.fixture
def telegram_agent(mock_agent):
    """Create a TelegramAgent instance with mocked dependencies."""
    return TelegramAgent(
        bot_token="test_token",
        agent=mock_agent,
        transcription_service=None,
    )


def test_telegram_agent_init(mock_agent):
    """Test TelegramAgent initialization."""
    agent = TelegramAgent(
        bot_token="test_token",
        agent=mock_agent,
    )
    
    assert agent.bot_token == "test_token"
    assert agent.agent == mock_agent
    assert agent.running is False


def test_parse_chat_ids():
    """Test parsing of chat IDs."""
    agent = Mock()
    
    bot = TelegramAgent(bot_token="test", agent=agent, allowed_chat_ids="123,456,789")
    assert bot.allowed_chat_ids == [123, 456, 789]
    
    bot = TelegramAgent(bot_token="test", agent=agent, allowed_chat_ids=[111, 222])
    assert bot.allowed_chat_ids == [111, 222]
    
    bot = TelegramAgent(bot_token="test", agent=agent, allowed_chat_ids=None)
    assert bot.allowed_chat_ids is None


def test_parse_usernames():
    """Test parsing of usernames."""
    agent = Mock()
    
    bot = TelegramAgent(bot_token="test", agent=agent, allowed_usernames="alice,bob")
    assert bot.allowed_usernames == ["alice", "bob"]
    
    bot = TelegramAgent(bot_token="test", agent=agent, allowed_usernames="@alice,@bob")
    assert bot.allowed_usernames == ["alice", "bob"]


def test_is_authorized_no_restrictions(telegram_agent):
    """Test authorization with no restrictions."""
    message = TelegramMessage(
        message_id=1,
        date=1234567890,
        chat=TelegramChat(id=123, type="private"),
        text="Hello",
    )
    
    assert telegram_agent.is_authorized(message) is True


def test_is_authorized_with_chat_id_restriction(mock_agent):
    """Test authorization with chat ID restriction."""
    bot = TelegramAgent(
        bot_token="test",
        agent=mock_agent,
        allowed_chat_ids=[123],
    )
    
    allowed_message = TelegramMessage(
        message_id=1,
        date=1234567890,
        chat=TelegramChat(id=123, type="private"),
        text="Hello",
    )
    
    denied_message = TelegramMessage(
        message_id=2,
        date=1234567890,
        chat=TelegramChat(id=999, type="private"),
        text="Hello",
    )
    
    assert bot.is_authorized(allowed_message) is True
    assert bot.is_authorized(denied_message) is False


def test_is_authorized_with_username_restriction(mock_agent):
    """Test authorization with username restriction."""
    bot = TelegramAgent(
        bot_token="test",
        agent=mock_agent,
        allowed_usernames=["alice"],
    )
    
    # Use 'from' alias as Pydantic expects
    allowed_message = TelegramMessage.model_validate({
        "message_id": 1,
        "date": 1234567890,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "Alice", "username": "alice"},
        "text": "Hello",
    })
    
    denied_message = TelegramMessage.model_validate({
        "message_id": 2,
        "date": 1234567890,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 2, "is_bot": False, "first_name": "Bob", "username": "bob"},
        "text": "Hello",
    })
    
    assert bot.is_authorized(allowed_message) is True
    assert bot.is_authorized(denied_message) is False

