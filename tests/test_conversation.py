"""
Tests for ConversationManager.
"""

import pytest

from pydantic_ai_telegram.conversation import ConversationManager


def test_conversation_manager_init():
    """Test ConversationManager initialization."""
    manager = ConversationManager(max_history=10)
    assert manager.max_history == 10


def test_get_or_create_conversation():
    """Test getting or creating conversation."""
    manager = ConversationManager()
    
    conv1 = manager.get_or_create_conversation(123)
    assert conv1.chat_id == 123
    
    # Should return same conversation
    conv2 = manager.get_or_create_conversation(123)
    assert conv1 is conv2


def test_add_message():
    """Test adding message to conversation."""
    manager = ConversationManager()
    
    msg = manager.add_message(123, "user", "Hello")
    
    assert msg.role == "user"
    assert msg.content == "Hello"
    
    history = manager.get_history(123)
    assert len(history) == 1
    assert history[0].content == "Hello"


def test_reset_conversation():
    """Test resetting conversation."""
    manager = ConversationManager()
    
    manager.add_message(123, "user", "Hello")
    manager.add_message(123, "assistant", "Hi there")
    
    assert manager.get_message_count(123) == 2
    
    manager.reset_conversation(123)
    
    assert manager.get_message_count(123) == 0


def test_token_counting():
    """Test token counting."""
    manager = ConversationManager()
    
    manager.add_message(123, "user", "Hello world")
    
    token_count = manager.get_token_count(123)
    assert token_count > 0


def test_max_history_limit():
    """Test that history is limited to max_history."""
    manager = ConversationManager(max_history=3)
    
    # Add 5 messages
    for i in range(5):
        manager.add_message(123, "user", f"Message {i}")
    
    history = manager.get_history(123)
    
    # Should only keep last 3
    assert len(history) == 3
    assert history[0].content == "Message 2"
    assert history[-1].content == "Message 4"


def test_conversation_summary():
    """Test getting conversation summary."""
    manager = ConversationManager()
    
    manager.add_message(123, "user", "Hello")
    manager.add_message(123, "assistant", "Hi")
    
    summary = manager.get_conversation_summary(123)
    
    assert summary["chat_id"] == 123
    assert summary["message_count"] == 2
    assert summary["total_tokens"] > 0


def test_list_active_conversations():
    """Test listing active conversations."""
    manager = ConversationManager()
    
    manager.add_message(123, "user", "Hello")
    manager.add_message(456, "user", "Hi")
    
    active = manager.list_active_conversations()
    
    assert 123 in active
    assert 456 in active
    assert len(active) == 2

