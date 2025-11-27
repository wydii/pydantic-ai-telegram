"""
Tests for TelegramAPI.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from pydantic_ai_telegram.api import TelegramAPI, TelegramAPIError


@pytest.fixture
def api():
    """Create a TelegramAPI instance."""
    return TelegramAPI(bot_token="test_token")


def test_api_init(api):
    """Test TelegramAPI initialization."""
    assert api.bot_token == "test_token"
    assert api.base_url == "https://api.telegram.org/bottest_token"
    assert api.timeout == 30


def test_api_init_custom_timeout():
    """Test TelegramAPI with custom timeout."""
    api = TelegramAPI(bot_token="test", timeout=60)
    assert api.timeout == 60


@pytest.mark.asyncio
async def test_ensure_client(api):
    """Test client initialization."""
    client = await api._ensure_client()
    assert client is not None
    assert api.client is client
    await api.close()


@pytest.mark.asyncio
async def test_close(api):
    """Test closing the API client."""
    await api._ensure_client()
    assert api.client is not None
    
    await api.close()
    assert api.client is None


@pytest.mark.asyncio
async def test_request_success(api):
    """Test successful API request."""
    mock_response = Mock()
    mock_response.json.return_value = {"ok": True, "result": {"id": 123}}
    mock_response.raise_for_status = Mock()
    
    with patch.object(api, '_ensure_client') as mock_client:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_http
        
        result = await api._request("getMe")
        
        assert result["ok"] is True
        assert result["result"]["id"] == 123


@pytest.mark.asyncio
async def test_request_api_error(api):
    """Test API error handling."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "ok": False,
        "error_code": 401,
        "description": "Unauthorized"
    }
    mock_response.raise_for_status = Mock()
    
    with patch.object(api, '_ensure_client') as mock_client:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_http
        
        with pytest.raises(TelegramAPIError) as exc_info:
            await api._request("getMe")
        
        assert exc_info.value.error_code == 401
        assert "Unauthorized" in str(exc_info.value)


def test_telegram_api_error():
    """Test TelegramAPIError."""
    error = TelegramAPIError(404, "Not found")
    assert error.error_code == 404
    assert error.description == "Not found"
    assert "404" in str(error)
    assert "Not found" in str(error)

