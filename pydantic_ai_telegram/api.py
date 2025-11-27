"""
Telegram Bot API client using native HTTP requests via httpx.
No external Telegram bot libraries used.
"""

import logging
from typing import Any, Optional
import httpx
from pydantic_ai_telegram.models import (
    TelegramUpdate,
    TelegramFile,
    APIResponse,
)

logger = logging.getLogger(__name__)


class TelegramAPIError(Exception):
    """Exception raised for Telegram API errors."""
    
    def __init__(self, error_code: int, description: str) -> None:
        self.error_code = error_code
        self.description = description
        super().__init__(f"Telegram API Error {error_code}: {description}")


# Telegram API limits
MAX_MESSAGE_LENGTH = 4096


class TelegramAPI:
    """
    HTTP client for Telegram Bot API.
    Implements native HTTP calls without using telegram bot libraries.
    """
    
    def __init__(self, bot_token: str, timeout: int = 30) -> None:
        """
        Initialize the Telegram API client.
        
        Args:
            bot_token: Telegram bot token from BotFather
            timeout: Default timeout for HTTP requests in seconds
        """
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self) -> 'TelegramAPI':
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)
        return self.client
    
    async def _request(
        self,
        method: str,
        data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Make a request to Telegram Bot API.
        
        Args:
            method: API method name
            data: Request data
            files: Files to upload
            timeout: Request timeout (overrides default)
            
        Returns:
            API response as dictionary
            
        Raises:
            TelegramAPIError: If API returns an error
        """
        client = await self._ensure_client()
        url = f"{self.base_url}/{method}"
        
        try:
            if files:
                response = await client.post(
                    url,
                    data=data or {},
                    files=files,
                    timeout=timeout or self.timeout,
                )
            else:
                response = await client.post(
                    url,
                    json=data or {},
                    timeout=timeout or self.timeout,
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response using Pydantic model
            api_response = APIResponse(**result)
            
            if not api_response.ok:
                raise TelegramAPIError(
                    api_response.error_code or 0,
                    api_response.description or "Unknown error"
                )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling {method}: {e}")
            raise TelegramAPIError(e.response.status_code, str(e))
        except httpx.RequestError as e:
            logger.error(f"Request error calling {method}: {e}")
            raise TelegramAPIError(0, str(e))
    
    async def get_updates(
        self,
        offset: Optional[int] = None,
        limit: int = 100,
        timeout: int = 30,
    ) -> list[TelegramUpdate]:
        """
        Get incoming updates using long polling.
        
        Args:
            offset: Identifier of the first update to be returned
            limit: Maximum number of updates to retrieve
            timeout: Timeout for long polling
            
        Returns:
            List of TelegramUpdate objects
        """
        data: dict[str, Any] = {
            "limit": limit,
            "timeout": timeout,
        }
        
        if offset is not None:
            data["offset"] = offset
        
        result = await self._request("getUpdates", data=data, timeout=timeout + 5)
        
        updates = []
        for update_data in result.get("result", []):
            try:
                updates.append(TelegramUpdate(**update_data))
            except Exception as e:
                logger.error(f"Failed to parse update: {e}")
                continue
        
        return updates
    
    def _split_message(self, text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
        """
        Split a long message into chunks that fit within Telegram's limits.
        Tries to split at natural boundaries (newlines, then spaces) when possible.
        
        Args:
            text: The text to split
            max_length: Maximum length per chunk (default: 4096)
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks: list[str] = []
        remaining = text
        
        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break
            
            # Find the best split point
            chunk = remaining[:max_length]
            
            # Try to split at double newline (paragraph)
            split_pos = chunk.rfind('\n\n')
            if split_pos > max_length // 2:
                chunks.append(remaining[:split_pos].rstrip())
                remaining = remaining[split_pos:].lstrip()
                continue
            
            # Try to split at single newline
            split_pos = chunk.rfind('\n')
            if split_pos > max_length // 2:
                chunks.append(remaining[:split_pos].rstrip())
                remaining = remaining[split_pos:].lstrip()
                continue
            
            # Try to split at space
            split_pos = chunk.rfind(' ')
            if split_pos > max_length // 2:
                chunks.append(remaining[:split_pos].rstrip())
                remaining = remaining[split_pos:].lstrip()
                continue
            
            # Force split at max_length (no good boundary found)
            chunks.append(remaining[:max_length])
            remaining = remaining[max_length:]
        
        return chunks

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        parse_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send a text message. Automatically splits long messages into chunks.
        
        Args:
            chat_id: Unique identifier for the target chat
            text: Text of the message to be sent
            reply_to_message_id: If specified, the first message is sent as a reply
            parse_mode: Mode for parsing entities (Markdown, HTML)
            
        Returns:
            API response (from the last message sent)
        """
        # Split message if it exceeds Telegram's limit
        chunks = self._split_message(text)
        
        if len(chunks) > 1:
            logger.debug(f"Splitting long message ({len(text)} chars) into {len(chunks)} chunks")
        
        result: dict[str, Any] = {}
        
        for i, chunk in enumerate(chunks):
            data: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
            }
            
            # Only reply to the original message for the first chunk
            if reply_to_message_id and i == 0:
                data["reply_to_message_id"] = reply_to_message_id
            
            if parse_mode:
                data["parse_mode"] = parse_mode
            
            result = await self._request("sendMessage", data=data)
        
        return result
    
    async def send_chat_action(
        self,
        chat_id: int,
        action: str = "typing",
    ) -> dict[str, Any]:
        """
        Send chat action (e.g., typing indicator).
        
        Args:
            chat_id: Unique identifier for the target chat
            action: Type of action (typing, upload_photo, upload_document, etc.)
            
        Returns:
            API response
        """
        data = {
            "chat_id": chat_id,
            "action": action,
        }
        
        return await self._request("sendChatAction", data=data)
    
    async def get_file(self, file_id: str) -> TelegramFile:
        """
        Get basic info about a file and prepare it for downloading.
        
        Args:
            file_id: File identifier
            
        Returns:
            TelegramFile object
        """
        data = {"file_id": file_id}
        result = await self._request("getFile", data=data)
        
        return TelegramFile(**result["result"])
    
    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from Telegram servers.
        
        Args:
            file_path: File path returned by get_file
            
        Returns:
            File content as bytes
        """
        client = await self._ensure_client()
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            logger.error(f"Error downloading file: {e}")
            raise TelegramAPIError(0, f"Failed to download file: {e}")
    
    async def get_me(self) -> dict[str, Any]:
        """
        Get basic information about the bot.
        
        Returns:
            Bot information
        """
        return await self._request("getMe")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

