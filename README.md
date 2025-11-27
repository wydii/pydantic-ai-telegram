# Pydantic AI Telegram

Wrap any [Pydantic AI](https://ai.pydantic.dev/) agent with Telegram in one line of code.

**Why?** Pydantic AI makes building AI agents simple. This library lets you expose them via Telegram instantly — perfect for personal assistants, prototypes, or internal tools.

## Installation

```bash
pip install pydantic-ai-telegram
```

With local voice transcription (requires ffmpeg):

```bash
pip install pydantic-ai-telegram[whisper]
```

## Quick Start

### Option 1: CLI Configuration

Run the interactive setup:

```bash
pydantic-ai-telegram-config
```

This creates a `.env` file with your configuration.

### Option 2: Manual Configuration

Create a `.env` file:

```bash
# Required
TELEGRAM_BOT_TOKEN=your_token_from_botfather

# Optional - Access control
TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321
TELEGRAM_ALLOWED_USERNAMES=alice,bob

# Optional - Voice transcription (local whisper)
TRANSCRIPTION_SERVICE=local
WHISPER_MODEL=turbo

# Optional - OpenAI (for agent or transcription)
OPENAI_API_KEY=sk-...

# Optional - Conversation history limit
MAX_HISTORY_MESSAGES=50
```

## Minimal Example

```python
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai_telegram import TelegramAgent

load_dotenv()

# Your Pydantic AI agent
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

# Wrap it with Telegram
bot = TelegramAgent(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    agent=agent,
)

bot.run()
```

That's it. Your agent is now accessible via Telegram.

## Voice Transcription

Local transcription uses [OpenAI Whisper](https://github.com/openai/whisper) running on your machine:

```python
from pydantic_ai_telegram import TelegramAgent, LocalWhisperTranscription

transcription = LocalWhisperTranscription(model_name="turbo")

bot = TelegramAgent(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    agent=agent,
    transcription_service=transcription,
)
```

**Note:** Cloud transcription (OpenAI API) is planned for a future release.

## License

MIT
