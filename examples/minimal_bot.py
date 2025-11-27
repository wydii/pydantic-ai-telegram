"""
Minimal bot example - wrap any Pydantic AI agent with Telegram.
"""

import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai_telegram import TelegramAgent

load_dotenv()

# Create your Pydantic AI agent
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
