"""Application settings loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    api_sports_key: str
    api_sports_host: str
    nesine_url: str
    iddaa_url: str
    telegram_bot_token: str
    telegram_chat_id: str
    http_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            api_sports_key=os.getenv("API_SPORTS_KEY", "").strip(),
            api_sports_host=os.getenv("API_SPORTS_HOST", "v3.football.api-sports.io").strip(),
            nesine_url=os.getenv("NESINE_URL", "https://www.nesine.com/iddaa/futbol").strip(),
            iddaa_url=os.getenv("IDDAA_URL", "https://www.iddaa.com/program").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            http_timeout_seconds=int(os.getenv("HTTP_TIMEOUT_SECONDS", "15")),
        )
