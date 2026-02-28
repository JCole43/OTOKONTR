"""Application settings loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def _clean_env_value(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    if raw is None:
        return ""
    # Windows users often paste quoted values into .env files.
    return str(raw).strip().strip('"').strip("'")


def _load_env_files() -> None:
    root_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        root_dir / ".env",
        Path.cwd() / ".env.local",
        root_dir / ".env.local",
        Path.cwd() / ".env.example",
        root_dir / ".env.example",
    ]

    loaded_any = False
    seen: set[str] = set()
    for file_path in candidates:
        normalized = str(file_path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)

        if file_path.exists() and file_path.is_file():
            load_dotenv(dotenv_path=file_path, override=False)
            loaded_any = True

    if not loaded_any:
        load_dotenv(override=False)


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
        _load_env_files()
        api_sports_key = (
            _clean_env_value("API_SPORTS_KEY")
            or _clean_env_value("FOOTBALL_API_KEY")
        )
        api_sports_host = (
            _clean_env_value("API_SPORTS_HOST")
            or _clean_env_value("FOOTBALL_API_HOST")
        )
        api_sports_url = (
            _clean_env_value("API_SPORTS_URL")
            or _clean_env_value("FOOTBALL_API_URL")
        )

        if not api_sports_host and api_sports_url:
            api_sports_host = (
                api_sports_url.replace("https://", "")
                .replace("http://", "")
                .strip("/")
            )
        if not api_sports_host:
            api_sports_host = "v3.football.api-sports.io"

        timeout_raw = _clean_env_value("HTTP_TIMEOUT_SECONDS", "15")
        try:
            timeout_value = int(timeout_raw)
        except ValueError:
            timeout_value = 15

        return cls(
            gemini_api_key=_clean_env_value("GEMINI_API_KEY"),
            api_sports_key=api_sports_key,
            api_sports_host=api_sports_host,
            nesine_url=_clean_env_value("NESINE_URL", "https://www.nesine.com/iddaa/futbol"),
            iddaa_url=_clean_env_value("IDDAA_URL", "https://www.iddaa.com/program"),
            telegram_bot_token=_clean_env_value("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_clean_env_value("TELEGRAM_CHAT_ID"),
            http_timeout_seconds=timeout_value,
        )
