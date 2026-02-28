"""Telegram notification module for high-probability picks."""

from __future__ import annotations

from datetime import datetime
import html
import logging
from typing import Any, Dict, Iterable, List, Tuple

import requests

from analyzer import flatten_probabilities
from config import Settings

LOGGER = logging.getLogger(__name__)

MARKET_LABELS = {
    "over_under_2_5": "2.5 Alt/Ust",
    "btts": "KG Var/Yok",
    "match_result": "MS 1-X-2",
    "first_half_over_0_5": "IY 0.5 Ust",
    "first_half_result": "Ilk Yari Sonucu",
    "second_half_result": "Ikinci Yari Sonucu",
    "corners": "Korner 8.5 Alt/Ust",
}

OUTCOME_LABELS = {
    "over": "Ust",
    "under": "Alt",
    "yes": "Var",
    "no": "Yok",
    "home": "1",
    "draw": "X",
    "away": "2",
    "over_8_5": "Ust 8.5",
    "under_8_5": "Alt 8.5",
}


class TelegramNotifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.telegram_bot_token and self.settings.telegram_chat_id)

    def send_message(self, text: str) -> bool:
        if not self.is_configured:
            LOGGER.warning("Telegram is not configured. Message skipped.")
            return False

        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.settings.http_timeout_seconds)
            response.raise_for_status()
            response_payload = response.json()
            if not response_payload.get("ok", False):
                LOGGER.warning("Telegram API returned non-ok response: %s", response_payload)
                return False
            return True
        except requests.RequestException as exc:
            LOGGER.warning("Telegram sendMessage failed: %s", exc)
            return False

    def find_high_confidence_outcomes(
        self, prediction: Dict[str, Any], threshold: float
    ) -> List[Tuple[str, str, float]]:
        candidates = [
            (market, outcome, probability)
            for market, outcome, probability in flatten_probabilities(prediction.get("probabilities", {}))
            if probability >= threshold
        ]
        return sorted(candidates, key=lambda item: item[2], reverse=True)

    def format_alert(
        self,
        match: Dict[str, Any],
        prediction: Dict[str, Any],
        threshold: float,
    ) -> str:
        kickoff = match.get("kickoff_utc") or "-"
        kickoff_txt = kickoff
        try:
            kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            kickoff_txt = kickoff_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        home = html.escape(str(match.get("home_team", "Home")))
        away = html.escape(str(match.get("away_team", "Away")))
        league = html.escape(str(match.get("competition", "Unknown League")))
        model = html.escape(str(prediction.get("model", "unknown")))
        confidence = float(prediction.get("confidence", 0))
        summary = html.escape(str(prediction.get("summary", "")))

        candidates = self.find_high_confidence_outcomes(prediction, threshold)
        top_lines: List[str] = []
        for market, outcome, probability in candidates[:4]:
            market_label = MARKET_LABELS.get(market, market)
            outcome_label = OUTCOME_LABELS.get(outcome, outcome)
            top_lines.append(f"- <b>{market_label}</b>: {outcome_label} (%{probability:.2f})")

        if not top_lines:
            top_lines.append("- Eslik asilmadi (threshold altinda).")

        body = "\n".join(
            [
                "<b>Futbol Tahmin Uyarisi</b>",
                f"<b>Mac:</b> {home} - {away}",
                f"<b>Lig:</b> {league}",
                f"<b>Baslangic:</b> {html.escape(kickoff_txt)}",
                f"<b>Model:</b> {model}",
                f"<b>Genel Guven:</b> %{confidence:.2f}",
                f"<b>Esik:</b> %{threshold:.2f}",
                "",
                "<b>Yuksek Olasilikli Secimler</b>",
                *top_lines,
                "",
                f"<b>Ozet:</b> {summary}",
            ]
        )
        return body

    def notify_high_confidence(
        self,
        analyzed_matches: Iterable[Dict[str, Any]],
        threshold: float = 80.0,
    ) -> int:
        sent_count = 0
        for item in analyzed_matches:
            match = item.get("match", {})
            prediction = item.get("prediction", {})
            candidates = self.find_high_confidence_outcomes(prediction, threshold)
            if not candidates:
                continue

            message = self.format_alert(match, prediction, threshold=threshold)
            if self.send_message(message):
                sent_count += 1
        return sent_count
