"""CLI automation entrypoint for scheduled prediction runs.

Example:
python run_pipeline.py --date 2026-02-10 --threshold 80 --send-telegram
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import logging
from typing import Any, Dict, List

from analyzer import GeminiAnalyzer
from config import Settings
from scraper import MatchDataCollector
from telegram_bot import MARKET_LABELS, OUTCOME_LABELS, TelegramNotifier

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _summary_row(item: Dict[str, Any]) -> Dict[str, Any]:
    match = item["match"]
    prediction = item["prediction"]
    top_pick = prediction.get("top_pick", {})
    return {
        "match": f"{match.get('home_team', '-')} - {match.get('away_team', '-')}",
        "league": match.get("competition", "-"),
        "kickoff": match.get("kickoff_utc", "-"),
        "model": prediction.get("model", "-"),
        "confidence": prediction.get("confidence", 0.0),
        "top_market": MARKET_LABELS.get(top_pick.get("market", ""), top_pick.get("market", "-")),
        "top_outcome": OUTCOME_LABELS.get(top_pick.get("outcome", ""), top_pick.get("outcome", "-")),
        "top_probability": top_pick.get("probability", 0.0),
    }


def _top_probability(item: Dict[str, Any]) -> float:
    prediction = item.get("prediction", {})
    top_pick = prediction.get("top_pick", {})
    return float(top_pick.get("probability", 0.0))


def _send_demo_telegram(
    notifier: TelegramNotifier,
    analyzed: List[Dict[str, Any]],
    target_date: date,
) -> bool:
    if not notifier.is_configured:
        LOGGER.warning("Telegram credentials missing; demo notification skipped.")
        return False

    if analyzed:
        best_item = max(analyzed, key=_top_probability)
        best_match = best_item.get("match", {})
        best_prediction = best_item.get("prediction", {})
        demo_body = notifier.format_alert(best_match, best_prediction, threshold=0.0)
        message = "<b>DEMO MESAJI (TEK GONDERIM)</b>\n\n" + demo_body
    else:
        message = (
            "<b>DEMO MESAJI (TEK GONDERIM)</b>\n"
            f"Tarih: {target_date.isoformat()}\n"
            "Bu tarihte analiz edilecek mac bulunamadi, ancak Telegram baglantisi aktif."
        )

    return notifier.send_message(message)


def run(
    target_date: date,
    threshold: float,
    send_telegram: bool,
    send_telegram_demo: bool,
) -> None:
    settings = Settings.from_env()
    notifier = TelegramNotifier(settings=settings)

    if send_telegram_demo:
        sent = _send_demo_telegram(notifier=notifier, analyzed=[], target_date=target_date)
        print(f"Telegram demo sent: {sent}")
        return

    collector = MatchDataCollector(settings=settings)
    analyzer = GeminiAnalyzer(settings=settings)
    matches = collector.collect_matches(target_date=target_date)
    analyzed = analyzer.analyze_matches(matches)

    print(f"Target date: {target_date.isoformat()}")
    print(f"Match count: {len(analyzed)}")
    print("---- SUMMARY ----")
    for item in analyzed:
        row = _summary_row(item)
        print(json.dumps(row, ensure_ascii=True))

    if send_telegram:
        if not notifier.is_configured:
            LOGGER.warning("Telegram credentials missing; notifications skipped.")
            return
        sent = notifier.notify_high_confidence(analyzed, threshold=threshold)
        print(f"Telegram sent count: {sent}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Football prediction automation runner")
    parser.add_argument("--date", dest="date_str", default=date.today().isoformat())
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument(
        "--send-telegram-demo",
        action="store_true",
        help="Send only one demo notification to Telegram.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(
        target_date=_parse_date(args.date_str),
        threshold=args.threshold,
        send_telegram=args.send_telegram,
        send_telegram_demo=args.send_telegram_demo,
    )
