"""Persistent history, coupon suggestions, and ROI tracking utilities."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import html
import json
import logging
from math import prod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import Settings
from telegram_bot import MARKET_LABELS, OUTCOME_LABELS

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "history.json"
FINAL_STATUSES = {"FT", "AET", "PEN"}
VOID_STATUSES = {"CANC", "PST", "ABD", "AWD", "WO", "INT"}

ODD_KEY_MAP: Dict[str, Dict[str, str]] = {
    "match_result": {"home": "ms1", "draw": "msx", "away": "ms2"},
    "over_under_2_5": {"over": "over25", "under": "under25"},
    "btts": {"yes": "btts_yes", "no": "btts_no"},
    "first_half_over_0_5": {"over": "first_half_over05", "under": "first_half_under05"},
    "corners": {"over_8_5": "corners_over85", "under_8_5": "corners_under85"},
}

SUPPORTED_SETTLEMENT_MARKETS = {
    "match_result",
    "over_under_2_5",
    "btts",
    "first_half_over_0_5",
    "first_half_result",
    "second_half_result",
    "corners",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _history_default() -> Dict[str, Any]:
    return {"predictions": [], "coupons": []}


def _prediction_id(
    fixture_id: str,
    home_team: str,
    away_team: str,
    kickoff_utc: str,
    market: str,
    outcome: str,
) -> str:
    fixture_txt = (fixture_id or "").strip()
    if fixture_txt.isdigit():
        base = fixture_txt
    else:
        raw = f"{home_team}|{away_team}|{kickoff_utc}".lower().encode("utf-8")
        base = hashlib.sha1(raw).hexdigest()[:16]
    return f"{base}:{market}:{outcome}"


def _coupon_id(strategy: str, picks: Iterable[Dict[str, Any]]) -> str:
    pick_fingerprint = "|".join(sorted(str(item.get("prediction_id", "")) for item in picks))
    raw = f"{strategy}|{pick_fingerprint}|{_now_iso()}".encode("utf-8")
    return f"coupon-{hashlib.sha1(raw).hexdigest()[:14]}"


def resolve_pick_odd(match: Dict[str, Any], market: str, outcome: str) -> float:
    odd_key = ODD_KEY_MAP.get(market, {}).get(outcome)
    if not odd_key:
        return 0.0
    odd = _safe_float(match.get("odds", {}).get(odd_key))
    return odd if odd > 1.0 else 0.0


def _outcome_to_label(market: str, outcome: str) -> str:
    market_txt = MARKET_LABELS.get(market, market)
    outcome_txt = OUTCOME_LABELS.get(outcome, outcome)
    return f"{market_txt}: {outcome_txt}"


def _build_pick_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    match = item.get("match", {})
    prediction = item.get("prediction", {})
    top_pick = prediction.get("top_pick", {}) or {}

    market = str(top_pick.get("market", "")).strip()
    outcome = str(top_pick.get("outcome", "")).strip()
    probability = _safe_float(top_pick.get("probability", 0.0))
    if not market or not outcome:
        return None

    fixture_id = str(match.get("fixture_id", "")).strip()
    home_team = str(match.get("home_team", "Home"))
    away_team = str(match.get("away_team", "Away"))
    kickoff_utc = str(match.get("kickoff_utc", ""))
    prediction_id = _prediction_id(
        fixture_id=fixture_id,
        home_team=home_team,
        away_team=away_team,
        kickoff_utc=kickoff_utc,
        market=market,
        outcome=outcome,
    )
    odd = resolve_pick_odd(match, market, outcome)

    return {
        "id": prediction_id,
        "fixture_id": fixture_id,
        "competition": str(match.get("competition", "Unknown League")),
        "home_team": home_team,
        "away_team": away_team,
        "match_label": f"{home_team} - {away_team}",
        "kickoff_utc": kickoff_utc,
        "market": market,
        "outcome": outcome,
        "pick_label": _outcome_to_label(market, outcome),
        "probability": probability,
        "confidence": _safe_float(prediction.get("confidence", 0.0)),
        "odd": odd,
        "model": str(prediction.get("model", "unknown")),
        "summary": str(prediction.get("summary", "")),
        "source": str(match.get("source", "")),
        "status": "pending",
        "result": {},
        "profit": 0.0,
    }


def generate_coupon_suggestions(
    analyzed_matches: List[Dict[str, Any]],
    min_probability: float = 65.0,
    legs_count: int = 3,
    max_coupons: int = 2,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in analyzed_matches:
        pick = _build_pick_record(item)
        if not pick:
            continue
        if pick["probability"] < min_probability:
            continue
        if pick["odd"] <= 1.0:
            continue
        expected_value = (pick["probability"] / 100.0) * pick["odd"] - 1.0
        pick["expected_value"] = expected_value
        candidates.append(pick)

    if len(candidates) < legs_count:
        return []

    by_probability = sorted(candidates, key=lambda item: item["probability"], reverse=True)
    by_value = sorted(candidates, key=lambda item: item["expected_value"], reverse=True)

    suggestions: List[Dict[str, Any]] = []
    strategies = [
        ("safe", by_probability),
        ("value", by_value),
    ]
    for strategy, ordered in strategies:
        picks = ordered[:legs_count]
        if len(picks) < legs_count:
            continue
        coupon = _build_coupon(strategy=strategy, picks=picks)
        suggestions.append(coupon)

    # remove duplicate pick combinations
    unique: List[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for coupon in suggestions:
        fingerprint = "|".join(sorted(item["prediction_id"] for item in coupon["legs"]))
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        unique.append(coupon)

    return unique[:max_coupons]


def _build_coupon(strategy: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined_odd = prod(max(1.0, _safe_float(item.get("odd", 0.0), 1.0)) for item in picks)
    combined_probability = prod(
        max(0.0, min(1.0, _safe_float(item.get("probability", 0.0)) / 100.0)) for item in picks
    )
    expected_roi_pct = ((combined_probability * combined_odd) - 1.0) * 100.0
    coupon_legs = [
        {
            "prediction_id": item["id"],
            "fixture_id": item.get("fixture_id", ""),
            "match_label": item.get("match_label", ""),
            "competition": item.get("competition", ""),
            "market": item.get("market", ""),
            "outcome": item.get("outcome", ""),
            "pick_label": item.get("pick_label", ""),
            "probability": _safe_float(item.get("probability", 0.0)),
            "odd": _safe_float(item.get("odd", 0.0)),
        }
        for item in picks
    ]
    return {
        "id": _coupon_id(strategy=strategy, picks=coupon_legs),
        "strategy": strategy,
        "created_at": _now_iso(),
        "legs": coupon_legs,
        "legs_count": len(coupon_legs),
        "combined_odd": round(combined_odd, 4),
        "combined_probability": round(combined_probability * 100.0, 2),
        "expected_roi_pct": round(expected_roi_pct, 2),
        "stake": 1.0,
        "status": "pending",
        "profit": 0.0,
        "settled_at": "",
        "sent_to_telegram": False,
    }


def format_coupon_message(coupon: Dict[str, Any]) -> str:
    title = "Guvenli Kombine" if coupon.get("strategy") == "safe" else "Deger Kombinesi"
    lines = [
        f"<b>AI Kuponu - {html.escape(title)}</b>",
        f"<b>Bacak:</b> {int(coupon.get('legs_count', 0))}",
        f"<b>Kombine Oran:</b> {float(coupon.get('combined_odd', 0.0)):.2f}",
        f"<b>Tahmini Kazanma:</b> %{float(coupon.get('combined_probability', 0.0)):.2f}",
        f"<b>Beklenen ROI:</b> %{float(coupon.get('expected_roi_pct', 0.0)):.2f}",
        "",
    ]
    for idx, leg in enumerate(coupon.get("legs", []), start=1):
        lines.append(
            (
                f"{idx}) <b>{html.escape(str(leg.get('match_label', '-')))}</b>\n"
                f"   {html.escape(str(leg.get('pick_label', '-')))} | "
                f"Olasilik %{float(leg.get('probability', 0.0)):.2f} | "
                f"Oran {float(leg.get('odd', 0.0)):.2f}"
            )
        )
    return "\n".join(lines)


def compute_prediction_roi(history_data: Dict[str, Any]) -> Dict[str, float]:
    predictions = history_data.get("predictions", [])
    settled = [item for item in predictions if item.get("status") in {"win", "loss"}]
    wins = [item for item in settled if item.get("status") == "win"]
    total_stake = float(len(settled))
    total_profit = float(sum(_safe_float(item.get("profit", 0.0)) for item in settled))
    roi_pct = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    win_rate_pct = (len(wins) / len(settled) * 100.0) if settled else 0.0

    return {
        "total_predictions": float(len(predictions)),
        "pending_predictions": float(len([x for x in predictions if x.get("status") == "pending"])),
        "settled_predictions": float(len(settled)),
        "wins": float(len(wins)),
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "win_rate_pct": round(win_rate_pct, 2),
    }


def compute_coupon_roi(history_data: Dict[str, Any]) -> Dict[str, float]:
    coupons = history_data.get("coupons", [])
    settled = [item for item in coupons if item.get("status") in {"win", "loss"}]
    wins = [item for item in settled if item.get("status") == "win"]
    total_stake = sum(_safe_float(item.get("stake", 1.0), 1.0) for item in settled)
    total_profit = sum(_safe_float(item.get("profit", 0.0)) for item in settled)
    roi_pct = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    win_rate_pct = (len(wins) / len(settled) * 100.0) if settled else 0.0
    return {
        "total_coupons": float(len(coupons)),
        "pending_coupons": float(len([x for x in coupons if x.get("status") == "pending"])),
        "settled_coupons": float(len(settled)),
        "wins": float(len(wins)),
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "win_rate_pct": round(win_rate_pct, 2),
    }


class HistoryStore:
    def __init__(self, settings: Settings, history_path: Path | None = None) -> None:
        self.settings = settings
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._fixture_cache: Dict[str, Dict[str, Any]] = {}
        self._corners_cache: Dict[str, Optional[int]] = {}

    def load(self) -> Dict[str, Any]:
        if not self.history_path.exists():
            return _history_default()
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("History file unreadable. Reinitializing.")
            return _history_default()
        return self._coerce_schema(raw)

    def save(self, history_data: Dict[str, Any]) -> None:
        payload = self._coerce_schema(history_data)
        self.history_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def upsert_predictions(self, analyzed_matches: List[Dict[str, Any]], target_date: date) -> int:
        history_data = self.load()
        existing_map = {item.get("id", ""): item for item in history_data.get("predictions", [])}
        upserted = 0
        now = _now_iso()

        for item in analyzed_matches:
            pick = _build_pick_record(item)
            if not pick:
                continue

            previous = existing_map.get(pick["id"], {})
            pick["target_date"] = target_date.isoformat()
            pick["created_at"] = previous.get("created_at", now)
            pick["updated_at"] = now
            pick["status"] = previous.get("status", "pending")
            pick["result"] = previous.get("result", {})
            pick["profit"] = _safe_float(previous.get("profit", 0.0))
            if pick["status"] not in {"win", "loss", "void", "settled_no_odds"}:
                pick["status"] = "pending"
                pick["result"] = {}
                pick["profit"] = 0.0

            existing_map[pick["id"]] = pick
            upserted += 1

        history_data["predictions"] = sorted(
            existing_map.values(),
            key=lambda item: item.get("kickoff_utc", ""),
            reverse=True,
        )
        self.save(history_data)
        return upserted

    def add_coupon(self, coupon: Dict[str, Any], sent_to_telegram: bool = False) -> str:
        history_data = self.load()
        coupons = history_data.get("coupons", [])
        fingerprint = "|".join(sorted(item.get("prediction_id", "") for item in coupon.get("legs", [])))

        for item in coupons:
            existing_fingerprint = "|".join(
                sorted(leg.get("prediction_id", "") for leg in item.get("legs", []))
            )
            if existing_fingerprint == fingerprint and item.get("status") == "pending":
                if sent_to_telegram:
                    item["sent_to_telegram"] = True
                    item["updated_at"] = _now_iso()
                    self.save(history_data)
                return str(item.get("id", ""))

        coupon_to_store = {
            **coupon,
            "id": coupon.get("id") or _coupon_id(coupon.get("strategy", "safe"), coupon.get("legs", [])),
            "created_at": coupon.get("created_at", _now_iso()),
            "updated_at": _now_iso(),
            "status": coupon.get("status", "pending"),
            "stake": _safe_float(coupon.get("stake", 1.0), 1.0),
            "profit": _safe_float(coupon.get("profit", 0.0)),
            "settled_at": coupon.get("settled_at", ""),
            "sent_to_telegram": bool(sent_to_telegram or coupon.get("sent_to_telegram", False)),
        }
        coupons.append(coupon_to_store)
        history_data["coupons"] = sorted(coupons, key=lambda item: item.get("created_at", ""), reverse=True)
        self.save(history_data)
        return str(coupon_to_store["id"])

    def refresh_results(self, max_fixtures: int = 80) -> Dict[str, int]:
        history_data = self.load()
        prediction_updates = self._settle_predictions(history_data, max_fixtures=max_fixtures)
        coupon_updates = self._settle_coupons(history_data)
        self.save(history_data)
        return {
            "prediction_updates": prediction_updates,
            "coupon_updates": coupon_updates,
        }

    @staticmethod
    def _coerce_schema(history_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(history_data, dict):
            return _history_default()
        predictions = history_data.get("predictions")
        coupons = history_data.get("coupons")
        if not isinstance(predictions, list):
            predictions = []
        if not isinstance(coupons, list):
            coupons = []
        return {"predictions": predictions, "coupons": coupons}

    def _settle_predictions(self, history_data: Dict[str, Any], max_fixtures: int) -> int:
        if not self.settings.api_sports_key:
            return 0

        now_utc = datetime.now(timezone.utc)
        pending = []
        for item in history_data.get("predictions", []):
            if item.get("status") != "pending":
                continue
            kickoff_dt = _parse_dt(str(item.get("kickoff_utc", "")))
            if kickoff_dt and kickoff_dt > now_utc:
                continue
            fixture_id = str(item.get("fixture_id", "")).strip()
            if not fixture_id.isdigit():
                continue
            if item.get("market") not in SUPPORTED_SETTLEMENT_MARKETS:
                continue
            pending.append(item)

        if not pending:
            return 0

        fixture_ids = sorted({str(item["fixture_id"]) for item in pending})[:max_fixtures]
        fixture_map: Dict[str, Dict[str, Any]] = {}
        for fixture_id in fixture_ids:
            fixture_payload = self._fetch_fixture(fixture_id)
            if fixture_payload:
                fixture_map[fixture_id] = fixture_payload

        updated = 0
        for item in pending:
            fixture = fixture_map.get(str(item.get("fixture_id", "")))
            if not fixture:
                continue
            if self._settle_prediction_item(item, fixture):
                updated += 1
        return updated

    def _settle_prediction_item(self, item: Dict[str, Any], fixture: Dict[str, Any]) -> bool:
        status_short = str(fixture.get("fixture", {}).get("status", {}).get("short", ""))
        if status_short in VOID_STATUSES:
            item["status"] = "void"
            item["profit"] = 0.0
            item["result"] = {
                "status_short": status_short,
                "actual_outcome": "void",
                "score": "-",
            }
            item["updated_at"] = _now_iso()
            return True

        if status_short not in FINAL_STATUSES:
            return False

        actual_outcome = self._resolve_actual_outcome(item.get("market", ""), fixture)
        if actual_outcome is None:
            return False

        selected_outcome = str(item.get("outcome", ""))
        odd = _safe_float(item.get("odd", 0.0))
        won = selected_outcome == actual_outcome

        if odd <= 1.0:
            item["status"] = "settled_no_odds"
            item["profit"] = 0.0
        else:
            item["status"] = "win" if won else "loss"
            item["profit"] = round((odd - 1.0) if won else -1.0, 4)

        goals_home = _safe_int(fixture.get("goals", {}).get("home"))
        goals_away = _safe_int(fixture.get("goals", {}).get("away"))
        ht_home = _safe_int(fixture.get("score", {}).get("halftime", {}).get("home"))
        ht_away = _safe_int(fixture.get("score", {}).get("halftime", {}).get("away"))
        item["result"] = {
            "status_short": status_short,
            "actual_outcome": actual_outcome,
            "score": f"{goals_home}-{goals_away}",
            "half_time_score": f"{ht_home}-{ht_away}",
        }
        item["updated_at"] = _now_iso()
        return True

    def _resolve_actual_outcome(self, market: str, fixture: Dict[str, Any]) -> Optional[str]:
        home_goals = fixture.get("goals", {}).get("home")
        away_goals = fixture.get("goals", {}).get("away")
        if home_goals is None or away_goals is None:
            return None

        hg = _safe_int(home_goals)
        ag = _safe_int(away_goals)
        ht = fixture.get("score", {}).get("halftime", {})
        ht_h = ht.get("home")
        ht_a = ht.get("away")

        if market == "match_result":
            return "home" if hg > ag else "away" if hg < ag else "draw"
        if market == "over_under_2_5":
            return "over" if (hg + ag) >= 3 else "under"
        if market == "btts":
            return "yes" if hg > 0 and ag > 0 else "no"
        if market == "first_half_over_0_5":
            if ht_h is None or ht_a is None:
                return None
            return "over" if (_safe_int(ht_h) + _safe_int(ht_a)) >= 1 else "under"
        if market == "first_half_result":
            if ht_h is None or ht_a is None:
                return None
            hth = _safe_int(ht_h)
            hta = _safe_int(ht_a)
            return "home" if hth > hta else "away" if hth < hta else "draw"
        if market == "second_half_result":
            if ht_h is None or ht_a is None:
                return None
            sh_h = hg - _safe_int(ht_h)
            sh_a = ag - _safe_int(ht_a)
            return "home" if sh_h > sh_a else "away" if sh_h < sh_a else "draw"
        if market == "corners":
            fixture_id = str(fixture.get("fixture", {}).get("id", ""))
            total_corners = self._fetch_total_corners(fixture_id)
            if total_corners is None:
                return None
            return "over_8_5" if total_corners >= 9 else "under_8_5"
        return None

    def _settle_coupons(self, history_data: Dict[str, Any]) -> int:
        prediction_map = {item.get("id", ""): item for item in history_data.get("predictions", [])}
        updates = 0
        for coupon in history_data.get("coupons", []):
            if coupon.get("status") != "pending":
                continue

            legs = coupon.get("legs", [])
            leg_statuses = []
            for leg in legs:
                prediction = prediction_map.get(leg.get("prediction_id", ""))
                if not prediction:
                    leg_statuses.append("pending")
                else:
                    leg_statuses.append(str(prediction.get("status", "pending")))

            if not leg_statuses:
                continue
            if any(status == "pending" for status in leg_statuses):
                continue

            stake = _safe_float(coupon.get("stake", 1.0), 1.0)
            if any(status == "loss" for status in leg_statuses):
                coupon["status"] = "loss"
                coupon["profit"] = round(-stake, 4)
            else:
                effective_odd = 1.0
                for leg in legs:
                    prediction = prediction_map.get(leg.get("prediction_id", ""))
                    if not prediction:
                        continue
                    status = str(prediction.get("status", "pending"))
                    if status == "win":
                        effective_odd *= max(1.0, _safe_float(prediction.get("odd", leg.get("odd", 1.0)), 1.0))
                if effective_odd <= 1.0:
                    coupon["status"] = "void"
                    coupon["profit"] = 0.0
                else:
                    coupon["status"] = "win"
                    coupon["profit"] = round(stake * (effective_odd - 1.0), 4)

            coupon["settled_at"] = _now_iso()
            coupon["updated_at"] = _now_iso()
            updates += 1
        return updates

    def _fetch_fixture(self, fixture_id: str) -> Dict[str, Any]:
        if fixture_id in self._fixture_cache:
            return self._fixture_cache[fixture_id]
        payload = self._request(endpoint="/fixtures", params={"id": fixture_id})
        response = payload.get("response", []) if payload else []
        fixture = response[0] if response else {}
        self._fixture_cache[fixture_id] = fixture
        return fixture

    def _fetch_total_corners(self, fixture_id: str) -> Optional[int]:
        if not fixture_id:
            return None
        if fixture_id in self._corners_cache:
            return self._corners_cache[fixture_id]

        payload = self._request(endpoint="/fixtures/statistics", params={"fixture": fixture_id})
        total = 0
        found = False
        for team_pack in payload.get("response", []) if payload else []:
            for stat in team_pack.get("statistics", []):
                stat_type = str(stat.get("type", "")).lower()
                if "corner" not in stat_type:
                    continue
                found = True
                total += _safe_int(stat.get("value"))
        value = total if found else None
        self._corners_cache[fixture_id] = value
        return value

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.api_sports_key:
            return {}
        host = self.settings.api_sports_host
        if host.startswith("https://") or host.startswith("http://"):
            base_url = host.rstrip("/")
            rapidapi_host = host.replace("https://", "").replace("http://", "").rstrip("/")
        else:
            base_url = f"https://{host}".rstrip("/")
            rapidapi_host = host

        headers = {
            "x-apisports-key": self.settings.api_sports_key,
            "x-rapidapi-key": self.settings.api_sports_key,
            "x-rapidapi-host": rapidapi_host,
        }
        try:
            response = requests.get(
                f"{base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=self.settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors", {})
            if isinstance(errors, dict) and errors.get("rateLimit"):
                LOGGER.warning("API-Sports rate limit while updating history: %s", errors.get("rateLimit"))
                return {}
            return payload
        except requests.RequestException as exc:
            LOGGER.warning("History API request failed (%s): %s", endpoint, exc)
            return {}
