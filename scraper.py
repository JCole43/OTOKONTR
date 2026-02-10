"""Match data collection layer.

Primary sources:
1) Nesine.com
2) Iddaa.com

Fallback source:
3) API-Sports (api-football)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
import requests

from config import Settings

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _normalize_team_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    return normalized


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _default_odds() -> Dict[str, float]:
    return {
        "ms1": 0.0,
        "msx": 0.0,
        "ms2": 0.0,
        "over25": 0.0,
        "under25": 0.0,
        "btts_yes": 0.0,
        "btts_no": 0.0,
        "first_half_over05": 0.0,
        "first_half_under05": 0.0,
        "corners_over85": 0.0,
        "corners_under85": 0.0,
    }


@dataclass
class TeamStats:
    form_points_last5: float = 1.0
    goals_for_avg: float = 1.2
    goals_against_avg: float = 1.2
    xg_for: float = 1.2
    xg_against: float = 1.2
    btts_rate: float = 50.0
    over25_rate: float = 50.0
    corners_for_avg: float = 4.8
    corners_against_avg: float = 4.8


@dataclass
class MatchData:
    fixture_id: str
    competition: str
    kickoff_utc: str
    home_team: str
    away_team: str
    odds: Dict[str, float]
    injuries: Dict[str, List[str]]
    home_stats: TeamStats
    away_stats: TeamStats
    source: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "competition": self.competition,
            "kickoff_utc": self.kickoff_utc,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "odds": self.odds,
            "injuries": self.injuries,
            "stats": {
                "home": asdict(self.home_stats),
                "away": asdict(self.away_stats),
            },
            "source": self.source,
        }


class BaseHttpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                )
            }
        )

    def _get_text(self, url: str) -> str:
        response = self.session.get(url, timeout=self.settings.http_timeout_seconds)
        response.raise_for_status()
        return response.text

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(
            url,
            params=params,
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


class _SchemaMatchExtractor:
    @staticmethod
    def extract_events(html: str) -> List[Dict[str, Any]]:
        """Try to discover match-like objects from embedded JSON structures."""
        events: List[Dict[str, Any]] = []
        soup = BeautifulSoup(html, "html.parser")

        for script in soup.find_all("script", attrs={"id": "__NEXT_DATA__"}):
            content = (script.string or "").strip()
            if not content:
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            events.extend(_SchemaMatchExtractor._walk_for_events(parsed))

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            content = (script.string or "").strip()
            if not content:
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            events.extend(_SchemaMatchExtractor._walk_for_events(parsed))

        all_scripts_text = "\n".join(
            node.string for node in soup.find_all("script") if node and node.string
        )
        patterns = [
            r"__NEXT_DATA__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;",
            r"INITIAL_STATE\s*=\s*(\{.*?\})\s*;",
        ]
        for pattern in patterns:
            match = re.search(pattern, all_scripts_text, flags=re.DOTALL)
            if not match:
                continue
            payload_raw = match.group(1)
            try:
                parsed = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            events.extend(_SchemaMatchExtractor._walk_for_events(parsed))

        # Remove obvious duplicates.
        seen: set[Tuple[str, str, str]] = set()
        unique_events: List[Dict[str, Any]] = []
        for event in events:
            key = (
                _normalize_team_name(event.get("home_team", "")),
                _normalize_team_name(event.get("away_team", "")),
                event.get("kickoff_utc", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_events.append(event)
        return unique_events

    @staticmethod
    def _walk_for_events(node: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(node, dict):
            maybe_event = _SchemaMatchExtractor._parse_event(node)
            if maybe_event:
                items.append(maybe_event)
            for value in node.values():
                items.extend(_SchemaMatchExtractor._walk_for_events(value))
        elif isinstance(node, list):
            for value in node:
                items.extend(_SchemaMatchExtractor._walk_for_events(value))
        return items

    @staticmethod
    def _parse_event(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        home = _SchemaMatchExtractor._first_string(
            node,
            "homeTeam.name",
            "homeTeam",
            "home_team",
            "homeName",
            "team1.name",
            "team1",
            "home.name",
        )
        away = _SchemaMatchExtractor._first_string(
            node,
            "awayTeam.name",
            "awayTeam",
            "away_team",
            "awayName",
            "team2.name",
            "team2",
            "away.name",
        )
        kickoff = _SchemaMatchExtractor._first_string(
            node,
            "startDate",
            "matchDate",
            "kickoff",
            "date",
            "fixture.date",
        )
        if not home or not away:
            return None
        odds = _default_odds()

        event = {
            "fixture_id": str(
                _SchemaMatchExtractor._first_string(node, "id", "fixture.id", "matchId")
                or f"web-{home}-{away}-{kickoff or ''}"
            ),
            "competition": _SchemaMatchExtractor._first_string(
                node,
                "competition.name",
                "tournament.name",
                "league.name",
                "category",
            )
            or "Unknown League",
            "kickoff_utc": kickoff or "",
            "home_team": home,
            "away_team": away,
            "odds": odds,
        }
        return event

    @staticmethod
    def _first_string(node: Dict[str, Any], *paths: str) -> str:
        for path in paths:
            value: Any = node
            ok = True
            for segment in path.split("."):
                if isinstance(value, dict) and segment in value:
                    value = value[segment]
                else:
                    ok = False
                    break
            if not ok:
                continue
            if isinstance(value, dict) and "name" in value:
                value = value["name"]
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


class NesineScraper(BaseHttpClient):
    def fetch_matches(self, target_date: date) -> List[MatchData]:
        try:
            html = self._get_text(self.settings.nesine_url)
        except requests.RequestException as exc:
            LOGGER.warning("Nesine data fetch failed: %s", exc)
            return []
        return self._build_matches_from_html(html, target_date, source="nesine")

    def _build_matches_from_html(
        self,
        html: str,
        target_date: date,
        source: str,
    ) -> List[MatchData]:
        events = _SchemaMatchExtractor.extract_events(html)
        matches: List[MatchData] = []
        for event in events:
            kickoff_dt = _parse_dt(event.get("kickoff_utc", ""))
            if kickoff_dt and kickoff_dt.date() != target_date:
                continue
            matches.append(
                MatchData(
                    fixture_id=event.get("fixture_id", ""),
                    competition=event.get("competition", "Unknown League"),
                    kickoff_utc=event.get("kickoff_utc", ""),
                    home_team=event.get("home_team", "Home"),
                    away_team=event.get("away_team", "Away"),
                    odds={**_default_odds(), **event.get("odds", {})},
                    injuries={
                        event.get("home_team", "Home"): [],
                        event.get("away_team", "Away"): [],
                    },
                    home_stats=TeamStats(),
                    away_stats=TeamStats(),
                    source=source,
                )
            )
        return matches


class IddaaScraper(NesineScraper):
    def fetch_matches(self, target_date: date) -> List[MatchData]:
        try:
            html = self._get_text(self.settings.iddaa_url)
        except requests.RequestException as exc:
            LOGGER.warning("Iddaa data fetch failed: %s", exc)
            return []
        return self._build_matches_from_html(html, target_date, source="iddaa")


class ApiSportsClient(BaseHttpClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.base_url = f"https://{self.settings.api_sports_host}".rstrip("/")
        self._team_cache: Dict[Tuple[int, int, int], TeamStats] = {}
        self._injury_cache: Dict[Tuple[int, int, int], List[str]] = {}
        self._odds_cache: Dict[int, Dict[str, float]] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_sports_key)

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_configured:
            return {}

        headers = {
            "x-apisports-key": self.settings.api_sports_key,
            "x-rapidapi-key": self.settings.api_sports_key,
            "x-rapidapi-host": self.settings.api_sports_host,
        }
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers,
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch_matches(self, target_date: date) -> List[MatchData]:
        if not self.is_configured:
            return []

        try:
            payload = self._request(
                "/fixtures",
                {
                    "date": target_date.isoformat(),
                    "timezone": "Europe/Istanbul",
                },
            )
        except requests.RequestException as exc:
            LOGGER.warning("API-Sports fixture fetch failed: %s", exc)
            return []

        fixtures = payload.get("response", [])
        matches: List[MatchData] = []
        for fixture in fixtures:
            built = self._build_match(fixture)
            if built:
                matches.append(built)
        return matches

    def _build_match(self, fixture_payload: Dict[str, Any]) -> Optional[MatchData]:
        fixture = fixture_payload.get("fixture", {})
        league = fixture_payload.get("league", {})
        teams = fixture_payload.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})

        home_id = home.get("id")
        away_id = away.get("id")
        league_id = league.get("id")
        season = league.get("season")
        if not all([home_id, away_id, league_id, season]):
            return None

        home_name = home.get("name", "Home")
        away_name = away.get("name", "Away")
        fixture_id = str(fixture.get("id", ""))
        kickoff = fixture.get("date", "")
        competition = league.get("name", "Unknown League")

        odds = self._fetch_odds(int(fixture.get("id", 0)))
        home_stats = self._fetch_team_stats(int(home_id), int(league_id), int(season))
        away_stats = self._fetch_team_stats(int(away_id), int(league_id), int(season))

        injuries = {
            home_name: self._fetch_injuries(int(home_id), int(league_id), int(season)),
            away_name: self._fetch_injuries(int(away_id), int(league_id), int(season)),
        }

        return MatchData(
            fixture_id=fixture_id,
            competition=competition,
            kickoff_utc=kickoff,
            home_team=home_name,
            away_team=away_name,
            odds=odds,
            injuries=injuries,
            home_stats=home_stats,
            away_stats=away_stats,
            source="api-sports",
        )

    def _fetch_team_stats(self, team_id: int, league_id: int, season: int) -> TeamStats:
        cache_key = (team_id, league_id, season)
        if cache_key in self._team_cache:
            return self._team_cache[cache_key]

        stats = TeamStats()
        try:
            payload = self._request(
                "/fixtures",
                {"team": team_id, "league": league_id, "season": season, "last": 10},
            )
        except requests.RequestException as exc:
            LOGGER.warning("API-Sports team form fetch failed (team=%s): %s", team_id, exc)
            self._team_cache[cache_key] = stats
            return stats

        fixtures = payload.get("response", [])
        finished = [
            item
            for item in fixtures
            if item.get("fixture", {}).get("status", {}).get("short") in {"FT", "AET", "PEN"}
        ]
        if not finished:
            self._team_cache[cache_key] = stats
            return stats

        goals_for = 0.0
        goals_against = 0.0
        btts = 0
        over25 = 0
        last_five_points = 0
        sample_size = len(finished)

        for idx, game in enumerate(finished):
            home_team_id = game.get("teams", {}).get("home", {}).get("id")
            away_team_id = game.get("teams", {}).get("away", {}).get("id")
            home_goals = _safe_float(game.get("goals", {}).get("home"))
            away_goals = _safe_float(game.get("goals", {}).get("away"))
            if team_id == home_team_id:
                gf = home_goals
                ga = away_goals
            elif team_id == away_team_id:
                gf = away_goals
                ga = home_goals
            else:
                continue

            goals_for += gf
            goals_against += ga
            if gf > 0 and ga > 0:
                btts += 1
            if (gf + ga) >= 3:
                over25 += 1

            if idx < 5:
                if gf > ga:
                    last_five_points += 3
                elif gf == ga:
                    last_five_points += 1

        goals_for_avg = goals_for / sample_size
        goals_against_avg = goals_against / sample_size

        stats = TeamStats(
            form_points_last5=last_five_points / 5.0,
            goals_for_avg=goals_for_avg,
            goals_against_avg=goals_against_avg,
            # API-Sports does not expose xG in this endpoint. We use goals as proxy.
            xg_for=max(0.2, goals_for_avg * 1.08),
            xg_against=max(0.2, goals_against_avg * 1.08),
            btts_rate=(btts / sample_size) * 100,
            over25_rate=(over25 / sample_size) * 100,
            corners_for_avg=4.8 + min(goals_for_avg, 2.0) * 0.45,
            corners_against_avg=4.8 + min(goals_against_avg, 2.0) * 0.45,
        )
        self._team_cache[cache_key] = stats
        return stats

    def _fetch_injuries(self, team_id: int, league_id: int, season: int) -> List[str]:
        cache_key = (team_id, league_id, season)
        if cache_key in self._injury_cache:
            return self._injury_cache[cache_key]

        try:
            payload = self._request(
                "/injuries",
                {"team": team_id, "league": league_id, "season": season},
            )
        except requests.RequestException as exc:
            LOGGER.warning("API-Sports injury fetch failed (team=%s): %s", team_id, exc)
            self._injury_cache[cache_key] = []
            return []

        injuries_raw = payload.get("response", [])
        injuries: List[str] = []
        for item in injuries_raw[:8]:
            player_name = item.get("player", {}).get("name", "Unknown")
            reason = item.get("injury", {}).get("type") or item.get("injury", {}).get("reason")
            if reason:
                injuries.append(f"{player_name} ({reason})")
            else:
                injuries.append(player_name)

        self._injury_cache[cache_key] = injuries
        return injuries

    def _fetch_odds(self, fixture_id: int) -> Dict[str, float]:
        if fixture_id in self._odds_cache:
            return self._odds_cache[fixture_id]

        odds = _default_odds()
        if fixture_id <= 0:
            return odds

        try:
            payload = self._request("/odds", {"fixture": fixture_id})
        except requests.RequestException as exc:
            LOGGER.warning("API-Sports odds fetch failed (fixture=%s): %s", fixture_id, exc)
            self._odds_cache[fixture_id] = odds
            return odds

        for market_pack in payload.get("response", []):
            for bookmaker in market_pack.get("bookmakers", []):
                for bet in bookmaker.get("bets", []):
                    name = str(bet.get("name", "")).lower()
                    values = bet.get("values", [])

                    if "match winner" in name or "winner" == name:
                        for value in values:
                            key = str(value.get("value", "")).lower()
                            odd = _safe_float(value.get("odd"))
                            if key in {"home", "1"}:
                                odds["ms1"] = odd or odds["ms1"]
                            elif key in {"draw", "x"}:
                                odds["msx"] = odd or odds["msx"]
                            elif key in {"away", "2"}:
                                odds["ms2"] = odd or odds["ms2"]

                    elif "both teams score" in name or "both teams to score" in name:
                        for value in values:
                            key = str(value.get("value", "")).lower()
                            odd = _safe_float(value.get("odd"))
                            if key in {"yes", "var"}:
                                odds["btts_yes"] = odd or odds["btts_yes"]
                            elif key in {"no", "yok"}:
                                odds["btts_no"] = odd or odds["btts_no"]

                    elif "goals over/under" in name:
                        for value in values:
                            key = str(value.get("value", "")).lower()
                            odd = _safe_float(value.get("odd"))
                            if "over 2.5" in key:
                                odds["over25"] = odd or odds["over25"]
                            elif "under 2.5" in key:
                                odds["under25"] = odd or odds["under25"]
                            elif "over 0.5" in key and "1st half" in name:
                                odds["first_half_over05"] = odd or odds["first_half_over05"]
                            elif "under 0.5" in key and "1st half" in name:
                                odds["first_half_under05"] = odd or odds["first_half_under05"]

                    elif "1st half goals over/under" in name or "first half goals over/under" in name:
                        for value in values:
                            key = str(value.get("value", "")).lower()
                            odd = _safe_float(value.get("odd"))
                            if "over 0.5" in key:
                                odds["first_half_over05"] = odd or odds["first_half_over05"]
                            elif "under 0.5" in key:
                                odds["first_half_under05"] = odd or odds["first_half_under05"]

                    elif "corners over under" in name or "corners over/under" in name:
                        for value in values:
                            key = str(value.get("value", "")).lower()
                            odd = _safe_float(value.get("odd"))
                            if "over 8.5" in key:
                                odds["corners_over85"] = odd or odds["corners_over85"]
                            elif "under 8.5" in key:
                                odds["corners_under85"] = odd or odds["corners_under85"]

        self._odds_cache[fixture_id] = odds
        return odds


class MatchDataCollector:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.nesine = NesineScraper(self.settings)
        self.iddaa = IddaaScraper(self.settings)
        self.api_sports = ApiSportsClient(self.settings)

    def collect_matches(self, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        target_date = target_date or date.today()
        nesine_matches = self.nesine.fetch_matches(target_date)
        iddaa_matches = self.iddaa.fetch_matches(target_date)
        web_matches = self._deduplicate([*nesine_matches, *iddaa_matches])

        api_matches: List[MatchData] = []
        if self.api_sports.is_configured:
            api_matches = self.api_sports.fetch_matches(target_date)

        if web_matches and api_matches:
            merged = self._merge_web_and_api(web_matches, api_matches)
            return [match.to_payload() for match in merged]

        if web_matches:
            return [match.to_payload() for match in web_matches]

        if api_matches:
            LOGGER.info("Web scraping returned no matches. API-Sports fallback active.")
            return [match.to_payload() for match in api_matches]

        return []

    @staticmethod
    def _key(match: MatchData) -> str:
        kickoff_day = ""
        kickoff_dt = _parse_dt(match.kickoff_utc)
        if kickoff_dt:
            kickoff_day = kickoff_dt.date().isoformat()
        return (
            f"{_normalize_team_name(match.home_team)}::"
            f"{_normalize_team_name(match.away_team)}::{kickoff_day}"
        )

    def _deduplicate(self, matches: Iterable[MatchData]) -> List[MatchData]:
        deduped: Dict[str, MatchData] = {}
        for match in matches:
            key = self._key(match)
            if key not in deduped:
                deduped[key] = match
                continue

            # Keep the newest match while preserving richer odds if present.
            current = deduped[key]
            merged_odds = {
                k: (current.odds.get(k) or match.odds.get(k) or 0.0)
                for k in set(current.odds) | set(match.odds)
            }
            current.odds = {**_default_odds(), **merged_odds}
            current.source = f"{current.source}+{match.source}"
        return list(deduped.values())

    def _merge_web_and_api(self, web_matches: List[MatchData], api_matches: List[MatchData]) -> List[MatchData]:
        api_map = {self._key(match): match for match in api_matches}
        merged: List[MatchData] = []
        used_api_keys: set[str] = set()

        for web in web_matches:
            key = self._key(web)
            if key not in api_map:
                merged.append(web)
                continue

            api_match = api_map[key]
            used_api_keys.add(key)

            merged_odds = {
                name: (web.odds.get(name) or api_match.odds.get(name) or 0.0)
                for name in set(web.odds) | set(api_match.odds)
            }

            merged.append(
                MatchData(
                    fixture_id=api_match.fixture_id or web.fixture_id,
                    competition=api_match.competition or web.competition,
                    kickoff_utc=api_match.kickoff_utc or web.kickoff_utc,
                    home_team=api_match.home_team or web.home_team,
                    away_team=api_match.away_team or web.away_team,
                    odds={**_default_odds(), **merged_odds},
                    injuries=api_match.injuries or web.injuries,
                    home_stats=api_match.home_stats,
                    away_stats=api_match.away_stats,
                    source=f"{web.source}+api-sports",
                )
            )

        for key, api_match in api_map.items():
            if key not in used_api_keys:
                merged.append(api_match)

        return merged
