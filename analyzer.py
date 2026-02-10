"""Prediction and analysis engine.

The engine combines:
1) A deterministic statistical baseline model
2) Gemini 1.5 Pro refinement (if API key is configured)
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import Settings

LOGGER = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_percentages(distribution: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, _safe_float(v)) for k, v in distribution.items()}
    total = sum(cleaned.values())
    if total <= 0:
        equal = round(100.0 / max(len(cleaned), 1), 2)
        normalized = {k: equal for k in cleaned}
    else:
        normalized = {k: round((v / total) * 100, 2) for k, v in cleaned.items()}

    # rounding fix to keep exact 100
    diff = round(100.0 - sum(normalized.values()), 2)
    if normalized:
        first_key = next(iter(normalized))
        normalized[first_key] = round(normalized[first_key] + diff, 2)
    return normalized


def _poisson_pmf(lambda_value: float, goals: int) -> float:
    if lambda_value <= 0:
        return 0.0 if goals > 0 else 1.0
    return math.exp(-lambda_value) * (lambda_value**goals) / math.factorial(goals)


def _score_matrix(home_lambda: float, away_lambda: float, max_goals: int = 8) -> List[List[float]]:
    matrix: List[List[float]] = []
    for h_goals in range(max_goals + 1):
        row: List[float] = []
        for a_goals in range(max_goals + 1):
            row.append(_poisson_pmf(home_lambda, h_goals) * _poisson_pmf(away_lambda, a_goals))
        matrix.append(row)
    return matrix


def _poisson_cdf(lambda_value: float, goals: int) -> float:
    return sum(_poisson_pmf(lambda_value, k) for k in range(goals + 1))


def _matrix_market_probabilities(matrix: List[List[float]]) -> Dict[str, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over25 = 0.0
    btts_yes = 0.0

    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

            if home_goals + away_goals >= 3:
                over25 += probability
            if home_goals >= 1 and away_goals >= 1:
                btts_yes += probability

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over25": over25,
        "under25": 1.0 - over25,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
    }


def _odds_to_distribution(odds_values: Dict[str, float]) -> Dict[str, float]:
    implied = {}
    for key, odd in odds_values.items():
        odd_float = _safe_float(odd)
        if odd_float > 1.0:
            implied[key] = 1.0 / odd_float
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in implied.items()}


def _top_pick(probabilities: Dict[str, Dict[str, float]]) -> Tuple[str, str, float]:
    best_market = ""
    best_outcome = ""
    best_prob = 0.0
    for market, outcomes in probabilities.items():
        for outcome, probability in outcomes.items():
            if probability > best_prob:
                best_market = market
                best_outcome = outcome
                best_prob = probability
    return best_market, best_outcome, round(best_prob, 2)


class StatisticalModel:
    def predict(self, match: Dict[str, Any]) -> Dict[str, Any]:
        home_stats = match.get("stats", {}).get("home", {})
        away_stats = match.get("stats", {}).get("away", {})
        odds = match.get("odds", {})

        home_form = _safe_float(home_stats.get("form_points_last5"), 1.0)
        away_form = _safe_float(away_stats.get("form_points_last5"), 1.0)

        home_xg = (
            _safe_float(home_stats.get("xg_for"), 1.2)
            + _safe_float(away_stats.get("xg_against"), 1.2)
        ) / 2.0
        away_xg = (
            _safe_float(away_stats.get("xg_for"), 1.2)
            + _safe_float(home_stats.get("xg_against"), 1.2)
        ) / 2.0

        # Form-adjusted expected goals.
        home_xg *= 1.0 + (home_form - away_form) * 0.05
        away_xg *= 1.0 + (away_form - home_form) * 0.05
        home_xg = _clamp(home_xg, 0.2, 3.8)
        away_xg = _clamp(away_xg, 0.2, 3.8)

        full_time_matrix = _score_matrix(home_xg, away_xg)
        full_time = _matrix_market_probabilities(full_time_matrix)

        first_half_matrix = _score_matrix(home_xg * 0.46, away_xg * 0.46)
        first_half = _matrix_market_probabilities(first_half_matrix)
        first_half_over05 = 1.0 - _poisson_pmf((home_xg + away_xg) * 0.46, 0)

        second_half_matrix = _score_matrix(home_xg * 0.54, away_xg * 0.54)
        second_half = _matrix_market_probabilities(second_half_matrix)

        expected_total_corners = (
            (_safe_float(home_stats.get("corners_for_avg"), 4.8) + _safe_float(away_stats.get("corners_against_avg"), 4.8))
            / 2.0
            + (_safe_float(away_stats.get("corners_for_avg"), 4.8) + _safe_float(home_stats.get("corners_against_avg"), 4.8))
            / 2.0
        )
        expected_total_corners = _clamp(expected_total_corners, 6.0, 14.0)
        corners_over85 = 1.0 - _poisson_cdf(expected_total_corners, 8)

        probabilities = {
            "over_under_2_5": _normalize_percentages(
                {"over": full_time["over25"] * 100, "under": full_time["under25"] * 100}
            ),
            "btts": _normalize_percentages(
                {"yes": full_time["btts_yes"] * 100, "no": full_time["btts_no"] * 100}
            ),
            "match_result": _normalize_percentages(
                {
                    "home": full_time["home_win"] * 100,
                    "draw": full_time["draw"] * 100,
                    "away": full_time["away_win"] * 100,
                }
            ),
            "first_half_over_0_5": _normalize_percentages(
                {"over": first_half_over05 * 100, "under": (1.0 - first_half_over05) * 100}
            ),
            "first_half_result": _normalize_percentages(
                {
                    "home": first_half["home_win"] * 100,
                    "draw": first_half["draw"] * 100,
                    "away": first_half["away_win"] * 100,
                }
            ),
            "second_half_result": _normalize_percentages(
                {
                    "home": second_half["home_win"] * 100,
                    "draw": second_half["draw"] * 100,
                    "away": second_half["away_win"] * 100,
                }
            ),
            "corners": _normalize_percentages(
                {"over_8_5": corners_over85 * 100, "under_8_5": (1.0 - corners_over85) * 100}
            ),
        }

        self._blend_with_odds(probabilities, odds)

        market, outcome, probability = _top_pick(probabilities)
        confidence = round(
            sum(max(outcomes.values()) for outcomes in probabilities.values()) / len(probabilities), 2
        )

        return {
            "probabilities": probabilities,
            "confidence": confidence,
            "top_pick": {"market": market, "outcome": outcome, "probability": probability},
            "key_factors": [
                f"Home xG: {home_xg:.2f}",
                f"Away xG: {away_xg:.2f}",
                f"Expected corners: {expected_total_corners:.2f}",
            ],
            "summary": "Statistical baseline generated from form, xG proxy, odds and market priors.",
            "model": "statistical",
        }

    def _blend_with_odds(
        self, probabilities: Dict[str, Dict[str, float]], odds: Dict[str, Any], weight: float = 0.35
    ) -> None:
        if not odds:
            return

        market_to_odds_map = {
            "match_result": {"home": odds.get("ms1"), "draw": odds.get("msx"), "away": odds.get("ms2")},
            "over_under_2_5": {"over": odds.get("over25"), "under": odds.get("under25")},
            "btts": {"yes": odds.get("btts_yes"), "no": odds.get("btts_no")},
            "first_half_over_0_5": {
                "over": odds.get("first_half_over05"),
                "under": odds.get("first_half_under05"),
            },
            "corners": {"over_8_5": odds.get("corners_over85"), "under_8_5": odds.get("corners_under85")},
        }

        for market, odd_map in market_to_odds_map.items():
            implied = _odds_to_distribution(odd_map)
            if not implied:
                continue

            merged = {}
            for outcome, model_pct in probabilities[market].items():
                model_prob = model_pct / 100.0
                odd_prob = implied.get(outcome, model_prob)
                merged[outcome] = ((1.0 - weight) * model_prob + weight * odd_prob) * 100.0
            probabilities[market] = _normalize_percentages(merged)


class GeminiAnalyzer:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gemini-1.5-pro-latest",
        temperature: float = 0.15,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.model_name = model_name
        self.temperature = temperature
        self.stat_model = StatisticalModel()
        self._client = None
        self._genai = None

        if not self.settings.gemini_api_key:
            LOGGER.info("GEMINI_API_KEY is not set. Statistical-only mode enabled.")
            return

        try:
            import google.generativeai as genai
        except ImportError:
            LOGGER.warning("google-generativeai is not installed. Statistical-only mode enabled.")
            return

        try:
            genai.configure(api_key=self.settings.gemini_api_key)
            self._genai = genai
            self.model_name = self._resolve_model_name(preferred_model=self.model_name)
            self._client = genai.GenerativeModel(self.model_name)
            LOGGER.info("Gemini model selected: %s", self.model_name)
        except Exception as exc:  # pragma: no cover - network/client setup
            LOGGER.warning("Gemini client initialization failed: %s", exc)
            self._client = None

    @property
    def gemini_enabled(self) -> bool:
        return self._client is not None

    def analyze_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        baseline = self.stat_model.predict(match)
        if not self.gemini_enabled:
            baseline["model"] = "statistical_fallback"
            baseline["summary"] = (
                "Gemini API unavailable or not configured. Returning statistical fallback."
            )
            return baseline

        try:
            gemini_payload = self._call_gemini(match, baseline)
            merged = self._merge_predictions(baseline, gemini_payload)
            merged["model"] = "hybrid_statistical_gemini"
            return merged
        except Exception as exc:  # pragma: no cover - network/model runtime
            if self._should_retry_with_fallback_model(exc) and self._reinitialize_with_fallback_model():
                try:
                    gemini_payload = self._call_gemini(match, baseline)
                    merged = self._merge_predictions(baseline, gemini_payload)
                    merged["model"] = "hybrid_statistical_gemini"
                    return merged
                except Exception as retry_exc:  # pragma: no cover - network/model runtime
                    LOGGER.warning(
                        "Gemini retry with fallback model failed; using statistical model: %s",
                        retry_exc,
                    )

            LOGGER.warning("Gemini analysis failed; fallback to statistical model: %s", exc)
            baseline["model"] = "statistical_fallback"
            baseline["summary"] = (
                "Gemini request failed at runtime. Returning statistical fallback."
            )
            return baseline

    def analyze_matches(self, matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        analyzed: List[Dict[str, Any]] = []
        for match in matches:
            analyzed.append(
                {
                    "match": match,
                    "prediction": self.analyze_match(match),
                }
            )
        return analyzed

    def _call_gemini(self, match: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are an expert football betting analyst.\n"
            "Use the provided match data and baseline probabilities to calibrate risk-aware predictions.\n"
            "Return STRICT JSON only. Do not include markdown fences.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "probabilities": {\n'
            '    "over_under_2_5": {"over": number, "under": number},\n'
            '    "btts": {"yes": number, "no": number},\n'
            '    "match_result": {"home": number, "draw": number, "away": number},\n'
            '    "first_half_over_0_5": {"over": number, "under": number},\n'
            '    "first_half_result": {"home": number, "draw": number, "away": number},\n'
            '    "second_half_result": {"home": number, "draw": number, "away": number},\n'
            '    "corners": {"over_8_5": number, "under_8_5": number}\n'
            "  },\n"
            '  "confidence": number,\n'
            '  "summary": "short text",\n'
            '  "key_factors": ["text1", "text2", "text3"]\n'
            "}\n\n"
            f"MATCH_DATA:\n{json.dumps(match, ensure_ascii=True)}\n\n"
            f"BASELINE:\n{json.dumps(baseline, ensure_ascii=True)}\n"
        )

        response = self._client.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        content = getattr(response, "text", "") or ""
        payload = self._extract_json(content)
        return self._coerce_payload(payload)

    def _resolve_model_name(self, preferred_model: str) -> str:
        if not self._genai:
            return preferred_model

        available_models: List[str] = []
        try:
            for model in self._genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" not in methods:
                    continue
                model_name = str(getattr(model, "name", "")).strip()
                if model_name:
                    available_models.append(model_name)
        except Exception as exc:  # pragma: no cover - network/runtime
            LOGGER.warning("Gemini model listing failed; using configured model '%s': %s", preferred_model, exc)
            return preferred_model

        if not available_models:
            return preferred_model

        # First, try exact matches with or without "models/" prefix.
        normalized_target = preferred_model.replace("models/", "")
        for model_name in available_models:
            if model_name == preferred_model or model_name.replace("models/", "") == normalized_target:
                return model_name

        # Then prefer 1.5 Pro variants as requested, followed by stable alternatives.
        priority_tokens = [
            "gemini-1.5-pro",
            "gemini-2.5-pro",
            "gemini-2.0-pro",
            "gemini-pro",
        ]
        for token in priority_tokens:
            for model_name in available_models:
                if token in model_name:
                    return model_name

        return available_models[0]

    @staticmethod
    def _should_retry_with_fallback_model(exc: Exception) -> bool:
        message = str(exc).lower()
        retry_tokens = (
            "is not found",
            "not supported",
            "model",
            "404",
        )
        return any(token in message for token in retry_tokens)

    def _reinitialize_with_fallback_model(self) -> bool:
        if not self._genai:
            return False

        previous_model = self.model_name
        fallback_candidates = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
            "gemini-2.5-pro",
            "gemini-2.0-pro",
            "gemini-pro",
        ]

        for candidate in fallback_candidates:
            resolved = self._resolve_model_name(candidate)
            if resolved.replace("models/", "") == previous_model.replace("models/", ""):
                continue
            try:
                self._client = self._genai.GenerativeModel(resolved)
                self.model_name = resolved
                LOGGER.info("Gemini fallback model selected: %s", self.model_name)
                return True
            except Exception:
                continue
        return False

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Gemini returned empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in Gemini response.")
        return json.loads(match.group(0))

    def _coerce_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        probabilities = payload.get("probabilities", {})
        safe_payload = {
            "probabilities": {
                "over_under_2_5": _normalize_percentages(probabilities.get("over_under_2_5", {})),
                "btts": _normalize_percentages(probabilities.get("btts", {})),
                "match_result": _normalize_percentages(probabilities.get("match_result", {})),
                "first_half_over_0_5": _normalize_percentages(
                    probabilities.get("first_half_over_0_5", {})
                ),
                "first_half_result": _normalize_percentages(probabilities.get("first_half_result", {})),
                "second_half_result": _normalize_percentages(probabilities.get("second_half_result", {})),
                "corners": _normalize_percentages(probabilities.get("corners", {})),
            },
            "confidence": _clamp(_safe_float(payload.get("confidence"), 60.0), 0.0, 100.0),
            "summary": str(payload.get("summary", "Gemini analysis completed.")),
            "key_factors": [str(item) for item in payload.get("key_factors", [])][:8],
        }
        return safe_payload

    def _merge_predictions(
        self, baseline: Dict[str, Any], gemini_payload: Dict[str, Any], weight: float = 0.55
    ) -> Dict[str, Any]:
        merged_probabilities: Dict[str, Dict[str, float]] = {}
        for market, baseline_outcomes in baseline.get("probabilities", {}).items():
            gemini_outcomes = gemini_payload.get("probabilities", {}).get(market, {})
            merged_outcomes: Dict[str, float] = {}
            for outcome, baseline_value in baseline_outcomes.items():
                gemini_value = _safe_float(gemini_outcomes.get(outcome), baseline_value)
                merged_outcomes[outcome] = (
                    (1.0 - weight) * _safe_float(baseline_value) + weight * gemini_value
                )
            merged_probabilities[market] = _normalize_percentages(merged_outcomes)

        market, outcome, probability = _top_pick(merged_probabilities)
        confidence = (
            (1.0 - weight) * _safe_float(baseline.get("confidence"), 60.0)
            + weight * _safe_float(gemini_payload.get("confidence"), 60.0)
        )
        confidence = round(_clamp(confidence, 0.0, 100.0), 2)

        return {
            "probabilities": merged_probabilities,
            "confidence": confidence,
            "top_pick": {"market": market, "outcome": outcome, "probability": probability},
            "summary": gemini_payload.get("summary") or baseline.get("summary"),
            "key_factors": gemini_payload.get("key_factors") or baseline.get("key_factors"),
        }


def flatten_probabilities(probabilities: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    flat: List[Tuple[str, str, float]] = []
    for market, outcomes in probabilities.items():
        for outcome, probability in outcomes.items():
            flat.append((market, outcome, _safe_float(probability)))
    return flat
