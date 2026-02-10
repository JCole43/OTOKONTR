# Football Match Prediction and Analysis System

This project provides a modular Python workflow for football match analytics and betting-market probability estimation.

## Features

- **Data ingestion**
  - Attempts to collect match cards from:
    - `Nesine.com`
    - `Iddaa.com`
  - If scraping returns no usable records, it falls back to **API-Sports** (`api-football`) when configured.
- **Analysis engine**
  - Statistical baseline model (form + xG proxy + Poisson probabilities + odds priors).
  - Optional Gemini 1.5 Pro calibration (`GEMINI_API_KEY`) for hybrid predictions.
- **Prediction markets**
  - 2.5 Over/Under
  - BTTS (KG Var/Yok)
  - Match Result (1-X-2)
  - First Half Over 0.5
  - First Half Result
  - Second Half Result
  - Corners Over/Under 8.5
- **Dashboard**
  - Streamlit UI with:
    - clickable match selection and detailed match panel
    - one-click per-match Telegram notification
    - charts and match-level explanations
- **Coupon engine**
  - Builds AI combined tickets from highest-confidence picks.
  - Supports saving coupons and sending coupon messages to Telegram.
- **History and ROI tracking**
  - Stores past predictions and coupons in local JSON history.
  - Fetches completed match results and settles predictions.
  - Calculates win rate, P/L and ROI for both picks and coupons.
- **Automation**
  - Threshold-based Telegram notifications (for example `%80+`).
  - CLI runner for cron / scheduler usage.

## File Structure

- `scraper.py` -> Source collectors and fallback strategy
- `analyzer.py` -> Statistical + Gemini hybrid analysis engine
- `telegram_bot.py` -> Telegram notifier and alert formatting
- `app.py` -> Streamlit dashboard
- `run_pipeline.py` -> Command-line automation entrypoint
- `config.py` -> `.env` loader
- `history.py` -> Prediction history, coupon and ROI settlement engine

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template and fill secrets:

```bash
cp .env.example .env
```

The app first reads `.env`, and if not found it also falls back to `.env.example`.

Required keys:
- `GEMINI_API_KEY` (optional, statistical fallback works without it)
- `API_SPORTS_KEY` or `FOOTBALL_API_KEY` (recommended for reliable match/stat data fallback)
- `API_SPORTS_HOST` / `API_SPORTS_URL` OR `FOOTBALL_API_HOST` / `FOOTBALL_API_URL` (defaults to `v3.football.api-sports.io`)
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (for notifications)

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

## Run Automation CLI

```bash
python run_pipeline.py --date 2026-02-10 --threshold 80 --send-telegram
```

Single demo Telegram message:

```bash
python run_pipeline.py --date 2026-02-10 --send-telegram-demo
```

## Notes

- Direct scraping may break because source sites can change HTML or apply anti-bot protection.
- The collector is implemented to try web scraping first and then use API-Sports fallback.
- xG values are approximated when direct xG feeds are not available from the active source.
- History data is persisted in `data/history.json`.
