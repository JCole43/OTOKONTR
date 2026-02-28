# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Football Match Prediction and Analysis System — a Python/Streamlit application that collects football match data, runs statistical + optional Gemini AI analysis, and presents predictions via a web dashboard with Telegram notifications. See `README.md` for full feature details.

### Running the application

- **Streamlit dashboard (dev):** `source .venv/bin/activate && streamlit run app.py --server.port 8501 --server.headless true`
- **CLI pipeline:** `source .venv/bin/activate && python run_pipeline.py --date YYYY-MM-DD`
- Streamlit is the primary dev interface; no build step required.

### Key caveats

- **No automated tests exist** in this repository — there is no test suite to run. Validation is done by running the application directly.
- **No linter/formatter is configured** — there are no `pyproject.toml`, `setup.cfg`, or linting configs. You can run `python -m py_compile <file>` to verify syntax.
- The app works fully in **statistical-fallback mode** without any API keys. With `FOOTBALL_API_KEY` / `API_SPORTS_KEY` set, real match data is fetched; with `GEMINI_API_KEY`, hybrid AI predictions are enabled.
- The `.env` file is loaded by `config.py` via `python-dotenv`; if `.env` doesn't exist, `.env.example` is used as fallback.
- History is persisted in `data/history.json` (created automatically on first write).
- `python3.12-venv` system package is needed to create the virtualenv (not installed by default in the base image).
