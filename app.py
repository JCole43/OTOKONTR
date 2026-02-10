"""Streamlit dashboard for football prediction workflow."""

from __future__ import annotations

from datetime import date
import logging
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import GeminiAnalyzer
from config import Settings
from scraper import MatchDataCollector
from telegram_bot import MARKET_LABELS, OUTCOME_LABELS, TelegramNotifier

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Futbol Tahmin Sistemi", layout="wide")
st.title("Profesyonel Futbol Mac Tahmin ve Analiz Sistemi")

settings = Settings.from_env()
collector = MatchDataCollector(settings)
analyzer = GeminiAnalyzer(settings=settings)
notifier = TelegramNotifier(settings=settings)


def _table_rows(analyzed_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in analyzed_matches:
        match = item["match"]
        prediction = item["prediction"]
        probabilities = prediction.get("probabilities", {})
        top_pick = prediction.get("top_pick", {})

        rows.append(
            {
                "League": match.get("competition", "-"),
                "Match": f"{match.get('home_team', '-')} - {match.get('away_team', '-')}",
                "Kickoff": match.get("kickoff_utc", "-"),
                "Source": match.get("source", "-"),
                "Model": prediction.get("model", "-"),
                "Confidence": prediction.get("confidence", 0.0),
                "Top Market": MARKET_LABELS.get(top_pick.get("market", ""), top_pick.get("market", "-")),
                "Top Outcome": OUTCOME_LABELS.get(top_pick.get("outcome", ""), top_pick.get("outcome", "-")),
                "Top Probability": top_pick.get("probability", 0.0),
                "Over 2.5": probabilities.get("over_under_2_5", {}).get("over", 0.0),
                "BTTS Yes": probabilities.get("btts", {}).get("yes", 0.0),
                "MS 1": probabilities.get("match_result", {}).get("home", 0.0),
                "MS X": probabilities.get("match_result", {}).get("draw", 0.0),
                "MS 2": probabilities.get("match_result", {}).get("away", 0.0),
            }
        )
    return rows


with st.sidebar:
    st.header("Ayarlar")
    target_date = st.date_input("Mac Tarihi", value=date.today())
    threshold = st.slider("Telegram Esik (%)", min_value=50, max_value=95, value=80, step=1)
    run_button = st.button("Veri Cek ve Analiz Et", type="primary")
    st.caption("Gemini durumu: " + ("Aktif" if analyzer.gemini_enabled else "Pasif (fallback)"))
    st.caption("Telegram durumu: " + ("Aktif" if notifier.is_configured else "Pasif"))


if run_button:
    with st.spinner("Maclar toplanip analiz ediliyor..."):
        matches = collector.collect_matches(target_date=target_date)
        analyzed = analyzer.analyze_matches(matches)
        st.session_state["analyzed_matches"] = analyzed
        st.session_state["threshold"] = threshold


analyzed_matches: List[Dict[str, Any]] = st.session_state.get("analyzed_matches", [])

if not analyzed_matches:
    st.info("Analiz sonucunu gormek icin soldan tarih secip 'Veri Cek ve Analiz Et' butonuna basin.")
    st.stop()


rows = _table_rows(analyzed_matches)
df = pd.DataFrame(rows)

high_confidence_count = 0
for item in analyzed_matches:
    if notifier.find_high_confidence_outcomes(item.get("prediction", {}), threshold):
        high_confidence_count += 1

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Toplam Mac", len(analyzed_matches))
metric_col2.metric("Yuksek Olasilikli Mac", high_confidence_count)
metric_col3.metric("Ortalama Guven", round(df["Confidence"].mean(), 2))

st.subheader("Tahmin Tablosu")
st.dataframe(df, use_container_width=True, hide_index=True)

chart_df = df.sort_values("Top Probability", ascending=False).head(20)
fig = px.bar(
    chart_df,
    x="Match",
    y="Top Probability",
    color="Top Market",
    title="Mac Bazli En Guclu Tahminler",
)
fig.update_layout(xaxis_title=None, yaxis_title="Olasilik (%)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Mac Bazli Detaylar")
for item in analyzed_matches:
    match = item["match"]
    prediction = item["prediction"]
    title = f"{match.get('home_team', '-')}-{match.get('away_team', '-')} | {match.get('competition', '-')}"
    with st.expander(title):
        st.write(f"**Kaynak:** {match.get('source', '-')}")
        st.write(f"**Model:** {prediction.get('model', '-')}")
        st.write(f"**Guven:** %{prediction.get('confidence', 0.0):.2f}")
        st.write(f"**Ozet:** {prediction.get('summary', '-')}")
        st.write("**Faktorler:**")
        for factor in prediction.get("key_factors", []):
            st.write(f"- {factor}")
        st.json(prediction.get("probabilities", {}))

send_telegram = st.button("Yuksek Olasilikli Maclari Telegram'a Gonder")
if send_telegram:
    if not notifier.is_configured:
        st.error("Telegram icin TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ayarlanmalidir.")
    else:
        with st.spinner("Telegram bildirimleri gonderiliyor..."):
            sent_count = notifier.notify_high_confidence(analyzed_matches, threshold=threshold)
        st.success(f"Gonderilen bildirim adedi: {sent_count}")
