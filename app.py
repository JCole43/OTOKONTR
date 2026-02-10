"""Streamlit dashboard for live analysis, coupons, and ROI history."""

from __future__ import annotations

from datetime import date
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import GeminiAnalyzer
from config import Settings
from history import (
    HistoryStore,
    compute_coupon_roi,
    compute_prediction_roi,
    format_coupon_message,
    generate_coupon_suggestions,
)
from scraper import MatchDataCollector
from telegram_bot import MARKET_LABELS, OUTCOME_LABELS, TelegramNotifier

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Futbol Tahmin Sistemi", layout="wide")
st.title("Profesyonel Futbol Mac Tahmin ve Analiz Sistemi")

settings = Settings.from_env()
collector = MatchDataCollector(settings)
analyzer = GeminiAnalyzer(settings=settings)
notifier = TelegramNotifier(settings=settings)
history_store = HistoryStore(settings=settings)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_rows(analyzed_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(analyzed_matches):
        match = item["match"]
        prediction = item["prediction"]
        probabilities = prediction.get("probabilities", {})
        top_pick = prediction.get("top_pick", {})
        rows.append(
            {
                "_row_idx": idx,
                "League": match.get("competition", "-"),
                "Match": f"{match.get('home_team', '-')} - {match.get('away_team', '-')}",
                "Kickoff": match.get("kickoff_utc", "-"),
                "Source": match.get("source", "-"),
                "Model": prediction.get("model", "-"),
                "Confidence": _safe_float(prediction.get("confidence", 0.0)),
                "Top Market": MARKET_LABELS.get(top_pick.get("market", ""), top_pick.get("market", "-")),
                "Top Outcome": OUTCOME_LABELS.get(top_pick.get("outcome", ""), top_pick.get("outcome", "-")),
                "Top Probability": _safe_float(top_pick.get("probability", 0.0)),
                "Over 2.5": _safe_float(probabilities.get("over_under_2_5", {}).get("over", 0.0)),
                "BTTS Yes": _safe_float(probabilities.get("btts", {}).get("yes", 0.0)),
                "MS 1": _safe_float(probabilities.get("match_result", {}).get("home", 0.0)),
                "MS X": _safe_float(probabilities.get("match_result", {}).get("draw", 0.0)),
                "MS 2": _safe_float(probabilities.get("match_result", {}).get("away", 0.0)),
            }
        )
    return rows


def _pick_selected_row_index(df_rows: pd.DataFrame) -> Optional[int]:
    if df_rows.empty:
        return None

    display_df = df_rows.drop(columns=["_row_idx"])
    selected_data_idx: Optional[int] = None
    selection_rows: List[int] = []
    try:
        table_state = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if hasattr(table_state, "selection") and hasattr(table_state.selection, "rows"):
            selection_rows = list(table_state.selection.rows)
        elif isinstance(table_state, dict):
            selection_rows = list(table_state.get("selection", {}).get("rows", []))
    except TypeError:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if selection_rows:
        row_pos = int(selection_rows[0])
        selected_data_idx = int(df_rows.iloc[row_pos]["_row_idx"])
        st.session_state["selected_match_idx"] = selected_data_idx
        return selected_data_idx

    existing_selection = st.session_state.get("selected_match_idx")
    if isinstance(existing_selection, int) and 0 <= existing_selection < len(df_rows):
        return existing_selection

    option_map = {
        f"{row['Match']} | {row['League']} | {row['Kickoff']}": int(row["_row_idx"])
        for _, row in df_rows.iterrows()
    }
    fallback_label = st.selectbox("Detay icin mac secin", options=list(option_map.keys()))
    selected_data_idx = option_map[fallback_label]
    st.session_state["selected_match_idx"] = selected_data_idx
    return selected_data_idx


def _history_predictions_df(history_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in history_data.get("predictions", []):
        result = item.get("result", {}) or {}
        rows.append(
            {
                "Created": item.get("created_at", ""),
                "Kickoff": item.get("kickoff_utc", ""),
                "Match": item.get("match_label", ""),
                "League": item.get("competition", ""),
                "Pick": item.get("pick_label", ""),
                "Probability": _safe_float(item.get("probability", 0.0)),
                "Odd": _safe_float(item.get("odd", 0.0)),
                "Status": item.get("status", ""),
                "Score": result.get("score", "-"),
                "Actual Outcome": result.get("actual_outcome", "-"),
                "Profit": _safe_float(item.get("profit", 0.0)),
                "Model": item.get("model", ""),
            }
        )
    return pd.DataFrame(rows)


def _history_coupons_df(history_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in history_data.get("coupons", []):
        rows.append(
            {
                "Created": item.get("created_at", ""),
                "Strategy": item.get("strategy", ""),
                "Legs": int(item.get("legs_count", 0)),
                "Combined Odd": _safe_float(item.get("combined_odd", 0.0)),
                "Combined Probability": _safe_float(item.get("combined_probability", 0.0)),
                "Expected ROI %": _safe_float(item.get("expected_roi_pct", 0.0)),
                "Status": item.get("status", ""),
                "Profit": _safe_float(item.get("profit", 0.0)),
                "Sent Telegram": bool(item.get("sent_to_telegram", False)),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Ayarlar")
    target_date = st.date_input("Mac Tarihi", value=date.today())
    threshold = st.slider("Telegram Esik (%)", min_value=50, max_value=95, value=80, step=1)
    max_match_count = st.slider("Maksimum analiz mac adedi", min_value=5, max_value=80, value=25, step=1)
    run_button = st.button("Veri Cek ve Analiz Et", type="primary")
    refresh_results_button = st.button("Gecmis Sonuclari Guncelle")
    st.caption("Gemini durumu: " + ("Aktif" if analyzer.gemini_enabled else "Pasif (fallback)"))
    st.caption("Telegram durumu: " + ("Aktif" if notifier.is_configured else "Pasif"))


if refresh_results_button:
    with st.spinner("Gecmis mac sonuclari ve ROI guncelleniyor..."):
        sync_stats = history_store.refresh_results(max_fixtures=80)
    st.sidebar.success(
        (
            f"Guncellendi | Tahmin: {sync_stats['prediction_updates']}, "
            f"Kupon: {sync_stats['coupon_updates']}"
        )
    )


if run_button:
    with st.spinner("Maclar toplanip analiz ediliyor..."):
        matches = collector.collect_matches(target_date=target_date)
        if len(matches) > max_match_count:
            matches = matches[:max_match_count]
        analyzed = analyzer.analyze_matches(matches)
        st.session_state["analyzed_matches"] = analyzed
        st.session_state["threshold"] = threshold
        st.session_state["target_date"] = target_date.isoformat()
        saved_count = history_store.upsert_predictions(analyzed, target_date=target_date)
    st.sidebar.success(f"Analiz tamamlandi. Kaydedilen/yenilenen tahmin: {saved_count}")


analyzed_matches: List[Dict[str, Any]] = st.session_state.get("analyzed_matches", [])
history_data = history_store.load()
prediction_roi = compute_prediction_roi(history_data)
coupon_roi = compute_coupon_roi(history_data)

combined_stake = prediction_roi["total_stake"] + coupon_roi["total_stake"]
combined_profit = prediction_roi["total_profit"] + coupon_roi["total_profit"]
combined_roi = (combined_profit / combined_stake * 100.0) if combined_stake > 0 else 0.0

roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
roi_col1.metric("Tahmin ROI", f"%{prediction_roi['roi_pct']:.2f}")
roi_col2.metric("Kupon ROI", f"%{coupon_roi['roi_pct']:.2f}")
roi_col3.metric("Toplam Kar/Zarar (u)", f"{combined_profit:.2f}")
roi_col4.metric("Genel ROI", f"%{combined_roi:.2f}")

tab_live, tab_coupon, tab_history = st.tabs(["Canli Analiz", "Kupon Motoru", "Gecmis ve ROI"])

with tab_live:
    if not analyzed_matches:
        st.info("Analiz sonucunu gormek icin soldan tarih secip 'Veri Cek ve Analiz Et' butonuna basin.")
    else:
        rows = _table_rows(analyzed_matches)
        df = pd.DataFrame(rows)

        high_confidence_count = 0
        for item in analyzed_matches:
            if notifier.find_high_confidence_outcomes(item.get("prediction", {}), threshold):
                high_confidence_count += 1

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Toplam Mac", len(analyzed_matches))
        metric_col2.metric("Yuksek Olasilikli Mac", high_confidence_count)
        metric_col3.metric("Ortalama Guven", f"%{df['Confidence'].mean():.2f}")

        st.subheader("Tahmin Tablosu (satira tiklayip detay gorun)")
        selected_idx = _pick_selected_row_index(df)

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

        if selected_idx is None or selected_idx >= len(analyzed_matches):
            st.warning("Detay icin bir mac secin.")
        else:
            selected_item = analyzed_matches[selected_idx]
            match = selected_item["match"]
            prediction = selected_item["prediction"]
            top_pick = prediction.get("top_pick", {})

            st.subheader("Secili Mac Detayi")
            detail_col1, detail_col2 = st.columns([2, 1])
            with detail_col1:
                st.markdown(
                    (
                        f"**Mac:** {match.get('home_team', '-')} - {match.get('away_team', '-')}\n\n"
                        f"**Lig:** {match.get('competition', '-')}\n\n"
                        f"**Baslangic:** {match.get('kickoff_utc', '-')}\n\n"
                        f"**Model:** {prediction.get('model', '-')}\n\n"
                        f"**Ozet:** {prediction.get('summary', '-')}"
                    )
                )
                st.write("**Ana Faktorler:**")
                for factor in prediction.get("key_factors", []):
                    st.write(f"- {factor}")
            with detail_col2:
                st.metric("Genel Guven", f"%{_safe_float(prediction.get('confidence', 0.0)):.2f}")
                st.metric(
                    "En Guclu Tahmin",
                    (
                        f"{MARKET_LABELS.get(top_pick.get('market', ''), top_pick.get('market', '-'))} / "
                        f"{OUTCOME_LABELS.get(top_pick.get('outcome', ''), top_pick.get('outcome', '-'))}"
                    ),
                )
                st.metric("Olasilik", f"%{_safe_float(top_pick.get('probability', 0.0)):.2f}")

            st.json(prediction.get("probabilities", {}))

            send_selected = st.button(
                "Secili Maci Telegram'a Gonder",
                key=f"send_selected_{match.get('fixture_id', selected_idx)}",
            )
            if send_selected:
                if not notifier.is_configured:
                    st.error("Telegram icin TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ayarlanmalidir.")
                else:
                    message = notifier.format_alert(match, prediction, threshold=threshold)
                    sent = notifier.send_message(message)
                    if sent:
                        st.success("Secili mac Telegram kanalina gonderildi.")
                    else:
                        st.error("Telegram gonderimi basarisiz oldu.")

        send_telegram = st.button("Yuksek Olasilikli Maclari Telegram'a Gonder")
        if send_telegram:
            if not notifier.is_configured:
                st.error("Telegram icin TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ayarlanmalidir.")
            else:
                with st.spinner("Telegram bildirimleri gonderiliyor..."):
                    sent_count = notifier.notify_high_confidence(analyzed_matches, threshold=threshold)
                st.success(f"Gonderilen bildirim adedi: {sent_count}")


with tab_coupon:
    st.subheader("AI Kupon Uretici")
    if not analyzed_matches:
        st.info("Kupon onerileri icin once canli analiz calistirin.")
    else:
        coupon_min_prob = st.slider(
            "Kupon icin minimum olasilik (%)",
            min_value=55,
            max_value=90,
            value=68,
            step=1,
        )
        coupon_legs_count = st.selectbox("Kombine bacak sayisi", options=[2, 3, 4, 5], index=1)

        suggestions = generate_coupon_suggestions(
            analyzed_matches=analyzed_matches,
            min_probability=float(coupon_min_prob),
            legs_count=int(coupon_legs_count),
            max_coupons=2,
        )

        if not suggestions:
            st.warning(
                (
                    "Yeterli kupon adayi yok. Esigi dusurun veya farkli tarih icin analiz calistirin."
                )
            )
        for idx, coupon in enumerate(suggestions, start=1):
            strategy_label = "Guvenli" if coupon.get("strategy") == "safe" else "Deger"
            with st.expander(
                f"Kupon {idx} | {strategy_label} | Oran {coupon.get('combined_odd', 0.0):.2f}",
                expanded=(idx == 1),
            ):
                m1, m2, m3 = st.columns(3)
                m1.metric("Kombine Oran", f"{_safe_float(coupon.get('combined_odd', 0.0)):.2f}")
                m2.metric(
                    "Tahmini Basari",
                    f"%{_safe_float(coupon.get('combined_probability', 0.0)):.2f}",
                )
                m3.metric("Beklenen ROI", f"%{_safe_float(coupon.get('expected_roi_pct', 0.0)):.2f}")

                legs_df = pd.DataFrame(
                    [
                        {
                            "Match": leg.get("match_label", ""),
                            "Pick": leg.get("pick_label", ""),
                            "Probability": _safe_float(leg.get("probability", 0.0)),
                            "Odd": _safe_float(leg.get("odd", 0.0)),
                        }
                        for leg in coupon.get("legs", [])
                    ]
                )
                st.dataframe(legs_df, use_container_width=True, hide_index=True)

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button(
                        "Kuponu Kaydet",
                        key=f"coupon_save_{idx}_{coupon.get('id')}",
                    ):
                        saved_coupon_id = history_store.add_coupon(coupon, sent_to_telegram=False)
                        st.success(f"Kupon kaydedildi. ID: {saved_coupon_id}")
                        st.rerun()
                with action_col2:
                    if st.button(
                        "Kuponu Telegram'a Gonder",
                        key=f"coupon_send_{idx}_{coupon.get('id')}",
                    ):
                        if not notifier.is_configured:
                            st.error("Telegram ayarlari eksik.")
                        else:
                            sent = notifier.send_message(format_coupon_message(coupon))
                            if sent:
                                saved_coupon_id = history_store.add_coupon(coupon, sent_to_telegram=True)
                                st.success(f"Kupon Telegram'a gonderildi ve kaydedildi. ID: {saved_coupon_id}")
                                st.rerun()
                            else:
                                st.error("Kupon Telegram'a gonderilemedi.")

        pending_coupons = [item for item in history_data.get("coupons", []) if item.get("status") == "pending"]
        st.markdown(f"**Aktif (pending) kayitli kupon sayisi:** {len(pending_coupons)}")


with tab_history:
    st.subheader("Gecmis Tahminler ve ROI")
    if st.button("Gecmis Sonuclari Simdi Guncelle", key="refresh_history_tab"):
        with st.spinner("Sonuclar cekiliyor, ROI hesaplanıyor..."):
            sync_stats = history_store.refresh_results(max_fixtures=120)
        st.success(
            f"Guncellendi | Tahmin: {sync_stats['prediction_updates']}, Kupon: {sync_stats['coupon_updates']}"
        )
        st.rerun()

    hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
    hist_col1.metric("Tahmin Win Rate", f"%{prediction_roi['win_rate_pct']:.2f}")
    hist_col2.metric("Tahmin ROI", f"%{prediction_roi['roi_pct']:.2f}")
    hist_col3.metric("Kupon Win Rate", f"%{coupon_roi['win_rate_pct']:.2f}")
    hist_col4.metric("Kupon ROI", f"%{coupon_roi['roi_pct']:.2f}")

    st.markdown("### Gecmis Mac Tahminleri")
    prediction_history_df = _history_predictions_df(history_data)
    if prediction_history_df.empty:
        st.info("Henuz kayitli tahmin gecmisi yok.")
    else:
        status_options = sorted(prediction_history_df["Status"].dropna().unique().tolist())
        selected_statuses = st.multiselect(
            "Durum filtrele",
            options=status_options,
            default=status_options,
        )
        filtered_df = prediction_history_df[
            prediction_history_df["Status"].isin(selected_statuses)
        ].copy()
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.markdown("### Gecmis Kuponlar")
    coupon_history_df = _history_coupons_df(history_data)
    if coupon_history_df.empty:
        st.info("Henuz kayitli kupon gecmisi yok.")
    else:
        st.dataframe(coupon_history_df, use_container_width=True, hide_index=True)
