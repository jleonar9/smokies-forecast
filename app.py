"""Smokies Visitation Forecast - Streamlit app.

Run:  streamlit run app.py
Requires the outputs of prepare_data.py, train.py and weather_effects.py (see README).
"""
import json
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
PROC = ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src"))
from events import EVENTS, PARKING_TAG_SOURCE  # noqa: E402

st.set_page_config(page_title="Smokies Visitation Forecast", page_icon="🏔️", layout="wide")
LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))


@st.cache_data
def load():
    hist = pd.read_csv(PROC / "grsm_monthly.csv", parse_dates=["date"])
    fc = pd.read_csv(PROC / "forecast.csv", parse_dates=["date"])
    bt = pd.read_csv(PROC / "backtest.csv", parse_dates=["date"])
    metrics = pd.read_csv(PROC / "metrics.csv")
    effects = pd.read_csv(PROC / "weather_effects.csv")
    misses = pd.read_csv(PROC / "biggest_misses.csv")
    summary = json.loads((ROOT / "reports/summary.json").read_text())
    wx = json.loads((ROOT / "reports/weather_summary.json").read_text())
    return hist, fc, bt, metrics, effects, misses, summary, wx


hist, fc, bt, metrics, effects, misses, summary, wx = load()
best = summary["best_model"]
cmp = summary["comparisons"]

st.title("Great Smoky Mountains National Park: Visitation Forecast")
st.caption(
    f"Monthly recreation visits, Jan 1979 to {pd.Timestamp(summary['data_through']):%b %Y} "
    "(NPS Visitor Use Statistics), with weather (Open-Meteo) and park events. 2026 figures are preliminary."
)

# ---- KPIs ----
last12 = hist.tail(12).visits.sum()
prev12 = hist.iloc[-24:-12].visits.sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last 12 months (actual)", f"{last12/1e6:.2f}M", f"{(last12/prev12-1)*100:+.1f}% vs prior 12")
c2.metric("Next 12 months (forecast)", f"{fc.forecast.sum()/1e6:.2f}M", f"{(fc.forecast.sum()/last12-1)*100:+.1f}%")
peak = fc.loc[fc.forecast.idxmax()]
c3.metric("Busiest forecast month", f"{peak.date:%b %Y}", f"{peak.forecast/1e6:.2f}M visits", delta_color="off")
c4.metric("Best model error (MAPE)", f"{metrics.iloc[0].mape_all_years:.1f}%", best, delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Model comparison", "Weather & events", "Seasonality"])

# ---- Forecast ----
with tab1:
    years = st.slider("Years of history to show", 2, 20, 6)
    h = hist[hist.date >= hist.date.max() - pd.DateOffset(years=years)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fc.date) + list(fc.date[::-1]), y=list(fc.high_80) + list(fc.low_80[::-1]),
        fill="toself", fillcolor="rgba(46,125,50,0.15)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h.date, y=h.visits, name="Actual", line=dict(color="#37474F")))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.forecast, name=f"Forecast ({best})", line=dict(color="#2E7D32", dash="dash")))
    fig.update_layout(height=430, yaxis_title="Monthly visits", hovermode="x unified", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    table = fc.assign(Month=fc.date.dt.strftime("%b %Y"))[["Month", "forecast", "low_80", "high_80"]]
    table = table.rename(columns={"forecast": "Forecast", "low_80": "Low (80%)", "high_80": "High (80%)"})
    st.dataframe(table.style.format({c: "{:,.0f}" for c in ["Forecast", "Low (80%)", "High (80%)"]}),
                 hide_index=True, use_container_width=True)
    tot = summary["all_model_totals_12m"]
    st.caption(
        "The 80% range comes from the best model's actual backtest errors, not an assumed distribution. "
        f"Because the top models are statistically tied, note the spread: the eight models' 12-month totals range from "
        f"{min(tot.values())/1e6:.2f}M to {max(tot.values())/1e6:.2f}M.")

# ---- Model comparison ----
with tab2:
    st.subheader("Walk-forward backtest")
    st.write(
        "Each model was refit at the start of every year from Oct 2016 to Oct 2025 and asked to forecast the next "
        "12 months, the same way it would be used in practice. Months disrupted by the 2013 shutdown, 2016 wildfire, "
        "COVID closure and Hurricane Helene are excluded from scoring.")
    m = metrics.rename(columns={
        "model": "Model", "mape_all_years": "MAPE, all years (%)", "mape_last_24m": "MAPE, last 24 months (%)",
        "annual_total_error_pct": "Annual total error (%)", "months_scored": "Months scored", "note": "How it works"})
    st.dataframe(m, hide_index=True, use_container_width=True)
    g, c = cmp["best_vs_seasonal_naive"], cmp["calendar_vs_no_calendar_regression"]
    st.info(
        f"**Honest finding:** the best model beats 'same month last year' by {g['gain_points']:.2f} points of MAPE "
        f"(95% interval {g['ci95'][0]:+.2f} to {g['ci95'][1]:+.2f}). That interval includes zero, so with 10 years "
        f"of backtests the edge isn't statistically clear. Calendar features (Easter, weekends, parking tags) add "
        f"{c['gain_points']:.2f} points (95% interval {c['ci95'][0]:+.2f} to {c['ci95'][1]:+.2f}), also not clearly "
        "different from zero. Seasonality does most of the work.")
    choice = st.selectbox("Compare a model against actuals", metrics.model.tolist())
    b = bt[bt.model == choice]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=b.date, y=b.actual, name="Actual", line=dict(color="#37474F")))
    fig2.add_trace(go.Scatter(x=b.date, y=b.forecast, name=choice, line=dict(color="#1565C0", dash="dash")))
    fig2.update_layout(height=380, yaxis_title="Monthly visits", hovermode="x unified", **LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

# ---- Weather & events ----
with tab3:
    st.subheader("How weather moves visits")
    st.write(
        "Next year's weather is unknown when a 12-month forecast is made, so weather can't go into the forecast. "
        "Instead it's used to **explain** past surprises. Each bar shows the estimated change in a month's visits "
        "when that month's weather is unusual for the time of year, holding everything else equal.")
    e = effects.iloc[::-1]
    colors = ["#2E7D32" if (ce and v > 0) else ("#C62828" if ce else "#B0BEC5")
              for ce, v in zip(e.clear_effect, e.effect_pct)]
    fig5 = go.Figure(go.Bar(
        x=e.effect_pct, y=e.factor, orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False, array=e.high_95 - e.effect_pct, arrayminus=e.effect_pct - e.low_95),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    fig5.update_layout(height=320, xaxis_title="Change in monthly visits (%), with 95% interval",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Grey bars: the 95% interval crosses zero, so the effect isn't clear. "
               "Intervals come from resampling whole years (block bootstrap).")
    st.write(
        f"**If we had known each month's actual weather in advance**, the backtest error would have dropped from "
        f"{wx['mape_without_weather']:.2f}% to {wx['mape_with_actual_weather']:.2f}% "
        f"(95% interval for the gain: {wx['mape_gain_ci95'][0]:+.2f} to {wx['mape_gain_ci95'][1]:+.2f} points). "
        "So weather is real but modest. Most of the remaining misses come from things not in the data.")

    st.subheader("Biggest forecast misses")
    mm = misses.rename(columns={"date": "Month", "actual": "Actual", "forecast": "Forecast",
                                "surprise_pct": "Actual vs forecast (%)", "tmean_anom": "Temp vs normal (F)",
                                "precip_anom": "Rain vs normal (in)", "snow_anom": "Snow vs normal (in)"})
    st.dataframe(mm[["Month", "Actual", "Forecast", "Actual vs forecast (%)", "Temp vs normal (F)",
                     "Rain vs normal (in)", "Snow vs normal (in)"]]
                 .style.format({"Actual": "{:,.0f}", "Forecast": "{:,.0f}", "Actual vs forecast (%)": "{:+.1f}",
                                "Temp vs normal (F)": "{:+.1f}", "Rain vs normal (in)": "{:+.2f}",
                                "Snow vs normal (in)": "{:+.2f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Six of the seven biggest upside surprises were at least 1.7 F warmer than normal; the biggest downside miss (Jan 2025) was 4.5 F colder. Oct-Nov 2020 also reflect the post-COVID outdoor travel surge.")

    st.subheader("Park events in the data")
    ev = pd.DataFrame([{"Event": e["label"], "Months flagged": ", ".join(e["months"]), "Source": e["source"]}
                       for e in EVENTS])
    ev.loc[len(ev)] = {"Event": "Park It Forward parking tags required from Mar 1, 2023 (used as a forecast feature)",
                       "Months flagged": "Mar 2023 onward", "Source": PARKING_TAG_SOURCE}
    st.dataframe(ev, hide_index=True, use_container_width=True,
                 column_config={"Source": st.column_config.LinkColumn("Source")})
    st.caption("Disruption months are cleaned out of model inputs and excluded from scoring, but kept in the charts.")

# ---- Seasonality ----
with tab4:
    st.subheader("When do people visit?")
    recent = hist[(hist.date.dt.year >= hist.date.dt.year.max() - 10) & (hist.disrupted == 0)]
    by_month = recent.groupby(recent.date.dt.month).visits.mean().reset_index()
    by_month["Month"] = pd.to_datetime(by_month.date, format="%m").dt.strftime("%b")
    fig3 = go.Figure(go.Bar(x=by_month.Month, y=by_month.visits, marker_color="#2E7D32"))
    fig3.update_layout(height=360, yaxis_title="Average monthly visits (last 10 years)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    annual = hist[hist.date.dt.year < hist.date.dt.year.max()].groupby(hist.date.dt.year).visits.sum()
    fig4 = go.Figure(go.Scatter(x=annual.index, y=annual.values, mode="lines+markers", line=dict(color="#37474F")))
    fig4.update_layout(height=320, yaxis_title="Annual visits", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("July and October are the twin peaks: summer vacations and fall color. January is the low point.")

st.divider()
st.caption("Data: NPS Visitor Use Statistics (irma.nps.gov/Stats); weather: Open-Meteo historical archive "
           "(open-meteo.com), average of Gatlinburg, TN and Cherokee, NC.")
