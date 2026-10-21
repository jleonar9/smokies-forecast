"""Backtest every model, pick the best, and write a 12-month forecast.

Usage:  python src/train.py
Outputs go to data/processed/ and reports/.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from models import ALL_MODELS, forecast_future, load, mape, rolling_backtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, REPORTS = ROOT / "data/processed", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = load(PROC / "grsm_monthly.csv")

# Walk-forward backtest: 10 one-year blocks, Oct 2016 - Sep 2026
bt = rolling_backtest(df, "2016-10-01", "2025-10-01")

bt["disrupted"] = bt["date"].map(df["disrupted"]).astype(int)
bt.to_csv(PROC / "backtest.csv", index=False)

scored = bt[bt.disrupted == 0]
recent_cut = df.index.max() - pd.DateOffset(months=23)
metrics = []
for name, g in scored.groupby("model", sort=False):
    r = g[g.date >= recent_cut]
    metrics.append({
        "model": name,
        "mape_all_years": round(mape(g.actual, g.forecast), 2),
        "mape_last_24m": round(mape(r.actual, r.forecast), 2),
        "annual_total_error_pct": round(
            float(g.groupby(g.date.dt.year).apply(lambda x: abs(x.forecast.sum() / x.actual.sum() - 1)).mean() * 100), 2),
        "months_scored": int(len(g)),
    })
# Rank by all-years MAPE; break ties on the most recent 24 months
metrics = pd.DataFrame(metrics).sort_values(["mape_all_years", "mape_last_24m"]).reset_index(drop=True)
metrics["note"] = metrics.model.map({M.name: M.note for M in ALL_MODELS})
metrics.to_csv(PROC / "metrics.csv", index=False)

best_name = metrics.iloc[0]["model"]
best_cls = next(M for M in ALL_MODELS if M.name == best_name)
best_bt = scored[scored.model == best_name]
resid_logs = np.log(best_bt.actual / best_bt.forecast).values
fc = forecast_future(df, best_cls, 12, resid_logs)
fc.to_csv(PROC / "forecast.csv", index=False)

# Is the best model's edge over the naive baseline (and calendar over no calendar) real?
def paired_gain(a_name, b_name, seed=0):
    """Mean drop in absolute % error going from model a to model b, with a 95% bootstrap interval."""
    a = scored[scored.model == a_name].set_index("date")
    b = scored[scored.model == b_name].set_index("date")
    gain = ((a.forecast / a.actual - 1).abs() - (b.forecast / b.actual - 1).abs()) * 100
    rng = np.random.default_rng(seed)
    boots = [gain.sample(len(gain), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    return {"gain_points": round(float(gain.mean()), 2),
            "ci95": [round(float(v), 2) for v in np.percentile(boots, [2.5, 97.5])]}


comparisons = {
    "best_vs_seasonal_naive": paired_gain("Seasonal naive", best_name),
    "calendar_vs_no_calendar_regression": paired_gain("Seasonal regression", "Seasonal regression + calendar"),
}

# Every model's 12-month total, to show how much the choice of model matters
all_totals = {M.name: int(forecast_future(df, M, 12).forecast.sum()) for M in ALL_MODELS}

summary = {
    "best_model": best_name,
    "all_model_totals_12m": all_totals,
    "comparisons": comparisons,
    "data_through": f"{df.index.max():%Y-%m}",
    "backtest": "walk-forward, 10 annual blocks Oct 2016 - Sep 2026; disrupted months (2013 shutdown, 2016 wildfire, COVID, Helene) not scored",
    "forecast_total_12m": int(fc.forecast.sum()),
    "metrics": metrics.to_dict(orient="records"),
}
(REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
print(metrics.to_string(index=False))
print(f"\nBest: {best_name}. Next 12 months forecast: {fc.forecast.sum():,.0f} visits")
print(fc.round({"forecast": 0, "low_80": 0, "high_80": 0}).to_string(index=False))
