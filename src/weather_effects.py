"""How much does weather explain? (Explanation, not forecasting.)

Future weather is unknown when a 12-month forecast is made, so weather cannot go
into the forecast. Instead this script asks two questions:

1. If we HAD known each month's weather, how much better would the backtest have been?
   (An upper bound on what weather knowledge is worth.)
2. What is the effect of unusual weather on visits?
   e.g. one extra heavy-rain day, a month 5 F warmer than normal, 1 inch more snow.

Weather is expressed as anomalies: departure from the 1991-2020 normal for that
calendar month, so "July is hot" isn't confused with "this July was unusually hot".
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models import SeasonalRidgeCal, _features, clean_series, load, mape

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, REPORTS = ROOT / "data/processed", ROOT / "reports"

WX = ["tmean_f", "precip_in", "heavy_rain_days", "snow_in", "hot_days"]
LABELS = {
    "tmean_anom": "Temperature, +5 F vs normal",
    "precip_anom": "Rainfall, +1 inch vs normal",
    "heavy_rain_anom": "Heavy-rain days (1+ inch), +1 day",
    "snow_anom": "Snowfall, +1 inch vs normal",
    "hot_anom": "Hot days (90 F+), +1 day vs normal",
}
SCALE = {"tmean_anom": 5.0, "precip_anom": 1.0, "heavy_rain_anom": 1.0, "snow_anom": 1.0, "hot_anom": 1.0}


def anomalies(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df.index.year >= 1991) & (df.index.year <= 2020)]
    normals = base.groupby(base.index.month)[WX].mean()
    norm = normals.loc[df.index.month].set_index(df.index)
    return pd.DataFrame({
        "tmean_anom": df.tmean_f - norm.tmean_f,
        "precip_anom": df.precip_in - norm.precip_in,
        "heavy_rain_anom": df.heavy_rain_days - norm.heavy_rain_days,
        "snow_anom": df.snow_in - norm.snow_in,
        "hot_anom": df.hot_days - norm.hot_days,
    }, index=df.index)


ANOM_COLS = list(LABELS)


class SeasonalRidgeCalWeather(SeasonalRidgeCal):
    """Same as the calendar model, plus the ACTUAL weather of the target month.
    Not usable for real forecasts; it measures what perfect weather knowledge is worth."""
    name = "Seasonal regression + calendar + actual weather"
    anom: pd.DataFrame = None

    def _design(self, X):
        d = super()._design(X)
        return d.join(self.anom.reindex(X.index))

    def fit(self, y):
        self.y = y
        X = _features(y, y.index).dropna()
        Xd = self._design(X).dropna()
        self.dcols = Xd.columns
        self.m = Ridge(alpha=1.0).fit(Xd, np.log(y.loc[Xd.index]))
        return self


def main():
    df = load(PROC / "grsm_monthly.csv")
    anom = anomalies(df)
    SeasonalRidgeCalWeather.anom = anom
    y_clean, y_raw = clean_series(df), df.visits.astype(float)

    # ---- 1. Walk-forward backtest with and without actual weather ----
    rows = []
    for s in pd.date_range("2016-10-01", "2025-10-01", freq="12MS"):
        block = pd.date_range(s, periods=12, freq="MS")
        block = block[block <= y_raw.index.max()]
        train = y_clean[y_clean.index < s]
        for M in (SeasonalRidgeCal, SeasonalRidgeCalWeather):
            p = M().fit(train).predict(block)
            rows += [{"date": d, "model": M.name, "actual": y_raw[d], "forecast": p[d]} for d in block]
    bt = pd.DataFrame(rows)
    bt["disrupted"] = bt.date.map(df.disrupted)
    bt = bt[bt.disrupted == 0]
    no_wx = bt[bt.model == SeasonalRidgeCal.name].set_index("date")
    with_wx = bt[bt.model == SeasonalRidgeCalWeather.name].set_index("date")
    ape_a = (no_wx.forecast / no_wx.actual - 1).abs() * 100
    ape_b = (with_wx.forecast / with_wx.actual - 1).abs() * 100
    gain = ape_a - ape_b
    rng = np.random.default_rng(0)
    boots = [gain.sample(len(gain), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]

    # ---- 2. Effect sizes: full-history fit, block bootstrap by year for intervals ----
    X = _features(y_clean, y_clean.index).dropna()
    model = SeasonalRidgeCalWeather()
    Xd = model._design(X).dropna()
    target = np.log(y_clean.loc[Xd.index])
    keep = df.disrupted.loc[Xd.index] == 0
    Xd, target = Xd[keep], target[keep]
    coef = pd.Series(Ridge(alpha=1.0).fit(Xd, target).coef_, index=Xd.columns)[ANOM_COLS]
    years = Xd.index.year.unique()
    boot = []
    for _ in range(500):
        pick = rng.choice(years, len(years), replace=True)
        idx = np.concatenate([np.where(Xd.index.year == yy)[0] for yy in pick])
        boot.append(Ridge(alpha=1.0).fit(Xd.iloc[idx], target.iloc[idx]).coef_[[Xd.columns.get_loc(c) for c in ANOM_COLS]])
    boot = np.array(boot)
    effects = pd.DataFrame({
        "factor": [LABELS[c] for c in ANOM_COLS],
        "effect_pct": [(np.exp(coef[c] * SCALE[c]) - 1) * 100 for c in ANOM_COLS],
        "low_95": [(np.exp(np.percentile(boot[:, i], 2.5) * SCALE[c]) - 1) * 100 for i, c in enumerate(ANOM_COLS)],
        "high_95": [(np.exp(np.percentile(boot[:, i], 97.5) * SCALE[c]) - 1) * 100 for i, c in enumerate(ANOM_COLS)],
    })
    effects["clear_effect"] = (np.sign(effects.low_95) == np.sign(effects.high_95)) & (effects.low_95 != 0)
    effects = effects.round(2)
    effects.to_csv(PROC / "weather_effects.csv", index=False)

    # biggest misses and their weather
    # surprise_pct > 0: more visitors than forecast
    miss = no_wx.assign(surprise_pct=(no_wx.actual / no_wx.forecast - 1) * 100).join(anom)
    miss = miss.reindex(miss.surprise_pct.abs().sort_values(ascending=False).index).head(8)
    miss = miss.reset_index()
    miss["date"] = miss.date.dt.strftime("%Y-%m")
    miss.round(2).to_csv(PROC / "biggest_misses.csv", index=False)

    out = {
        "mape_without_weather": round(ape_a.mean(), 2),
        "mape_with_actual_weather": round(ape_b.mean(), 2),
        "mape_gain_points": round(gain.mean(), 2),
        "mape_gain_ci95": [round(float(v), 2) for v in np.percentile(boots, [2.5, 97.5])],
        "months_improved_pct": round(float((gain > 0).mean() * 100), 1),
        "months_scored": int(len(gain)),
    }
    (REPORTS / "weather_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(effects.to_string(index=False))
    print(miss[["date", "actual", "forecast", "surprise_pct"] + ANOM_COLS].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
