"""Forecasting models for monthly Great Smoky Mountains NP visitation.

Every model forecasts up to 12 months ahead using only information available at
forecast time: visit lags of 12+ months, plus calendar features (Easter timing,
weekend days, the 2023 parking-tag policy) that are known years in advance.
Weather is NOT used for forecasting because future weather is unknown; see
weather_effects.py for how weather explains forecast misses.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from events import calendar_features  # noqa: E402

CAL_COLS = ["weekend_days", "easter_in_month", "parking_tag"]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").asfreq("MS")


def clean_series(df: pd.DataFrame) -> pd.Series:
    """Visits with disrupted months replaced by the same-month median of the
    surrounding 2 years on each side, so one-off shocks don't poison lag features.
    The raw values stay in the data; this is only used as model input."""
    y = df["visits"].astype(float).copy()
    bad = df["disrupted"] == 1
    for d in y.index[bad]:
        neigh = [d + pd.DateOffset(years=k) for k in (-2, -1, 1, 2)]
        vals = [y[n] for n in neigh if n in y.index and not bad.get(n, False)]
        if vals:
            y[d] = float(np.median(vals))
    return y


def _features(y: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for target months in `index`, built only from y (lags >= 12) and the calendar."""
    full = y.reindex(y.index.union(index))
    log = np.log(full)
    X = pd.DataFrame(index=full.index)
    X["month"] = X.index.month
    X["t"] = (X.index.year - 1979) * 12 + X.index.month
    X["lag12"] = log.shift(12)
    X["lag24"] = log.shift(24)
    X["lag36"] = log.shift(36)
    X["level12"] = log.shift(12).rolling(12).mean()
    X["level24"] = log.shift(24).rolling(12).mean()
    X["growth"] = X["level12"] - X["level24"]
    cal = calendar_features(X.index)
    X = X.join(cal[CAL_COLS])
    # Easter shift relative to the lag month: did Easter move into / out of this month vs last year?
    X["easter_change"] = X["easter_in_month"] - X["easter_in_month"].shift(12)
    X["weekend_change"] = X["weekend_days"] - X["weekend_days"].shift(12)
    return X.loc[index]


class SeasonalNaive:
    name = "Seasonal naive"
    note = "Same month last year"
    uses_calendar = False

    def fit(self, y):
        self.y = y
        return self

    def predict(self, index):
        return pd.Series([self.y[d - pd.DateOffset(years=1)] for d in index], index=index)


class SeasonalNaiveGrowth:
    name = "Seasonal naive + growth"
    note = "Same month last year x trailing 12-month growth"
    uses_calendar = False

    def fit(self, y):
        self.y = y
        return self

    def predict(self, index):
        out = []
        for d in index:
            base = d - pd.DateOffset(years=1)
            last12 = self.y[base - pd.DateOffset(months=11): base].sum()
            prev12 = self.y[base - pd.DateOffset(months=23): base - pd.DateOffset(months=12)].sum()
            out.append(self.y[base] * np.clip(last12 / prev12, 0.85, 1.15))
        return pd.Series(out, index=index)


class SeasonalRidge:
    name = "Seasonal regression"
    note = "Ridge on log visits: month effects + lags + level"
    uses_calendar = False
    base_cols = ["lag12", "lag24", "level12", "growth"]
    extra_cols: list = []

    def _design(self, X):
        return pd.concat([X[self.base_cols + self.extra_cols],
                          pd.get_dummies(X["month"], prefix="m", dtype=float)], axis=1)

    def fit(self, y):
        self.y = y
        X = _features(y, y.index).dropna()
        Xd = self._design(X)
        self.dcols = Xd.columns
        self.m = Ridge(alpha=1.0).fit(Xd, np.log(y.loc[X.index]))
        return self

    def predict(self, index):
        Xd = self._design(_features(self.y, index)).reindex(columns=self.dcols, fill_value=0.0)
        return pd.Series(np.exp(self.m.predict(Xd)), index=index)


class SeasonalRidgeCal(SeasonalRidge):
    name = "Seasonal regression + calendar"
    note = "Adds Easter shift, weekend-day change, parking-tag policy"
    uses_calendar = True
    extra_cols = ["easter_change", "weekend_change", "parking_tag"]


class GradientBoosting:
    name = "Gradient boosting"
    note = "HistGradientBoosting on log visits with lag and calendar-month features"
    uses_calendar = False
    cols = ["month", "t", "lag12", "lag24", "lag36", "level12", "growth"]

    def fit(self, y):
        self.y = y
        X = _features(y, y.index).dropna()
        self.m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=8, l2_regularization=0.5, random_state=0,
            categorical_features=[0],
        ).fit(X[self.cols], np.log(y.loc[X.index]))
        return self

    def predict(self, index):
        X = _features(self.y, index)
        return pd.Series(np.exp(self.m.predict(X[self.cols])), index=index)


class GradientBoostingCal(GradientBoosting):
    name = "Gradient boosting + calendar"
    note = "Adds Easter, weekend days, parking-tag policy"
    uses_calendar = True
    cols = GradientBoosting.cols + ["easter_in_month", "easter_change", "weekend_days", "parking_tag"]


class Ensemble:
    name = "Ensemble"
    note = "Average of seasonal regression and gradient boosting"
    uses_calendar = False
    parts = (SeasonalRidge, GradientBoosting)

    def fit(self, y):
        self.fitted = [P().fit(y) for P in self.parts]
        return self

    def predict(self, index):
        return sum(p.predict(index) for p in self.fitted) / len(self.fitted)


class EnsembleCal(Ensemble):
    name = "Ensemble + calendar"
    note = "Average of the two calendar-aware models"
    uses_calendar = True
    parts = (SeasonalRidgeCal, GradientBoostingCal)


ALL_MODELS = [SeasonalNaive, SeasonalNaiveGrowth, SeasonalRidge, GradientBoosting, Ensemble,
              SeasonalRidgeCal, GradientBoostingCal, EnsembleCal]


def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def rolling_backtest(df: pd.DataFrame, start: str, end: str, models=ALL_MODELS):
    """Walk forward one year at a time: fit on everything before each 12-month
    block, forecast the block, repeat. Mimics how the model is used in practice."""
    y_clean = clean_series(df)
    y_raw = df["visits"].astype(float)
    rows = []
    for s in pd.date_range(start, end, freq="12MS"):
        block = pd.date_range(s, periods=12, freq="MS")
        block = block[block <= y_raw.index.max()]
        train = y_clean[y_clean.index < s]
        for M in models:
            pred = M().fit(train).predict(block)
            for d in block:
                rows.append({"date": d, "model": M.name, "actual": y_raw[d], "forecast": pred[d]})
    return pd.DataFrame(rows)


def forecast_future(df: pd.DataFrame, model_cls, horizon: int = 12, resid_logs=None):
    y_clean = clean_series(df)
    future = pd.date_range(y_clean.index.max() + pd.DateOffset(months=1), periods=horizon, freq="MS")
    pred = model_cls().fit(y_clean).predict(future)
    out = pd.DataFrame({"date": future, "forecast": pred.values})
    if resid_logs is not None and len(resid_logs):
        lo, hi = np.quantile(resid_logs, [0.10, 0.90])
        out["low_80"] = out.forecast * np.exp(lo)
        out["high_80"] = out.forecast * np.exp(hi)
    return out
