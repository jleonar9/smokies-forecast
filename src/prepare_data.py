"""Build the modeling table: NPS monthly visits + weather + calendar + disruption events.

Inputs  (data/raw):  grsm_monthly_raw.txt  NPS report table
                     weather_monthly.csv   Open-Meteo archive, Gatlinburg + Cherokee average
Output  (data/processed): grsm_monthly.csv
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from events import EVENT_COLS, calendar_features, event_flags  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- Visits ----
raw = pd.read_csv(ROOT / "data/raw/grsm_monthly_raw.txt", sep="|", dtype=str)
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
for c in months + ["Total"]:
    raw[c] = pd.to_numeric(raw[c].str.replace(",", ""), errors="coerce")
bad = raw[(raw[months].sum(axis=1) - raw["Total"]).abs() > 0]
if len(bad):
    sys.exit(f"Row totals don't match for years: {bad.Year.tolist()}")
long = raw.melt(id_vars="Year", value_vars=months, var_name="mon", value_name="visits").dropna()
long["date"] = pd.to_datetime(long.Year + "-" + long.mon, format="%Y-%b")
df = long[["date", "visits"]].sort_values("date").set_index("date")
df["visits"] = df.visits.astype(int)
df["preliminary"] = (df.index.year == 2026).astype(int)

# ---- Weather ----
wx = pd.read_csv(ROOT / "data/raw/weather_monthly.csv")
wx["date"] = pd.to_datetime(wx.month + "-01")
wx = wx[wx.days == wx.date.dt.days_in_month].set_index("date").drop(columns=["month", "days"])
df = df.join(wx, how="left")
missing_wx = df.tmean_f.isna().sum()
if missing_wx:
    sys.exit(f"{missing_wx} months have no complete weather record")

# ---- Calendar + events ----
df = df.join(calendar_features(df.index)).join(event_flags(df.index))
df["disrupted"] = (df[EVENT_COLS].sum(axis=1) > 0).astype(int)

(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
df.reset_index().to_csv(ROOT / "data/processed/grsm_monthly.csv", index=False)
print(f"OK: {len(df)} months, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}; "
      f"all {len(raw)} visit totals validated; weather complete; {df.disrupted.sum()} disrupted months flagged")
