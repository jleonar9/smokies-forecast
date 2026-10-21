"""Static figures for the README (matplotlib only)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "reports"
hist = pd.read_csv(P / "grsm_monthly.csv", parse_dates=["date"])
fc = pd.read_csv(P / "forecast.csv", parse_dates=["date"])
m = pd.read_csv(P / "metrics.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

h = hist[hist.date >= "2019-01-01"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(fc.date, fc.low_80 / 1e6, fc.high_80 / 1e6, color="#2E7D32", alpha=0.15, label="80% range")
ax.plot(h.date, h.visits / 1e6, color="#37474F", lw=1.6, label="Actual")
ax.plot(fc.date, fc.forecast / 1e6, color="#2E7D32", lw=1.8, ls="--", label="Forecast")
ax.annotate("COVID closure\nApr 2020", xy=(pd.Timestamp("2020-04-01"), 0.1), xytext=(pd.Timestamp("2020-09-01"), 0.25),
            arrowprops=dict(arrowstyle="->", color="#888"), fontsize=8, color="#555")
ax.set_ylabel("Monthly visits (millions)")
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.suptitle("Great Smoky Mountains NP: monthly visits and 12-month forecast", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "forecast.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 3.2))
mm = m.sort_values("mape_all_years", ascending=False)
colors = ["#2E7D32" if i == m.index[0] else ("#90A4AE" if n != "Seasonal naive" else "#455A64") for i, n in zip(mm.index, mm.model)]
ax.barh(mm.model, mm.mape_all_years, color=colors)
for y, v in enumerate(mm.mape_all_years):
    ax.text(v + 0.1, y, f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlabel("Backtest MAPE, Oct 2016 - Sep 2026 (lower is better)")
ax.set_xlim(0, mm.mape_all_years.max() * 1.2)
ax.set_title("Model comparison", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=150)
print("figures written")

# ---- Weather effects ----
e = pd.read_csv(P / "weather_effects.csv").iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
cols = ["#2E7D32" if (c and v > 0) else ("#C62828" if c else "#B0BEC5") for c, v in zip(e.clear_effect, e.effect_pct)]
ax.barh(e.factor, e.effect_pct, color=cols,
        xerr=[e.effect_pct - e.low_95, e.high_95 - e.effect_pct], error_kw=dict(ecolor="#555", lw=1, capsize=3))
ax.axvline(0, color="#333", lw=0.8)
for y, v in enumerate(e.effect_pct):
    ax.text(e.high_95.iloc[y] + 0.3 if v >= 0 else e.low_95.iloc[y] - 0.3, y, f"{v:+.1f}%",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(e.low_95.min() - 2.5, e.high_95.max() + 2.5)
ax.set_xlabel("Change in monthly visits (%), 95% interval; grey = not clear")
ax.set_title("How unusual weather moves Smokies visits", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "weather_effects.png", dpi=150)
print("weather figure written")
