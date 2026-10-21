"""Calendar features (known in advance) and disruption events (known only after the fact).

Calendar features can be used for forecasting because they are known years ahead.
Disruption flags cannot be forecast; they are used to clean model inputs and to
exclude those months from scoring.
"""
from datetime import date

import pandas as pd


def easter(year: int) -> date:
    """Western Easter Sunday (Anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


# Disruptions, with sources. Month-level flags.
EVENTS = [
    {"name": "shutdown_2013", "months": ["2013-10"],
     "label": "Federal government shutdown closed the park Oct 1-15, 2013",
     "source": "https://www.tn.gov/former-governor-haslam/news/2013/10/15/haslam-announces-agreement-to-open-great-smoky-mountains.html"},
    {"name": "wildfire_2016", "months": ["2016-11", "2016-12"],
     "label": "Chimney Tops 2 fire spread into Gatlinburg Nov 28, 2016",
     "source": "https://en.wikipedia.org/wiki/2016_Great_Smoky_Mountains_wildfires"},
    {"name": "covid_closure", "months": ["2020-03", "2020-04", "2020-05"],
     "label": "COVID-19 closure Mar 24 - May 9, 2020",
     "source": "https://wbir.com/article/entertainment/places/news/great-smoky-mountains-to-reopen-in-phases/51-00d55816-8acd-4a2a-917a-86f12d29531d"},
    {"name": "helene", "months": ["2024-09", "2024-10"],
     "label": "Hurricane Helene closures from Sep 26-27, 2024",
     "source": "https://www.nps.gov/grsm/learn/news/newfound-gap-road-us-441-visitor-centers-and-other-park-areas-close-in-preparation-for-tropical-storm-impacts.htm"},
]
EVENT_COLS = [e["name"] for e in EVENTS]

# Known-in-advance policy change
PARKING_TAG_START = "2023-03-01"  # Park It Forward parking tags required from Mar 1, 2023
PARKING_TAG_SOURCE = "https://home.nps.gov/grsm/learn/news/park-announces-2023-fee-program-changes.htm"


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for any month, past or future."""
    out = pd.DataFrame(index=index)
    out["days_in_month"] = index.days_in_month
    out["weekend_days"] = [
        int(pd.date_range(d, d + pd.offsets.MonthEnd(0)).dayofweek.isin([5, 6]).sum()) for d in index
    ]
    out["easter_in_month"] = [int(easter(d.year).month == d.month) for d in index]
    out["parking_tag"] = (index >= pd.Timestamp(PARKING_TAG_START)).astype(int)
    return out


def event_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(0, index=index, columns=EVENT_COLS)
    months = index.strftime("%Y-%m")
    for e in EVENTS:
        out.loc[months.isin(e["months"]), e["name"]] = 1
    return out
