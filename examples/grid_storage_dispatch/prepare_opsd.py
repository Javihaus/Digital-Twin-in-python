"""Convert a raw Open Power System Data time-series CSV into the schema this
example consumes (data/eu_market.csv: timestamp, price_eur_mwh, load_mw).

Why a converter: the OPSD file is large and cannot be auto-downloaded by the
notebook tooling, so you download it once and this script slices it to a small,
single-country window.

Steps
-----
1. Download the OPSD 60-minute time series (CC-BY 4.0), e.g.:
     https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
   Save it as:  data/time_series_60min_singleindex.csv

2. Run:  python prepare_opsd.py            # defaults to Spain (ES)
         python prepare_opsd.py DE_LU 90   # country/zone, number of days

Output: data/eu_market.csv  (timestamp, price_eur_mwh, load_mw)
Then:   python run_dispatch.py            # auto-detects and uses the real data
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "time_series_60min_singleindex.csv"
OUT = HERE / "data" / "eu_market.csv"


def main(country: str = "ES", n_days: int = 90) -> None:
    if not RAW.exists():
        raise SystemExit(
            f"Raw OPSD file not found at {RAW}.\n"
            "Download time_series_60min_singleindex.csv from "
            "https://data.open-power-system-data.org/time_series/ and place it there."
        )
    price_col = f"{country}_price_day_ahead"
    load_col = f"{country}_load_actual_entsoe_transparency"
    usecols = ["utc_timestamp", price_col, load_col]

    df = pd.read_csv(RAW, usecols=lambda c: c in usecols)
    missing = [c for c in (price_col, load_col) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Columns {missing} not in the OPSD file for country '{country}'. "
            "Try another zone (e.g. DE_LU, FR, NL) — check the file header."
        )
    df = df.rename(
        columns={
            price_col: "price_eur_mwh",
            load_col: "load_mw",
            "utc_timestamp": "timestamp",
        }
    )
    df = df.dropna(subset=["price_eur_mwh", "load_mw"]).reset_index(drop=True)
    # take the last complete n_days * 24 hours available
    n = n_days * 24
    df = df.iloc[-((len(df) // 24) * 24) :]  # whole days
    if len(df) > n:
        df = df.iloc[-n:]
    df[["timestamp", "price_eur_mwh", "load_mw"]].to_csv(OUT, index=False)
    print(
        f"Wrote {OUT} | country={country} | days={len(df)//24} | rows={len(df)}\n"
        f"  price €/MWh: min {df.price_eur_mwh.min():.1f}  "
        f"max {df.price_eur_mwh.max():.1f}  mean {df.price_eur_mwh.mean():.1f}\n"
        f"  load  MW:    min {df.load_mw.min():.0f}  "
        f"max {df.load_mw.max():.0f}  (national; rescaled to site in run_dispatch)"
    )


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "ES"
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    main(c, d)
