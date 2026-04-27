#!/usr/bin/env python3
"""
Lightweight script to (re)build the company search index for the UI.

This script:
- Loads the test panel (latest fiscal observation per company)
- Normalizes gvkey as string
- Optionally pulls real stock tickers from WRDS crsp.stocknames
- Saves models/company_index.parquet

Run this whenever you want to refresh the searchable company list or tickers
WITHOUT retraining the models:

    python src/build_company_index.py
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("Building company index for Pulse Credit Risk UI...")

    # Load the same test panel used for 2026 forward predictions
    full_panel = pd.read_parquet(DATA_PROC / "monthly_test.parquet")

    # Latest observation per company (this is what the UI will score)
    latest = full_panel.sort_values("datadate").groupby("gvkey").tail(1)[
        ["gvkey", "conm", "datadate", "permno"]
    ].copy()

    latest = latest.rename(columns={"datadate": "latest_datadate"})

    # Always store gvkey as clean string (critical for matching)
    latest["gvkey"] = latest["gvkey"].astype(str).str.strip()

    # Ensure permno is string for safe merging (WRDS returns it as object sometimes)
    if "permno" in latest.columns:
        latest["permno"] = latest["permno"].astype(str).str.strip()

    # Try to attach ticker from WRDS (best & most accurate source)
    ticker_count = 0
    try:
        import wrds
        print("Connecting to WRDS for ticker lookup (crsp.stocknames)...")
        conn = wrds.Connection()
        tickers = conn.raw_sql("""
            SELECT DISTINCT ON (permno) permno, ticker
            FROM crsp.stocknames
            WHERE ticker IS NOT NULL
            ORDER BY permno, nameenddt DESC NULLS LAST
        """)
        conn.close()

        tickers["permno"] = tickers["permno"].astype(str).str.strip()
        latest = latest.merge(tickers[["permno", "ticker"]], on="permno", how="left")
        latest["ticker"] = latest["ticker"].fillna("").str.upper().str.strip()
        ticker_count = int((latest["ticker"] != "").sum())
        print(f"Attached real historical tickers for {ticker_count:,} companies.")
    except ImportError:
        print("Could not import 'wrds' module.")
        print("→ Run: pip install wrds")
        print("→ Then make sure your WRDS credentials are set up (same as the notebooks).")
        print("Ticker field will be left empty for now.")
        latest["ticker"] = ""
    except Exception as e:
        print(f"Could not fetch tickers from WRDS: {e}")
        print("Ticker field will be empty.")
        latest["ticker"] = ""

    # Build fast search string (name + gvkey + ticker)
    latest["search_text"] = (
        latest["conm"].astype(str).str.upper() + " " +
        latest["gvkey"] + " " +
        latest["ticker"].astype(str).str.upper()
    )

    # Final clean columns
    latest = latest[["gvkey", "conm", "ticker", "latest_datadate", "search_text"]]

    # Save
    out_path = MODELS_DIR / "company_index.parquet"
    latest.to_parquet(out_path, index=False)

    print(f"\n✅ Company index saved → {out_path}")
    print(f"   Total searchable companies: {len(latest):,}")
    print(f"   Companies with tickers:     {ticker_count:,}")
    print("\nYou can now restart the UI server — all suggestions will work.")


if __name__ == "__main__":
    main()
