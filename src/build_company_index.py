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

    # === Attach latest known credit rating (CapIQ / S&P issuer rating) for UI display ===
    # Ratings are NOT yet used as model features (per AGENTS.md spec); this is for transparency in the UI.
    # To populate with real data: add WRDS query on ciqsamp_ratings / ciq_rating / comp.adsprate (splticrm etc.)
    # and merge latest rating per gvkey. Wrapped in try so build never breaks.
    credit_count = 0
    latest["credit_rating"] = ""
    try:
        import wrds
        print("Connecting to WRDS for credit rating lookup (ciqsamp_ratings / adsprate)...")
        conn = wrds.Connection()
        # Try CapIQ ratings table first (common under ciqsamp or ciq_rating schemas)
        try:
            ratings = conn.raw_sql("""
                SELECT DISTINCT ON (gvkey) gvkey, ratingvalue as credit_rating
                FROM ciqsamp_ratings
                WHERE ratingvalue IS NOT NULL
                ORDER BY gvkey, ratingdate DESC NULLS LAST
            """)
        except Exception:
            # Fallback: S&P ratings via Compustat adsprate (splticrm = S&P Long Term Issuer Credit Rating)
            ratings = conn.raw_sql("""
                SELECT DISTINCT ON (gvkey) gvkey, splticrm as credit_rating
                FROM comp.adsprate
                WHERE splticrm IS NOT NULL
                ORDER BY gvkey, datadate DESC NULLS LAST
            """)
        conn.close()

        if "gvkey" in ratings.columns:
            ratings["gvkey"] = ratings["gvkey"].astype(str).str.strip().str.zfill(6)
            latest["gvkey"] = latest["gvkey"].astype(str).str.strip().str.zfill(6)

            # First try gvkey
            latest = latest.merge(ratings[["gvkey", "credit_rating"]], on="gvkey", how="left", suffixes=("", "_new"))

            # Ticker fallback (many rating records also contain ticker)
            if "ticker" in ratings.columns and "ticker" in latest.columns:
                still_missing = latest["credit_rating"].isna() | (latest["credit_rating"].astype(str).str.strip() == "")
                if still_missing.any() and still_missing.sum() > 0:
                    tr = ratings.dropna(subset=["ticker"])[["ticker", "credit_rating"]].copy()
                    tr["ticker"] = tr["ticker"].astype(str).str.upper().str.strip()
                    tr = tr.drop_duplicates("ticker")
                    ticker_map = dict(zip(tr["ticker"], tr["credit_rating"]))
                    latest.loc[still_missing, "credit_rating"] = latest.loc[still_missing, "ticker"].map(ticker_map)

            if "credit_rating_new" in latest.columns:
                latest["credit_rating"] = latest["credit_rating_new"].fillna(latest.get("credit_rating", "")).astype(str).str.strip().str.upper()
                latest = latest.drop(columns=["credit_rating_new"], errors="ignore")

            credit_count = int((latest["credit_rating"].astype(str).str.strip() != "").sum())
            print(f"Attached credit ratings for {credit_count:,} companies (using gvkey + ticker fallback).")
    except Exception as e:
        print(f"Could not fetch credit ratings from WRDS: {e}")
        print("credit_rating will be blank (extend WRDS query in build_company_index.py to populate CapIQ ratings).")
        latest["credit_rating"] = ""

    # Build fast search string (name + gvkey + ticker)
    latest["search_text"] = (
        latest["conm"].astype(str).str.upper() + " " +
        latest["gvkey"] + " " +
        latest["ticker"].astype(str).str.upper()
    )

    # Final clean columns (now includes credit_rating for UI)
    latest = latest[["gvkey", "conm", "ticker", "credit_rating", "latest_datadate", "search_text"]]

    # Save
    out_path = MODELS_DIR / "company_index.parquet"
    latest.to_parquet(out_path, index=False)

    print(f"\n✅ Company index saved → {out_path}")
    print(f"   Total searchable companies: {len(latest):,}")
    print(f"   Companies with tickers:     {ticker_count:,}")
    print("\nYou can now restart the UI server — all suggestions will work.")


if __name__ == "__main__":
    main()
