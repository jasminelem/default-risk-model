#!/usr/bin/env python3
"""
One-time script to pull historical CapIQ (or S&P) ratings and save them
so they become real features in the default prediction models.

Run:
    python src/pull_capiqu_ratings.py

It uses the non-interactive WRDS helper (never asks for password once ~/.pgpass is set).
"""

# Make "from src.xxx" imports work when running the file directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import pandas as pd
from src.wrds_utils import get_wrds_connection

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

def main():
    print("Pulling historical CapIQ / S&P ratings (non-interactively)...")
    conn = get_wrds_connection()

    # 1. Try CapIQ sources first (most users don't have access)
    df = None
    capiq_sources = [
        ("ciqsamp_ratings", """
            SELECT gvkey, ticker, ratingdate, ratingvalue as rating
            FROM ciqsamp_ratings
            WHERE ratingvalue IS NOT NULL
        """),
        ("ciq.rating", """
            SELECT gvkey, ticker, actdate as ratingdate, ratingvalue as rating
            FROM ciq.rating
            WHERE ratingvalue IS NOT NULL
        """),
    ]
    for name, q in capiq_sources:
        try:
            df = conn.raw_sql(q)
            if len(df) > 0:
                print(f"Got {len(df):,} rows from {name}")
                break
        except Exception:
            continue

    # 2. Try the fuller Compustat North America table first (recommended)
    if df is None or len(df) == 0:
        try:
            print("Trying comp_na_daily_all.adsprate (full S&P ratings table)...")
            df = conn.raw_sql("""
                SELECT gvkey,
                       datadate as ratingdate,
                       splticrm as rating
                FROM comp_na_daily_all.adsprate
                WHERE splticrm IS NOT NULL
            """)
            print(f"Got {len(df):,} rows from comp_na_daily_all.adsprate")
        except Exception as e:
            print(f"comp_na_daily_all.adsprate not available or failed: {e}")
            df = None

    # 3. Final fallback: the simpler comp.adsprate view
    if df is None or len(df) == 0:
        print("Falling back to comp.adsprate...")
        df = conn.raw_sql("""
            SELECT gvkey,
                   datadate as ratingdate,
                   splticrm as rating
            FROM comp.adsprate
            WHERE splticrm IS NOT NULL
        """)
        print(f"Got {len(df):,} rows from comp.adsprate")

    conn.close()

    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)

    # Safe ticker handling (adsprate doesn't have ticker, CapIQ might)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    else:
        df["ticker"] = ""

    df["ratingdate"] = pd.to_datetime(df["ratingdate"], errors="coerce")
    df = df.dropna(subset=["rating", "ratingdate"])

    out = DATA_RAW / "ciq_ratings.parquet"
    df.to_parquet(out, index=False)

    print(f"\n✅ Saved {len(df):,} rating records → {out}")
    print(f"   Covers {df['gvkey'].nunique():,} unique companies")
    print(f"   Date range: {df['ratingdate'].min().date()} → {df['ratingdate'].max().date()}")
    print("\nYou can now run:")
    print("   python src/build_clean_training_data.py")
    print("   python src/save_best_models.py")
    print("The two default prediction models will now use historical CapIQ/S&P ratings")
    print("as real features (rating_numeric, speculative_grade, downgrade_1y, rating_age_years).")


if __name__ == "__main__":
    main()