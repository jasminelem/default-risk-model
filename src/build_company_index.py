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

# Make "from src.xxx" work when running the script directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 2-digit SIC industry mapping (granular enough for meaningful peer comparison)
SIC2_NAMES = {
    1: "Crops", 2: "Livestock", 7: "Agricultural Services", 8: "Forestry", 9: "Fishing & Hunting",
    10: "Metal Mining", 12: "Coal Mining", 13: "Oil & Gas Extraction", 14: "Nonmetallic Minerals",
    15: "Building Construction", 16: "Heavy Construction", 17: "Special Trade Contractors",
    20: "Food & Kindred Products", 21: "Tobacco", 22: "Textile Mill Products", 23: "Apparel",
    24: "Lumber & Wood", 25: "Furniture & Fixtures", 26: "Paper & Allied Products",
    27: "Printing & Publishing", 28: "Chemicals & Pharmaceuticals", 29: "Petroleum Refining",
    30: "Rubber & Plastics", 31: "Leather", 32: "Stone, Clay & Glass", 33: "Primary Metals",
    34: "Fabricated Metals", 35: "Industrial Machinery & Computers", 36: "Electronic Equipment",
    37: "Transportation Equipment", 38: "Instruments & Medical Devices", 39: "Misc Manufacturing",
    40: "Railroads", 41: "Transit", 42: "Trucking & Warehousing", 44: "Water Transportation",
    45: "Air Transportation", 46: "Pipelines", 47: "Transportation Services", 48: "Telecommunications",
    49: "Electric, Gas & Sanitary Services",
    50: "Wholesale - Durable Goods", 51: "Wholesale - Nondurable Goods",
    52: "Building Materials Retail", 53: "General Merchandise", 54: "Food Stores",
    55: "Auto Dealers & Gas Stations", 56: "Apparel Retail", 57: "Furniture & Home Stores",
    58: "Eating & Drinking Places", 59: "Misc Retail",
    60: "Banks & Depository Institutions", 61: "Non-Depository Credit", 62: "Security Brokers & Dealers",
    63: "Insurance Carriers", 64: "Insurance Agents", 65: "Real Estate", 67: "Holding & Investment Offices",
    70: "Hotels & Lodging", 72: "Personal Services", 73: "Business Services & Software",
    75: "Auto Repair & Services", 78: "Motion Pictures", 79: "Amusement & Recreation",
    80: "Health Services", 81: "Legal Services", 82: "Educational Services",
    83: "Social Services", 87: "Engineering & Management Services", 89: "Misc Services",
    91: "Executive & Legislative", 92: "Justice & Public Order", 95: "Environmental Quality",
    96: "Administration of Economic Programs", 99: "Nonclassifiable",
}


def sic_to_industry(sic_code):
    """Map a 4-digit SIC code to its 2-digit industry name."""
    if pd.isna(sic_code):
        return ""
    sic2 = int(sic_code) // 100
    return SIC2_NAMES.get(sic2, "")


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

    # === Multi-source ticker lookup ===
    # Priority: 1) Compustat security (GVKEY→ticker, most direct)
    #           2) CRSP stocknames (PERMNO→ticker, date-filtered to avoid recycled PERMNOs)
    #           3) CIQ ratings (local fallback)
    ticker_count = 0
    latest["ticker"] = ""
    try:
        from src.wrds_utils import get_wrds_connection
        conn = get_wrds_connection(verbose=False)

        # Source 1: Compustat names table (direct GVKEY → ticker, most accurate)
        print("  Ticker source 1: Compustat comp.names (GVKEY-based)...")
        try:
            comp_tickers = conn.raw_sql("""
                SELECT DISTINCT ON (gvkey) gvkey, tic AS ticker
                FROM comp.names
                WHERE tic IS NOT NULL
                ORDER BY gvkey, year2 DESC NULLS LAST
            """)
            comp_tickers["gvkey"] = comp_tickers["gvkey"].astype(str).str.strip()
            comp_tickers["ticker"] = comp_tickers["ticker"].str.upper().str.strip()
            latest = latest.merge(comp_tickers[["gvkey", "ticker"]], on="gvkey", how="left", suffixes=("", "_comp"))
            if "ticker_comp" in latest.columns:
                latest["ticker"] = latest["ticker_comp"].fillna(latest["ticker"])
                latest = latest.drop(columns=["ticker_comp"])
            s1 = int((latest["ticker"].str.strip() != "").sum())
            print(f"    {s1:,} tickers from Compustat")
        except Exception as e:
            print(f"    Compustat names query failed: {e}")

        # Source 2: CRSP stocknames (date-filtered to company's observation period)
        still_missing = latest["ticker"].str.strip() == ""
        if still_missing.any() and "permno" in latest.columns:
            print(f"  Ticker source 2: CRSP stocknames (date-filtered, {still_missing.sum():,} remaining)...")
            try:
                crsp_tickers = conn.raw_sql("""
                    SELECT permno, ticker, namedt, nameenddt
                    FROM crsp.stocknames
                    WHERE ticker IS NOT NULL
                """)
                crsp_tickers["permno"] = crsp_tickers["permno"].astype(str).str.strip()
                crsp_tickers["ticker"] = crsp_tickers["ticker"].str.upper().str.strip()
                crsp_tickers["nameenddt"] = pd.to_datetime(crsp_tickers["nameenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
                crsp_tickers["namedt"] = pd.to_datetime(crsp_tickers["namedt"], errors="coerce")

                # Join and filter: the ticker must have been active around the company's latest date
                missing = latest[still_missing][["gvkey", "permno", "latest_datadate"]].copy()
                missing["latest_datadate"] = pd.to_datetime(missing["latest_datadate"])
                merged = missing.merge(crsp_tickers, on="permno", how="inner")
                merged = merged[
                    (merged["latest_datadate"] >= merged["namedt"]) &
                    (merged["latest_datadate"] <= merged["nameenddt"])
                ]
                # Take latest valid ticker per gvkey
                best = merged.sort_values("nameenddt", ascending=False).drop_duplicates("gvkey", keep="first")

                ticker_map = dict(zip(best["gvkey"], best["ticker"]))
                latest.loc[still_missing, "ticker"] = latest.loc[still_missing, "gvkey"].map(ticker_map).fillna("")
                s2 = int((latest["ticker"].str.strip() != "").sum()) - (len(latest) - still_missing.sum())
                print(f"    {s2:,} additional tickers from CRSP")
            except Exception as e:
                print(f"    CRSP stocknames query failed: {e}")

        conn.close()

        # Source 3: local CIQ ratings file (no WRDS needed)
        still_missing = latest["ticker"].str.strip() == ""
        if still_missing.any():
            try:
                ciq = pd.read_parquet(DATA_RAW / "ciq_ratings.parquet")
                ciq_tic = ciq.dropna(subset=["ticker"]).drop_duplicates("gvkey", keep="last")
                ciq_tic["gvkey"] = ciq_tic["gvkey"].astype(str).str.strip()
                ciq_tic["ticker"] = ciq_tic["ticker"].str.upper().str.strip()
                ciq_map = dict(zip(ciq_tic["gvkey"], ciq_tic["ticker"]))
                latest.loc[still_missing, "ticker"] = latest.loc[still_missing, "gvkey"].map(ciq_map).fillna("")
                s3 = int((latest["ticker"].str.strip() != "").sum()) - (len(latest) - still_missing.sum())
                print(f"  Ticker source 3: CIQ ratings (local): {s3:,} additional")
            except Exception:
                pass

        latest["ticker"] = latest["ticker"].fillna("").str.upper().str.strip()
        ticker_count = int((latest["ticker"] != "").sum())
        print(f"  Total tickers attached: {ticker_count:,} / {len(latest):,}")

    except ImportError:
        print("Could not import 'wrds' module. Run: pip install wrds")
        latest["ticker"] = ""
    except Exception as e:
        print(f"Ticker lookup failed: {e}")
        latest["ticker"] = latest["ticker"].fillna("")

    # === Attach latest known credit rating (CapIQ / S&P issuer rating) for UI display ===
    # Ratings are NOT yet used as model features (per AGENTS.md spec); this is for transparency in the UI.
    # To populate with real data: add WRDS query on ciqsamp_ratings / ciq_rating / comp.adsprate (splticrm etc.)
    # and merge latest rating per gvkey. Wrapped in try so build never breaks.
    credit_count = 0
    latest["credit_rating"] = ""
    try:
        from src.wrds_utils import get_wrds_connection
        print("Connecting to WRDS for credit rating lookup (ciqsamp_ratings / adsprate)...")
        conn = get_wrds_connection(verbose=False)
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

    # === Attach industry sector from Compustat SIC codes ===
    industry_count = 0
    latest["industry"] = ""
    try:
        funda_na = pd.read_parquet(DATA_RAW / "comp_funda_na.parquet", columns=["gvkey", "datadate", "sich"])
        funda_na["gvkey"] = funda_na["gvkey"].astype(str).str.strip().str.zfill(6)
        funda_na["sich"] = pd.to_numeric(funda_na["sich"], errors="coerce")
        # Take the most recent SIC code per company
        sic_latest = (funda_na.dropna(subset=["sich"])
                      .sort_values("datadate")
                      .groupby("gvkey", as_index=False)
                      .tail(1)[["gvkey", "sich"]])
        sic_latest["industry"] = sic_latest["sich"].apply(sic_to_industry)
        sic_latest["sic4"] = sic_latest["sich"].astype(int)

        latest["gvkey"] = latest["gvkey"].astype(str).str.strip().str.zfill(6)
        latest = latest.merge(sic_latest[["gvkey", "industry", "sic4"]], on="gvkey", how="left", suffixes=("", "_new"))
        if "industry_new" in latest.columns:
            latest["industry"] = latest["industry_new"].fillna(latest["industry"])
            latest = latest.drop(columns=["industry_new"])
        latest["industry"] = latest["industry"].fillna("")
        industry_count = int((latest["industry"].str.strip() != "").sum())
        print(f"Attached industry sector for {industry_count:,} companies (from Compustat SIC).")
    except Exception as e:
        print(f"Could not attach industry data: {e}")

    # Build fast search string (name + gvkey + ticker + industry)
    latest["search_text"] = (
        latest["conm"].astype(str).str.upper() + " " +
        latest["gvkey"] + " " +
        latest["ticker"].astype(str).str.upper() + " " +
        latest["industry"].astype(str).str.upper()
    )

    # Final clean columns
    if "sic4" not in latest.columns:
        latest["sic4"] = 0
    latest["sic4"] = latest["sic4"].fillna(0).astype(int)
    latest = latest[["gvkey", "conm", "ticker", "credit_rating", "industry", "sic4", "latest_datadate", "search_text"]]

    # Save
    out_path = MODELS_DIR / "company_index.parquet"
    latest.to_parquet(out_path, index=False)

    print(f"\n✅ Company index saved → {out_path}")
    print(f"   Total searchable companies: {len(latest):,}")
    print(f"   Companies with tickers:     {ticker_count:,}")
    print("\nYou can now restart the UI server — all suggestions will work.")


if __name__ == "__main__":
    main()
