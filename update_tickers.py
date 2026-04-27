#!/usr/bin/env python3
"""
update_tickers.py
Merges reviewed tickers into company_index.parquet and the Top 10 risk list CSVs.

Usage:
1. Review and clean ticker_lookup_results.csv (delete bad suggestions, fix good ones)
2. Save it as approved_tickers.csv (or keep the same name)
3. Run: python update_tickers.py

The script will:
- Update models/company_index.parquet with the new tickers
- Update the two main Top 10 risk list CSVs
- Show you a summary of how many tickers were filled
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# === CONFIGURATION ===
REVIEWED_CSV = "ticker_lookup_results.csv"          # Change this if you renamed it to approved_tickers.csv
BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

# Paths
COMPANY_INDEX = Path("models/company_index.parquet")
TOP_12M = Path("outputs/12m_model/top_2026_companies.csv")
TOP_5Y = Path("outputs/5y_model/top_2026_risk_5y_2026data_only_alive.csv")

def main():
    print("=== Ticker Update Script ===\n")

    # 1. Load reviewed tickers
    print(f"Loading reviewed tickers from: {REVIEWED_CSV}")
    try:
        reviewed = pd.read_csv(REVIEWED_CSV)
    except FileNotFoundError:
        print(f"ERROR: {REVIEWED_CSV} not found. Please review the file and save it.")
        return

    # Support both column names people might use
    if "suggested_ticker" in reviewed.columns:
        reviewed = reviewed.rename(columns={"suggested_ticker": "ticker"})

    if "ticker" not in reviewed.columns or "gvkey" not in reviewed.columns:
        print("ERROR: CSV must have columns 'gvkey' and 'ticker' (or 'suggested_ticker').")
        return

    # Clean up
    reviewed["gvkey"] = reviewed["gvkey"].astype(str).str.strip()
    reviewed["ticker"] = reviewed["ticker"].astype(str).str.strip().str.upper()
    reviewed = reviewed[reviewed["ticker"] != ""]   # only keep rows where user provided a ticker

    print(f"Found {len(reviewed)} companies with filled tickers.\n")

    # 2. Backup original files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Creating backups...")
    shutil.copy(COMPANY_INDEX, BACKUP_DIR / f"company_index_{timestamp}.parquet")
    if TOP_12M.exists():
        shutil.copy(TOP_12M, BACKUP_DIR / f"top_2026_companies_{timestamp}.csv")
    if TOP_5Y.exists():
        shutil.copy(TOP_5Y, BACKUP_DIR / f"top_5y_list_{timestamp}.csv")
    print("Backups saved in ./backups/\n")

    # 3. Update company_index.parquet
    print("Updating models/company_index.parquet ...")
    company_index = pd.read_parquet(COMPANY_INDEX)
    company_index["gvkey"] = company_index["gvkey"].astype(str).str.strip()

    # Merge new tickers
    ticker_map = dict(zip(reviewed["gvkey"], reviewed["ticker"]))
    company_index["ticker"] = company_index.apply(
        lambda row: ticker_map.get(row["gvkey"], row["ticker"]), axis=1
    )

    company_index.to_parquet(COMPANY_INDEX, index=False)
    print(f"  → Updated {len(reviewed)} tickers in company_index.parquet\n")

    # 4. Update the two Top 10 risk list CSVs
    updated_12m = 0
    updated_5y = 0

    # 12-month list
    if TOP_12M.exists():
        print("Updating 12-month Top 10 list...")
        df12 = pd.read_csv(TOP_12M)
        df12["gvkey"] = df12["gvkey"].astype(str).str.strip()
        df12["ticker"] = df12.apply(
            lambda row: ticker_map.get(str(row["gvkey"]), row.get("ticker", "")), axis=1
        )
        df12.to_csv(TOP_12M, index=False)
        updated_12m = len(df12[df12["gvkey"].isin(ticker_map.keys())])
        print(f"  → Updated {updated_12m} tickers in top_2026_companies.csv")

    # 5-year list
    if TOP_5Y.exists():
        print("Updating 5-year Top 10 list...")
        df5y = pd.read_csv(TOP_5Y)
        df5y["gvkey"] = df5y["gvkey"].astype(str).str.strip()
        df5y["ticker"] = df5y.apply(
            lambda row: ticker_map.get(str(row["gvkey"]), row.get("ticker", "")), axis=1
        )
        df5y.to_csv(TOP_5Y, index=False)
        updated_5y = len(df5y[df5y["gvkey"].isin(ticker_map.keys())])
        print(f"  → Updated {updated_5y} tickers in top_2026_risk_5y_2026data_only_alive.csv")

    # 5. Final summary
    still_missing = len(company_index[company_index["ticker"].isna() | (company_index["ticker"] == "")])
    total = len(company_index)

    print("\n" + "="*55)
    print("UPDATE COMPLETE")
    print("="*55)
    print(f"Total companies in index     : {total}")
    print(f"Tickers you just added       : {len(reviewed)}")
    print(f"Still missing tickers        : {still_missing}")
    print(f"Companies with tickers now   : {total - still_missing}")
    print("\nAll Top 10 risk lists have also been updated.")
    print("You can now run the app — many more companies will show stock charts!")

if __name__ == "__main__":
    main()
