"""
src/ratings.py

Historical CapIQ / S&P credit ratings as first-class model features.

Used by:
- build_clean_training_data.py  → attaches rating_numeric, speculative_grade, downgrade_1y, rating_age_years
- build_company_index.py        → latest rating for UI (with ticker fallback)
- pull_ratings.py               → one-time data pull (uses non-interactive WRDS)

All WRDS connections go through src/wrds_utils.py so you never have to type a password.
"""

# Make "from src.xxx" imports work when running directly or as a module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from src.wrds_utils import get_wrds_connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# S&P-style numeric scale (lower = better). Used as a real feature.
# ---------------------------------------------------------------------------
RATING_MAP = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22, "SD": 23
}


def _to_numeric(rating: str) -> float:
    if pd.isna(rating):
        return np.nan
    return RATING_MAP.get(str(rating).strip().upper(), np.nan)


def _load_local() -> pd.DataFrame:
    """Load any ratings parquet the user has created."""
    for name in ["ciq_ratings.parquet", "adsprate.parquet", "ratings_history.parquet"]:
        p = DATA_RAW / name
        if p.exists():
            df = pd.read_parquet(p)
            # Normalise common column names
            df = df.rename(columns={
                "ratingvalue": "rating", "splticrm": "rating",
                "rvalue": "rating", "datadate": "ratingdate"
            })
            df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
            df["ratingdate"] = pd.to_datetime(df["ratingdate"], errors="coerce")
            df = df.dropna(subset=["rating", "ratingdate"])
            return df[["gvkey", "ticker", "ratingdate", "rating"]]
    return pd.DataFrame(columns=["gvkey", "ticker", "ratingdate", "rating"])


def get_ratings_df(live: bool = False) -> pd.DataFrame:
    """Returns a clean ratings frame (gvkey, ticker, ratingdate, rating)."""
    if live:
        try:
            conn = get_wrds_connection(verbose=True)
            # Try CapIQ first
            try:
                df = conn.raw_sql("""
                    SELECT gvkey, ticker, ratingdate, ratingvalue as rating
                    FROM ciqsamp_ratings
                    WHERE ratingvalue IS NOT NULL
                """)
            except Exception:
                # Fallback to Compustat S&P
                df = conn.raw_sql("""
                    SELECT gvkey, tic as ticker, datadate as ratingdate, splticrm as rating
                    FROM comp.adsprate
                    WHERE splticrm IS NOT NULL
                """)
            conn.close()
            return df
        except Exception as e:
            print(f"[ratings] Live WRDS pull failed: {e}")
            return pd.DataFrame(columns=["gvkey", "ticker", "ratingdate", "rating"])

    return _load_local()


# ---------------------------------------------------------------------------
# Public API used by the training pipeline and UI
# ---------------------------------------------------------------------------

def attach_historical_ratings(panel: pd.DataFrame, live: bool = False) -> pd.DataFrame:
    """
    Point-in-time attach CapIQ/S&P ratings to the firm-year panel.

    For each (gvkey, datadate) we take the most recent rating with ratingdate <= datadate.
    This respects the no-leak rule required by AGENTS.md.

    Also tries ticker fallback for better coverage on small companies.
    Produces the features the model will actually use:
        rating_numeric, speculative_grade, downgrade_1y, rating_age_years
    """
    df = panel.copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

    # Remove any bad dates first (this is the #1 cause of "left keys must be sorted")
    before_len = len(df)
    df = df[df["datadate"].notna()].copy()
    if len(df) < before_len:
        print(f"[ratings] Removed {before_len - len(df)} rows with invalid datadate before joining ratings.")

    ratings = get_ratings_df(live=live)
    if ratings.empty:
        print("[ratings] No CapIQ/S&P ratings data found. Skipping (model will run without rating features).")
        for c in ["rating_numeric", "speculative_grade", "downgrade_1y", "rating_age_years"]:
            df[c] = 0 if c == "speculative_grade" else np.nan
        return df

    # Clean right side too (bad ratingdate would also cause problems)
    ratings = ratings[ratings["ratingdate"].notna()].copy()

    # === FAST & RELIABLE METHOD: proper join + filter + last (what you suggested) ===
    # This is usually faster than groupby.apply(merge_asof) and never has "left keys must be sorted" problems.
    ratings = ratings[["gvkey", "ratingdate", "rating"]].dropna().copy()
    ratings["ratingdate"] = pd.to_datetime(ratings["ratingdate"])

    # 1. Left join on gvkey (vectorized)
    combined = pd.merge(
        df[["gvkey", "datadate"]],
        ratings,
        on="gvkey",
        how="left"
    )

    # 2. Keep only ratings that were known on or before the fiscal year-end (no leakage)
    valid = combined[combined["ratingdate"] <= combined["datadate"]]

    # 3. For each (gvkey, datadate) keep only the most recent rating
    if not valid.empty:
        best_rating = (valid.sort_values(["gvkey", "datadate", "ratingdate"])
                            .groupby(["gvkey", "datadate"], as_index=False)
                            .last()[["gvkey", "datadate", "rating", "ratingdate"]])
    else:
        best_rating = pd.DataFrame(columns=["gvkey", "datadate", "rating", "ratingdate"])

    # 4. Merge the chosen rating back to the original panel
    merged = pd.merge(df, best_rating, on=["gvkey", "datadate"], how="left")

    # Ticker fallback (optional, for extra coverage on microcaps)
    # We keep it light because the main gvkey join is now fast and reliable.
    if "ticker" in merged.columns:
        miss = merged["rating"].isna()
        if miss.any():
            # Use a copy of the original ratings for ticker matching
            rating_source = get_ratings_df(live=live)
            tr = rating_source.dropna(subset=["ticker"])[["ticker", "ratingdate", "rating"]].sort_values(["ticker", "ratingdate"]).reset_index(drop=True)
            left_t = merged.loc[miss, ["ticker", "datadate"]].sort_values(["ticker", "datadate"]).reset_index(drop=True)
            try:
                t_join = pd.merge_asof(
                    left_t, tr, left_on="datadate", right_on="ratingdate",
                    by="ticker", direction="backward"
                )
                merged.loc[miss, "rating"] = t_join["rating"].values
                merged.loc[miss, "ratingdate"] = t_join["ratingdate"].values
            except Exception:
                pass

    # Derived model features (these become real inputs to XGBoost)
    merged["rating_numeric"] = merged["rating"].apply(_to_numeric).astype("float32")
    merged["speculative_grade"] = (merged["rating_numeric"] >= 11).astype("Int64")          # BB+ and worse
    merged["rating_age_years"] = ((merged["datadate"] - pd.to_datetime(merged["ratingdate"])).dt.days / 365.25).clip(lower=0)

    # Downgrade signal (very strong predictor)
    merged = merged.sort_values(["gvkey", "datadate"])
    merged["prev_r"] = merged.groupby("gvkey")["rating_numeric"].shift(1)
    merged["downgrade_1y"] = ((merged["rating_numeric"] > merged["prev_r"]) & merged["prev_r"].notna()).astype("Int64")

    merged = merged.drop(columns=["prev_r"], errors="ignore")

    covered = (~merged["rating_numeric"].isna()).sum()
    print(f"[ratings] Attached historical CapIQ ratings for {covered:,} firm-years "
          f"({covered/len(merged)*100:.1f}% coverage). Features: rating_numeric, speculative_grade, downgrade_1y, rating_age_years")
    return merged


def get_latest_rating(gvkey: str, ticker: Optional[str] = None) -> Optional[str]:
    """Used by the UI to show the most recent known rating."""
    res = get_historical_rating(gvkey, datetime.now(), ticker)
    return res["rating"] if res else None


def get_historical_rating(gvkey: str, as_of: str, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """as_of can be a string or datetime (we convert inside)."""
    """Return the rating that was known on or before a specific date (for UI or analysis)."""
    gv = str(gvkey).zfill(6)
    asof = pd.to_datetime(as_of)
    ratings = get_ratings_df(live=False)
    if ratings.empty:
        return None

    # gvkey
    c = ratings[(ratings["gvkey"] == gv) & (ratings["ratingdate"] <= asof)]
    if not c.empty:
        r = c.sort_values("ratingdate").iloc[-1]
        return {"rating": r["rating"], "as_of": r["ratingdate"], "matched_by": "gvkey"}

    # ticker fallback
    if ticker:
        t = str(ticker).upper().strip()
        c = ratings[(ratings["ticker"] == t) & (ratings["ratingdate"] <= asof)]
        if not c.empty:
            r = c.sort_values("ratingdate").iloc[-1]
            return {"rating": r["rating"], "as_of": r["ratingdate"], "matched_by": "ticker"}
    return None