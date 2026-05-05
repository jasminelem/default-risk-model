"""
FastAPI backend for the Corporate Credit Risk Intelligence UI.

Features:
- Real-time company typeahead search (name or gvkey)
- Dual-horizon predictions (12-month + 5-year) using the best saved models
- Per-company SHAP explanations (top driving features)
- Clean JSON responses for a beautiful frontend

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import joblib
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import shap
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROC = PROJECT_ROOT / "data" / "processed"

app = FastAPI(title="Pulse Credit Risk", description="Corporate Default Risk Intelligence", version="1.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the beautiful frontend
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")


# ============== LOAD EVERYTHING AT STARTUP (fast inference) ==============
print("Loading production models and data for the Credit Risk UI...")

# Unified hazard model: single model, horizon is a feature
model = joblib.load(MODELS_DIR / "unified_model.joblib")

# Feature columns (includes horizon_months)
with open(MODELS_DIR / "feature_cols.json") as f:
    FEATURE_COLS: List[str] = json.load(f)

# Base features = everything except horizon_months
BASE_FEATURES = [c for c in FEATURE_COLS if c != "horizon_months"]

# All prediction horizons (months)
HORIZONS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

# Best params (for display)
with open(MODELS_DIR / "best_params.json") as f:
    BEST_PARAMS = json.load(f)

# Company index for lightning-fast typeahead
print(">>> LOADING company_index from:", (MODELS_DIR / "company_index.parquet").resolve())
company_index = pd.read_parquet(MODELS_DIR / "company_index.parquet")
zy = company_index[company_index["gvkey"].astype(str).str.zfill(6) == "038804"]
print(">>> ZYVERSA ticker in loaded index:", zy["ticker"].values[0] if len(zy) > 0 else "NOT FOUND")
print(">>> Total companies loaded:", len(company_index))

# Full latest data for feature extraction (we keep the test panel + fall back to full if needed)
try:
    full_latest = pd.read_parquet(DATA_PROC / "monthly_test.parquet")
except Exception:
    full_latest = pd.read_parquet(DATA_PROC / "monthly_panel_after_attachments.parquet")

# === CRITICAL: Normalize gvkey to string everywhere for reliable matching ===
full_latest["gvkey"] = full_latest["gvkey"].astype(str).str.strip().str.zfill(6)
company_index["gvkey"] = company_index["gvkey"].astype(str).str.strip().str.zfill(6)

# Defensive: never allow target columns into features
TARGET_COLS = {"default_in_next_12m", "default_in_next_5y"}
FEATURE_COLS = [c for c in FEATURE_COLS if c not in TARGET_COLS]
BASE_FEATURES = [c for c in BASE_FEATURES if c not in TARGET_COLS]

# Build delisted company lookup (gvkeys with bankrupt_delist=1)
_delist_latest = full_latest.sort_values("datadate").groupby("gvkey").tail(1)
DELISTED_GVKEYS = set(
    _delist_latest.loc[_delist_latest.get("bankrupt_delist", 0) == 1, "gvkey"].values
)
print(f">>> {len(DELISTED_GVKEYS)} companies flagged as delisted/bankrupt")

# Single SHAP explainer
explainer = shap.TreeExplainer(model)

# Load feature translations for nice SHAP explanations
try:
    feature_defs = pd.read_csv(PROJECT_ROOT / "outputs" / "feature_definitions.csv")
    FEATURE_MEANINGS = dict(zip(feature_defs["feature"], feature_defs["business_meaning"]))
except Exception:
    FEATURE_MEANINGS = {}
    print("Warning: Could not load feature_definitions.csv — raw names will be shown in SHAP.")

# === Load historical ratings for rich UI history view ===
RATING_MAP = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13, "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19, "CC": 20, "C": 21, "D": 22, "SD": 23
}

try:
    ratings_df = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "ciq_ratings.parquet")
    ratings_df["gvkey"] = ratings_df["gvkey"].astype(str).str.strip().str.zfill(6)
    ratings_df["ratingdate"] = pd.to_datetime(ratings_df["ratingdate"], errors="coerce")
    print(f">>> Loaded {len(ratings_df):,} historical S&P/CapIQ ratings for UI")
except Exception as e:
    ratings_df = pd.DataFrame(columns=["gvkey", "ratingdate", "rating"])
    print(f">>> Ratings history not loaded (UI will show limited info): {e}")

# === Pre-compute calibrated 12m PDs for all companies (used by peer comparison) ===
print("Pre-computing 12m PDs for all companies (peer comparison)...")
_latest_per_company = full_latest.sort_values("datadate").groupby("gvkey").tail(1)
_X_base = _latest_per_company[BASE_FEATURES].fillna(0).infer_objects(copy=False).astype("float32")
import numpy as np
_X_12m = _X_base.copy()
_X_12m["horizon_months"] = np.float32(12)
_pd_all = model.predict_proba(_X_12m)[:, 1]
all_company_pds = pd.DataFrame({
    "gvkey": _latest_per_company["gvkey"].values,
    "pd_12m": _pd_all,
    "conm": _latest_per_company["conm"].values if "conm" in _latest_per_company.columns else "",
})
# Attach industry + sic4 from company_index
_idx_cols = ["gvkey", "industry"]
if "sic4" in company_index.columns:
    _idx_cols.append("sic4")
all_company_pds = all_company_pds.merge(
    company_index[_idx_cols].drop_duplicates("gvkey"),
    on="gvkey", how="left"
)
all_company_pds["industry"] = all_company_pds["industry"].fillna("")
if "sic4" not in all_company_pds.columns:
    all_company_pds["sic4"] = 0
all_company_pds["sic4"] = all_company_pds["sic4"].fillna(0).astype(int)
print(f"Pre-computed PDs for {len(all_company_pds):,} companies across {all_company_pds['industry'].nunique()} industries.")

# === Load pre-computed calibration tables (generated by src/build_calibration_table.py) ===
CALIBRATION_TABLES: Dict[str, Any] = {}
for horizon in ("12m", "5y"):
    cal_path = PROJECT_ROOT / "outputs" / f"calibration_table_{horizon}.json"
    if cal_path.exists():
        with open(cal_path) as f:
            CALIBRATION_TABLES[horizon] = json.load(f)
        print(f">>> Loaded calibration table for {horizon} ({len(CALIBRATION_TABLES[horizon])} buckets)")
    else:
        print(f">>> Calibration table for {horizon} not found (run src/build_calibration_table.py to generate)")

print(f"Loaded {len(company_index):,} companies with normalized gvkeys. UI ready.")


def _get_latest_row(gvkey: str) -> pd.Series:
    """Return the most recent row for a gvkey. GVKEY is always treated as string."""
    gvkey = str(gvkey).strip()
    matches = full_latest[full_latest["gvkey"] == gvkey]
    if len(matches) == 0:
        # Try a more forgiving match (sometimes padding differs)
        matches = full_latest[full_latest["gvkey"].str.zfill(6) == gvkey.zfill(6)]
    if len(matches) == 0:
        raise HTTPException(404, f"No data found for gvkey {gvkey}")
    return matches.sort_values("datadate").iloc[-1]


def _normalize_ticker_for_yf(ticker: str) -> str:
    """Convert Compustat-style tickers to Yahoo Finance format.
    Compustat uses '.' for share classes (BRK.B) — Yahoo uses '-' (BRK-B).
    Compustat appends '.1', '.2' etc. for historical versions — strip those.
    """
    import re
    t = ticker.strip().upper()
    # Strip Compustat version suffixes like .1, .2, .3
    t = re.sub(r'\.\d+$', '', t)
    # Convert share class dots to dashes: BRK.B -> BRK-B, MOG.A -> MOG-A
    t = t.replace('.', '-')
    return t


def _get_live_market_data(ticker: str) -> Dict[str, Any]:
    """Fetch current stock price, basic stats, and 6-month price history using yfinance."""
    if not ticker or not ticker.strip():
        return {}

    tkr = _normalize_ticker_for_yf(ticker)
    try:
        stock = yf.Ticker(tkr)
        info = stock.info

        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if price is None:
            return {}

        change = info.get('regularMarketChangePercent', 0)
        market_cap = info.get('marketCap')
        volume = info.get('volume') or info.get('regularMarketVolume')

        # Fetch 6 months of daily prices for the chart
        hist = stock.history(period="6mo", interval="1d")
        price_history = []
        if not hist.empty:
            for idx, row in hist.iterrows():
                price_history.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 4) if float(row["Close"]) < 1 else round(float(row["Close"]), 2)
                })

        return {
            "current_price": round(float(price), 4) if price and float(price) < 1 else (round(float(price), 2) if price else None),
            "change_percent": round(float(change), 2) if change else 0,
            "market_cap": int(market_cap) if market_cap else None,
            "volume": int(volume) if volume else None,
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            "price_history": price_history,          # list of {date, close}
        }
    except Exception:
        return {}  # Graceful failure — never break the main credit risk view


def get_rating_history(gvkey: str, max_history: int = 8) -> Dict[str, Any]:
    """Return rich rating info for the UI: fiscal-year rating, latest rating, downgrade flag, and history table."""
    gv = str(gvkey).strip().zfill(6)
    if ratings_df.empty or "gvkey" not in ratings_df.columns:
        return {
            "fiscal_rating": None,
            "latest_rating": None,
            "recent_downgrade": False,
            "rating_history": []
        }

    comp_ratings = ratings_df[ratings_df["gvkey"] == gv].dropna(subset=["ratingdate", "rating"]).copy()
    if comp_ratings.empty:
        return {"fiscal_rating": None, "latest_rating": None, "recent_downgrade": False, "rating_history": []}

    # Latest known rating
    latest_row = comp_ratings.sort_values("ratingdate").iloc[-1]
    latest_rating = str(latest_row["rating"]).upper()

    # Fiscal year-end rating (the one used by the model)
    try:
        row = _get_latest_row(gv)
        fiscal_dt = pd.to_datetime(row.get("datadate"))
        as_of = comp_ratings[comp_ratings["ratingdate"] <= fiscal_dt]
        fiscal_rating = str(as_of.sort_values("ratingdate").iloc[-1]["rating"]).upper() if not as_of.empty else latest_rating
    except Exception:
        fiscal_rating = latest_rating

    # Build clean recent history (oldest → newest for nice table)
    hist_df = comp_ratings.sort_values("ratingdate", ascending=False).head(max_history)
    history = []
    prev_num = None
    for _, r in hist_df.iloc[::-1].iterrows():   # reverse to oldest first
        rstr = str(r["rating"]).upper()
        num = RATING_MAP.get(rstr, 99)
        change = ""
        if prev_num is not None:
            if num < prev_num:
                change = "↑ Improved"
            elif num > prev_num:
                change = "↓ Downgraded"
            else:
                change = "→ Stable"
        is_ig = num <= 10
        history.append({
            "year": int(r["ratingdate"].year),
            "date": str(r["ratingdate"].date()),
            "rating": rstr,
            "grade": "Investment Grade" if is_ig else "Speculative Grade",
            "change": change
        })
        prev_num = num

    # Recent downgrade flag (last 24 months)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
    recent = comp_ratings[comp_ratings["ratingdate"] >= cutoff].sort_values("ratingdate")
    recent_downgrade = False
    if len(recent) >= 2:
        nums = [RATING_MAP.get(str(x).upper(), 99) for x in recent["rating"]]
        recent_downgrade = nums[-1] > nums[0]

    return {
        "fiscal_rating": fiscal_rating,
        "latest_rating": latest_rating,
        "recent_downgrade": recent_downgrade,
        "rating_history": history
    }


def _predict_and_explain(row: pd.Series) -> Dict[str, Any]:
    """Core inference + SHAP for one company row using the unified hazard model."""
    X_base = row[BASE_FEATURES].fillna(0).infer_objects(copy=False).astype("float32")
    base_vals = list(X_base.values)

    # PD term structure: score at every horizon
    pd_curve = []
    prev_pd = 0.0
    for h in HORIZONS:
        X_h = pd.DataFrame([base_vals + [np.float32(h)]], columns=FEATURE_COLS)
        raw_pd = float(model.predict_proba(X_h)[0, 1])
        pd_val = max(raw_pd, prev_pd)  # monotonicity safety net
        pd_curve.append({"horizon": h, "pd": round(pd_val, 6)})
        prev_pd = pd_val

    prob_12m = pd_curve[1]["pd"]  # index 1 = 12m
    prob_5y = pd_curve[9]["pd"]   # index 9 = 60m

    # SHAP at yearly horizons (12, 24, 36, 48, 60 months)
    SHAP_HORIZONS = [12, 24, 36, 48, 60]
    shap_by_year = {}
    for h in SHAP_HORIZONS:
        X_h = pd.DataFrame([base_vals + [np.float32(h)]], columns=FEATURE_COLS)
        shap_by_year[h] = explainer.shap_values(X_h)[0]

    def top_features(shap_vals, top_k=10):
        contrib = [(f, v) for f, v in zip(FEATURE_COLS, shap_vals) if f != "horizon_months"]
        contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        return [
            {
                "feature": f,
                "meaning": FEATURE_MEANINGS.get(f, f.replace("_", " ").title()),
                "shap_value": round(float(v), 4),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
                "impact": abs(round(float(v), 4)),
            }
            for f, v in contrib
        ]

    # Build SHAP for each year
    shap_years = {}
    for h in SHAP_HORIZONS:
        label = f"{h // 12}Y"
        shap_years[label] = top_features(shap_by_year[h])

    return {
        "prob_12m": round(prob_12m, 4),
        "prob_5y": round(prob_5y, 4),
        "pd_curve": pd_curve,
        "risk_level_12m": "High" if prob_12m > 0.55 else ("Medium" if prob_12m > 0.30 else "Low"),
        "risk_level_5y": "High" if prob_5y > 0.40 else ("Medium" if prob_5y > 0.18 else "Low"),
        "top_features_12m": shap_years.get("1Y", []),
        "top_features_5y": shap_years.get("5Y", []),
        "shap_years": shap_years,
        "feature_count": len(BASE_FEATURES),
    }


# ============== API ENDPOINTS ==============

@app.get("/api/search")
def search_companies(q: str = "", limit: int = 12) -> List[Dict[str, Any]]:
    """Real-time typeahead. Search by company name, ticker, or gvkey."""
    if not q or len(q) < 1:
        return []

    q_upper = q.upper().strip()
    matches = company_index[
        company_index["search_text"].str.contains(q_upper, na=False)
        | company_index["conm"].str.upper().str.contains(q_upper, na=False)
        | company_index.get("ticker", pd.Series([""]*len(company_index))).str.upper().str.contains(q_upper, na=False)
        | company_index["gvkey"].astype(str).str.contains(q_upper, na=False)
    ].head(limit)

    results = []
    for _, r in matches.iterrows():
        ticker = str(r.get("ticker", "")).upper() if pd.notna(r.get("ticker")) else ""
        cr = str(r.get("credit_rating", "")).strip().upper() if pd.notna(r.get("credit_rating")) else ""
        industry = str(r.get("industry", "")).strip() if pd.notna(r.get("industry")) else ""
        results.append({
            "gvkey": str(r["gvkey"]),
            "conm": r["conm"],
            "ticker": ticker,
            "credit_rating": cr,
            "industry": industry,
            "latest_datadate": str(r["latest_datadate"].date()) if hasattr(r["latest_datadate"], "date") else str(r["latest_datadate"]),
        })
    return results


@app.get("/api/predict/{gvkey}")
def predict_company(gvkey: str) -> Dict[str, Any]:
    """Return full dual-horizon prediction + SHAP explanations for one company."""
    row = _get_latest_row(gvkey)

    result = _predict_and_explain(row)

    # Get ticker + credit rating from company index (padding-tolerant match)
    gvkey_norm = str(gvkey).strip().zfill(6)
    ticker_row = company_index[company_index["gvkey"] == gvkey_norm]
    if len(ticker_row) == 0:
        ticker_row = company_index[company_index["gvkey"].str.zfill(6) == gvkey_norm]
    ticker = str(ticker_row["ticker"].iloc[0]).upper() if len(ticker_row) > 0 and pd.notna(ticker_row["ticker"].iloc[0]) else ""
    credit_rating = ""
    if len(ticker_row) > 0 and "credit_rating" in ticker_row.columns:
        val = ticker_row["credit_rating"].iloc[0]
        credit_rating = str(val).strip().upper() if pd.notna(val) else ""
    industry = ""
    if len(ticker_row) > 0 and "industry" in ticker_row.columns:
        val = ticker_row["industry"].iloc[0]
        industry = str(val).strip() if pd.notna(val) else ""

    # Fetch live market data (price, market cap, etc.)
    live_data = _get_live_market_data(ticker) if ticker else {}

    result.update({
        "gvkey": str(gvkey),
        "conm": str(row.get("conm", "Unknown")),
        "ticker": ticker,
        "credit_rating": credit_rating,
        "industry": industry,
        "delisted": gvkey_norm in DELISTED_GVKEYS,
        "latest_datadate": str(row.get("datadate", "")),
        "best_params": BEST_PARAMS,
        **live_data,   # current_price, change_percent, market_cap, volume, as_of
    })

    # === Rich CapIQ/S&P Rating History for UI ===
    try:
        rating_info = get_rating_history(gvkey)
        result.update(rating_info)   # adds fiscal_rating, latest_rating, recent_downgrade, rating_history
    except Exception as e:
        print(f"Rating history lookup failed for {gvkey}: {e}")

    return result


@app.get("/api/peers/{gvkey}")
def peer_comparison(gvkey: str) -> Dict[str, Any]:
    """Return peer group PD statistics using tiered SIC matching.

    Tries 4-digit SIC first (e.g. SIC 5961 = Catalog/Mail-Order).
    Falls back to 2-digit industry (e.g. Misc Retail) if the 4-digit
    group has fewer than 5 peers.
    """
    gvkey_norm = str(gvkey).strip().zfill(6)
    MIN_PEERS = 5

    idx_row = company_index[company_index["gvkey"] == gvkey_norm]
    if len(idx_row) == 0:
        raise HTTPException(404, f"Company {gvkey} not found in index")

    industry = str(idx_row["industry"].iloc[0]).strip()
    sic4 = int(idx_row["sic4"].iloc[0]) if "sic4" in idx_row.columns and pd.notna(idx_row["sic4"].iloc[0]) else 0

    if not industry and sic4 == 0:
        return {"error": "no_industry", "peer_count": 0}

    # Tiered matching: 4-digit SIC first, then 2-digit industry
    peers = pd.DataFrame()
    peer_label = industry
    if sic4 > 0:
        peers_4d = all_company_pds[all_company_pds["sic4"] == sic4]
        if len(peers_4d) >= MIN_PEERS:
            peers = peers_4d.copy()
            peer_label = f"SIC {sic4} ({industry})"

    if len(peers) < MIN_PEERS and industry:
        peers = all_company_pds[all_company_pds["industry"] == industry].copy()
        peer_label = industry

    if len(peers) < 2:
        return {"industry": peer_label, "peer_count": len(peers)}

    # This company's PD
    company_row = peers[peers["gvkey"] == gvkey_norm]
    if len(company_row) == 0:
        try:
            row = _get_latest_row(gvkey)
            X_s = row[BASE_FEATURES].fillna(0).infer_objects(copy=False).astype("float32")
            X_12m = pd.DataFrame([list(X_s.values) + [np.float32(12)]], columns=FEATURE_COLS)
            company_pd = float(model.predict_proba(X_12m)[0, 1])
        except Exception:
            return {"industry": peer_label, "peer_count": len(peers)}
    else:
        company_pd = float(company_row["pd_12m"].iloc[0])

    pds = peers["pd_12m"].values
    percentile = float((pds < company_pd).sum() / len(pds) * 100)

    # 5 closest peers by PD (excluding the company itself)
    others = peers[peers["gvkey"] != gvkey_norm].copy()
    others["dist"] = (others["pd_12m"] - company_pd).abs()
    closest = others.nsmallest(5, "dist")[["gvkey", "conm", "pd_12m"]]

    return {
        "industry": peer_label,
        "peer_count": int(len(peers)),
        "company_pd": round(company_pd, 6),
        "percentile_rank": round(percentile, 1),
        "peer_median_pd": round(float(peers["pd_12m"].median()), 6),
        "peer_min_pd": round(float(peers["pd_12m"].min()), 6),
        "peer_max_pd": round(float(peers["pd_12m"].max()), 6),
        "closest_peers": [
            {"gvkey": r["gvkey"], "conm": str(r["conm"]), "pd_12m": round(float(r["pd_12m"]), 6)}
            for _, r in closest.iterrows()
        ],
    }


@app.get("/api/calibration/{horizon}")
def calibration_table(horizon: str) -> List[Dict[str, Any]]:
    """Return the backtesting calibration table for a given horizon (12m or 5y)."""
    if horizon not in CALIBRATION_TABLES:
        raise HTTPException(404, f"Calibration table for '{horizon}' not found. Run src/build_calibration_table.py first.")
    return CALIBRATION_TABLES[horizon]


@app.get("/api/industries")
def industry_ranking() -> List[Dict[str, Any]]:
    """Return all industries ranked by median 12-month PD, with per-industry top 3 features."""
    if all_company_pds.empty or "industry" not in all_company_pds.columns:
        return []

    valid = all_company_pds[all_company_pds["industry"].str.strip() != ""].copy()
    if valid.empty:
        return []

    grouped = valid.groupby("industry")["pd_12m"].agg(
        median_pd="median", count="count"
    ).reset_index()

    # Filter out tiny industries (< 5 companies) — too noisy to be meaningful
    grouped = grouped[grouped["count"] >= 5]

    # Sort by median PD descending, break ties by firm count (larger industries first)
    grouped = grouped.sort_values(["median_pd", "count"], ascending=[False, False])

    return [
        {
            "industry": str(r["industry"]),
            "median_pd": round(float(r["median_pd"]), 6),
            "count": int(r["count"]),
        }
        for _, r in grouped.iterrows()
    ]


@app.get("/api/industry_detail")
def industry_detail(industry: str = "", limit: int = 10, feat_limit: int = 10) -> Dict[str, Any]:
    """Return example companies + SHAP feature importance for a specific industry."""
    industry = industry.strip()
    if not industry:
        raise HTTPException(400, "industry parameter required")

    # Find companies in this industry
    ind_companies = all_company_pds[all_company_pds["industry"] == industry].copy()
    if ind_companies.empty:
        raise HTTPException(404, f"No companies found for industry '{industry}'")

    # Top companies by PD (highest risk first) + lowest risk
    top_risk = ind_companies.nlargest(limit, "pd_12m")
    low_risk = ind_companies.nsmallest(min(5, len(ind_companies)), "pd_12m")

    # Attach tickers
    ticker_map = company_index.set_index("gvkey")["ticker"].to_dict() if "ticker" in company_index.columns else {}

    def _company_list(df):
        return [
            {
                "gvkey": str(r["gvkey"]),
                "conm": str(r["conm"]),
                "ticker": str(ticker_map.get(r["gvkey"], "")).upper(),
                "pd_12m": round(float(r["pd_12m"]), 6),
            }
            for _, r in df.iterrows()
        ]

    # SHAP feature importance for this industry
    ind_gvkeys = set(ind_companies["gvkey"].values)
    pool = _latest_per_company[_latest_per_company["gvkey"].isin(ind_gvkeys)]
    features = []
    sample_size = 0
    try:
        sample = pool.sample(n=min(200, len(pool)), random_state=42)
        sample_size = len(sample)
        X_s = sample[BASE_FEATURES].fillna(0).infer_objects(copy=False).astype("float32")
        X_s["horizon_months"] = np.float32(12)
        sv = explainer.shap_values(X_s)
        mean_abs = np.abs(sv).mean(axis=0)
        top_feats = sorted(
            [(f, float(v)) for f, v in zip(FEATURE_COLS, mean_abs) if f != "horizon_months"],
            key=lambda x: x[1], reverse=True
        )[:feat_limit]
        features = [
            {"feature": f, "meaning": FEATURE_MEANINGS.get(f, f.replace("_", " ").title()), "importance": round(v, 4)}
            for f, v in top_feats
        ]
    except Exception:
        pass

    return {
        "industry": industry,
        "total_companies": len(ind_companies),
        "median_pd": round(float(ind_companies["pd_12m"].median()), 6),
        "mean_pd": round(float(ind_companies["pd_12m"].mean()), 6),
        "highest_risk": _company_list(top_risk),
        "lowest_risk": _company_list(low_risk),
        "features": features,
        "sample_size": sample_size,
    }


@app.get("/api/models/info")
def model_info():
    """Return metadata about the hosted models."""
    return {
        "model": "unified_hazard",
        "horizons": [12, 60],
        "params": BEST_PARAMS,
        "feature_count": len(BASE_FEATURES),
        "companies_indexed": len(company_index),
    }


@app.get("/debug/files")
def debug_files():
    import os
    base = PROJECT_ROOT
    def safe_list(p):
        try: return os.listdir(p)
        except: return "NOT FOUND"
    return {
        "unified_model_exists": (base / "models" / "unified_calibrated.joblib").exists(),
        "company_index_exists": (base / "models" / "company_index.parquet").exists(),
        "top_12m_csv_exists": (base / "outputs" / "12m_model" / "top_2026_companies.csv").exists(),
        "top_5y_csv_exists": (base / "outputs" / "5y_model" / "top_2026_risk_5y_2026data_only_alive.csv").exists(),
        "model_files": safe_list(base / "models"),
    }


# === Top Risk Lists (for the clean main dashboard) ===
def _safe_records(df: pd.DataFrame):
    """Most reliable sanitizer - sanitizes AFTER to_dict to guarantee no NaN/inf reaches JSON."""
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        new_row = {}
        for k, v in row.items():
            if pd.isna(v):
                new_row[k] = None
            elif isinstance(v, float) and not math.isfinite(v):
                new_row[k] = None
            else:
                new_row[k] = v
        cleaned.append(new_row)
    return cleaned


def _enrich_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """Attach credit_rating and industry from company_index to a risk-list DataFrame."""
    if "gvkey" not in df.columns:
        return df
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    idx_cols = ["gvkey", "credit_rating"]
    if "industry" in company_index.columns:
        idx_cols.append("industry")
    idx = company_index[idx_cols].copy()
    idx["gvkey"] = idx["gvkey"].astype(str).str.strip().str.zfill(6)
    df = df.merge(idx, on="gvkey", how="left")
    df["credit_rating"] = df["credit_rating"].fillna("").astype(str).str.strip().str.upper()
    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("")
    return df


@app.get("/api/top/12m")
def top_12m(limit: int = 10):
    path = PROJECT_ROOT / "outputs" / "12m_model" / "top_2026_companies.csv"
    if not path.exists():
        return []
    df = _enrich_from_index(pd.read_csv(path)).head(limit)
    return _safe_records(df)


@app.get("/api/top/5y")
def top_5y(limit: int = 10):
    path = PROJECT_ROOT / "outputs" / "5y_model" / "top_2026_risk_5y_2026data_only_alive.csv"
    if not path.exists():
        return []
    df = _enrich_from_index(pd.read_csv(path)).head(limit)
    return _safe_records(df)


@app.get("/api/top/combined")
def top_combined(limit: int = 10):
    """Top companies ranked by 12-month PD, scored live from the raw model.
    Only includes companies with fiscal data from 2025 or later."""
    if all_company_pds.empty:
        return []

    # Filter to 2025+ fiscal dates only
    if "latest_datadate" in company_index.columns:
        recent_gvkeys = set(
            company_index[
                pd.to_datetime(company_index["latest_datadate"], errors="coerce") >= pd.Timestamp("2025-01-01")
            ]["gvkey"].values
        )
        pool = all_company_pds[all_company_pds["gvkey"].isin(recent_gvkeys)]
    else:
        pool = all_company_pds

    if pool.empty:
        pool = all_company_pds  # fallback if no 2025+ data

    df = pool.nlargest(limit, "pd_12m").copy()

    # Score 5-year PD live for the top N (must always be > 12m)
    pd_5y_list = []
    for _, r in df.iterrows():
        try:
            row = _get_latest_row(r["gvkey"])
            X_base = row[BASE_FEATURES].fillna(0).infer_objects(copy=False).astype("float32")
            X_60m = pd.DataFrame([list(X_base.values) + [np.float32(60)]], columns=FEATURE_COLS)
            p5y = float(model.predict_proba(X_60m)[0, 1])
            p12m = r["pd_12m"]
            pd_5y_list.append(max(p5y, p12m))
        except Exception:
            pd_5y_list.append(None)
    df["pred_risk_5y"] = pd_5y_list
    df = df.rename(columns={"pd_12m": "pred_risk_2026"})

    # Attach ticker + latest fiscal date from company_index
    if "ticker" in company_index.columns:
        ticker_map = company_index.set_index("gvkey")["ticker"].to_dict()
        df["ticker"] = df["gvkey"].map(ticker_map).fillna("")
    if "latest_datadate" in company_index.columns:
        date_map = company_index.set_index("gvkey")["latest_datadate"].to_dict()
        df["latest_datadate"] = df["gvkey"].map(date_map)

    # Attach delisted flag
    df["delisted"] = df["gvkey"].isin(DELISTED_GVKEYS)

    df = _enrich_from_index(df)
    return _safe_records(df)


@app.get("/api/top/industries")
def top_industries(horizon: str = "12m", limit: int = 10):
    """Return industries ranked by average default risk across their companies."""
    if horizon == "5y":
        path = PROJECT_ROOT / "outputs" / "5y_model" / "top_2026_risk_5y_2026data_only_alive.csv"
        risk_col = "pred_risk_5y"
    else:
        path = PROJECT_ROOT / "outputs" / "12m_model" / "top_2026_companies.csv"
        risk_col = "pred_risk_2026"
    if not path.exists():
        return []
    df = _enrich_from_index(pd.read_csv(path))
    if "industry" not in df.columns or df["industry"].str.strip().eq("").all():
        return []
    df = df[df["industry"].str.strip() != ""]
    grouped = df.groupby("industry").agg(
        avg_risk=(risk_col, "mean"),
        max_risk=(risk_col, "max"),
        company_count=(risk_col, "size"),
    ).sort_values("avg_risk", ascending=False).head(limit).reset_index()
    grouped["avg_risk"] = grouped["avg_risk"].round(4)
    grouped["max_risk"] = grouped["max_risk"].round(4)
    return _safe_records(grouped)


@app.get("/api/stock/{ticker}")
def get_stock_history(ticker: str, period: str = "1y"):
    """Return price history for a ticker. Tries hard to find data for microcap/distressed names."""
    if not ticker:
        return {"price_history": [], "ticker": ticker}

    sym = _normalize_ticker_for_yf(ticker)
    yf_period = {"12m": "1y", "5y": "5y", "all": "max"}.get(period.lower(), period)

    for attempt_period in [yf_period, "max"]:          # fallback to full history
        try:
            tkr = yf.Ticker(sym)
            hist = tkr.history(period=attempt_period, interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty:
                price_history = []
                for idx, row in hist.iterrows():
                    price_history.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": round(float(row["Close"]), 4) if float(row["Close"]) < 1 else round(float(row["Close"]), 2)
                    })
                return {"price_history": price_history, "period": period, "ticker": sym}
        except Exception:
            continue

    # Last resort: sometimes the ticker needs a suffix or is quoted elsewhere
    return {"price_history": [], "period": period, "ticker": sym}


# ============== FRONTEND ROUTES ==============

@app.get("/")
async def root():
    """Serve the beautiful Google/ChatGPT-style UI."""
    return FileResponse(str(PROJECT_ROOT / "app" / "static" / "index.html"))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
