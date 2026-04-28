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

# Models
model_12m = joblib.load(MODELS_DIR / "defaultinnext12m_model.joblib")
model_5y = joblib.load(MODELS_DIR / "defaultinnext5y_model.joblib")

# Feature columns (identical for both)
with open(MODELS_DIR / "feature_cols.json") as f:
    FEATURE_COLS: List[str] = json.load(f)

# Best params (for display)
with open(MODELS_DIR / "best_params_defaultinnext12m.json") as f:
    BEST_PARAMS_12M = json.load(f)
with open(MODELS_DIR / "best_params_defaultinnext5y.json") as f:
    BEST_PARAMS_5Y = json.load(f)

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

# Defensive: never allow target columns into features (in case old feature_cols.json is loaded)
TARGET_COLS = {"default_in_next_12m", "default_in_next_5y"}
FEATURE_COLS = [c for c in FEATURE_COLS if c not in TARGET_COLS]

# SHAP explainers (created once)
explainer_12m = shap.TreeExplainer(model_12m)
explainer_5y = shap.TreeExplainer(model_5y)

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


def _get_live_market_data(ticker: str) -> Dict[str, Any]:
    """Fetch current stock price, basic stats, and 6-month price history using yfinance."""
    if not ticker or not ticker.strip():
        return {}

    tkr = ticker.strip().upper()
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
                    "close": round(float(row["Close"]), 2)
                })

        return {
            "current_price": round(float(price), 2) if price else None,
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
    """Core inference + SHAP for one company row."""
    # Robust feature vector — ensure exact columns in correct order, no target leakage
    X_series = row[FEATURE_COLS].fillna(0).infer_objects(copy=False).astype("float32")
    X_df = pd.DataFrame([X_series.values], columns=FEATURE_COLS)

    # 12-month
    prob_12m = float(model_12m.predict_proba(X_df)[0, 1])
    shap_12m = explainer_12m.shap_values(X_df)[0]

    # 5-year
    prob_5y = float(model_5y.predict_proba(X_df)[0, 1])
    shap_5y = explainer_5y.shap_values(X_df)[0]

    # Build top feature contributions (positive = increases risk) with human-readable meaning
    def top_features(shap_vals, top_k=12):
        contrib = sorted(zip(FEATURE_COLS, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]
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

    return {
        "prob_12m": round(prob_12m, 4),
        "prob_5y": round(prob_5y, 4),
        # Risk bucketing — user feedback: 26.4% 12m PD should be Low (not Medium).
        # "High" remains reserved for really distressed names (>55% 12-month default probability).
        "risk_level_12m": "High" if prob_12m > 0.55 else ("Medium" if prob_12m > 0.30 else "Low"),
        "risk_level_5y": "High" if prob_5y > 0.40 else ("Medium" if prob_5y > 0.18 else "Low"),
        "top_features_12m": top_features(shap_12m),
        "top_features_5y": top_features(shap_5y),
        "feature_count": len(FEATURE_COLS),
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
        results.append({
            "gvkey": str(r["gvkey"]),
            "conm": r["conm"],
            "ticker": ticker,
            "credit_rating": cr,
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

    # Fetch live market data (price, market cap, etc.)
    live_data = _get_live_market_data(ticker) if ticker else {}

    result.update({
        "gvkey": str(gvkey),
        "conm": str(row.get("conm", "Unknown")),
        "ticker": ticker,
        "credit_rating": credit_rating,
        "latest_datadate": str(row.get("datadate", "")),
        "best_params_12m": BEST_PARAMS_12M,
        "best_params_5y": BEST_PARAMS_5Y,
        **live_data,   # current_price, change_percent, market_cap, volume, as_of
    })

    # === Rich CapIQ/S&P Rating History for UI ===
    try:
        rating_info = get_rating_history(gvkey)
        result.update(rating_info)   # adds fiscal_rating, latest_rating, recent_downgrade, rating_history
    except Exception as e:
        print(f"Rating history lookup failed for {gvkey}: {e}")

    return result


@app.get("/api/models/info")
def model_info():
    """Return metadata about the hosted models."""
    return {
        "12m_model": {"target": "default_in_next_12m", "params": BEST_PARAMS_12M},
        "5y_model": {"target": "default_in_next_5y", "params": BEST_PARAMS_5Y},
        "feature_count": len(FEATURE_COLS),
        "companies_indexed": len(company_index),
    }


# === TEMPORARY DEBUG ENDPOINT (remove after fixing) ===
@app.get("/debug/files")
def debug_files():
    import os
    base = PROJECT_ROOT
    def safe_list(p): 
        try: return os.listdir(p)
        except: return "NOT FOUND"
    return {
        "company_index_exists": (base / "models" / "company_index.parquet").exists(),
        "top_12m_csv_exists": (base / "outputs" / "12m_model" / "top_2026_companies.csv").exists(),
        "top_5y_csv_exists": (base / "outputs" / "5y_model" / "top_2026_risk_5y_2026data_only_alive.csv").exists(),
        "outputs_12m_files": safe_list(base / "outputs" / "12m_model"),
        "outputs_5y_files": safe_list(base / "outputs" / "5y_model"),
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


@app.get("/api/top/12m")
def top_12m(limit: int = 10):
    path = PROJECT_ROOT / "outputs" / "12m_model" / "top_2026_companies.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    # Attach credit rating from the full index (for UI visibility)
    if "gvkey" in df.columns:
        df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
        idx = company_index[["gvkey", "credit_rating"]].copy()
        idx["gvkey"] = idx["gvkey"].astype(str).str.strip().str.zfill(6)
        df = df.merge(idx, on="gvkey", how="left")
        df["credit_rating"] = df["credit_rating"].fillna("").astype(str).str.strip().str.upper()
    return _safe_records(df)


@app.get("/api/top/5y")
def top_5y(limit: int = 10):
    path = PROJECT_ROOT / "outputs" / "5y_model" / "top_2026_risk_5y_2026data_only_alive.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    # Attach credit rating from the full index (for UI visibility)
    if "gvkey" in df.columns:
        df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
        idx = company_index[["gvkey", "credit_rating"]].copy()
        idx["gvkey"] = idx["gvkey"].astype(str).str.strip().str.zfill(6)
        df = df.merge(idx, on="gvkey", how="left")
        df["credit_rating"] = df["credit_rating"].fillna("").astype(str).str.strip().str.upper()
    return _safe_records(df)


@app.get("/api/stock/{ticker}")
def get_stock_history(ticker: str, period: str = "1y"):
    """Return price history for a ticker. Tries hard to find data for microcap/distressed names."""
    if not ticker:
        return {"price_history": [], "ticker": ticker}

    sym = ticker.strip().upper()
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
                        "close": round(float(row["Close"]), 2)
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
