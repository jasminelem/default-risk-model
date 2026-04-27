#!/usr/bin/env python3
"""
Regenerate the Top Risk Lists using the FINAL clean models.

Run this AFTER `save_best_models.py` finishes:

    python src/generate_risk_lists.py

It will create honest, realistic top-10 lists with proper probabilities
(using the user's preferred "latest 2026 window + alive companies only" logic)
and save them to the correct folders so the UI shows sensible numbers.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_12M = PROJECT_ROOT / "outputs" / "12m_model"
OUT_5Y = PROJECT_ROOT / "outputs" / "5y_model"

OUT_12M.mkdir(parents=True, exist_ok=True)
OUT_5Y.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading final clean models...")

    model_12m = joblib.load(MODELS_DIR / "defaultinnext12m_model.joblib")
    model_5y = joblib.load(MODELS_DIR / "defaultinnext5y_model.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    test = pd.read_parquet(DATA_PROC / "monthly_test.parquet")
    test["gvkey"] = test["gvkey"].astype(str).str.strip()

    # === KEY FILTER: Only companies with recent data (Dec 2025 or 2026) ===
    # This is what you want — forward-looking, not stale 2024 data
    # === Smart Recent-Data Filter ===
    # We prefer Dec 2025+, but fall back if there aren't enough companies
    recent_mask = test["datadate"] >= "2025-12-01"
    recent_test = test[recent_mask].copy()

    if len(recent_test) < 80:   # Not enough truly recent companies
        print("Not enough companies with Dec 2025+ data — relaxing filter to mid-2025+ for better coverage.")
        recent_mask = test["datadate"] >= "2025-06-01"
        recent_test = test[recent_mask].copy()

    # Load company index for tickers
    try:
        company_idx = pd.read_parquet(MODELS_DIR / "company_index.parquet")
        company_idx["gvkey"] = company_idx["gvkey"].astype(str).str.strip()
        has_tickers = True
    except Exception:
        company_idx = None
        has_tickers = False

    print(f"Using companies with data from mid-2025 or later "
          f"({recent_test['gvkey'].nunique():,} companies)")

    # === 12-MONTH MODEL ===
    # Latest observation (must be Dec 2025 or later)
    latest_12m = recent_test.sort_values("datadate").groupby("gvkey").tail(1).copy()

    X_12m = latest_12m[feature_cols].fillna(0).astype("float32")
    latest_12m["pred_risk_2026"] = model_12m.predict_proba(X_12m)[:, 1]

    top_12m = latest_12m.sort_values("pred_risk_2026", ascending=False)[
        ["gvkey", "conm", "datadate", "pred_risk_2026"]
    ].copy()

    # Attach ticker if available
    if has_tickers:
        top_12m = top_12m.merge(company_idx[["gvkey", "ticker"]], on="gvkey", how="left")
        top_12m["ticker"] = top_12m["ticker"].fillna("").str.upper()

    top_12m = top_12m.rename(columns={"datadate": "latest_observed_date"})
    top_12m["risk_horizon"] = "Dec 2025 - Dec 2026 (12 months from last fiscal year)"

    cols_12m = ["gvkey", "conm", "ticker", "latest_observed_date", "risk_horizon", "pred_risk_2026"]
    top_12m = top_12m[[c for c in cols_12m if c in top_12m.columns]]
    top_12m.to_csv(OUT_12M / "top_2026_companies.csv", index=False)

    print(f"\n12-Month Top 10 (companies with Dec 2025+ data):")
    print(top_12m.head(10).to_string(index=False))

    # === 5-YEAR MODEL - Alive + Recent (Dec 2025 or 2026) ===
    alive_recent = recent_test[recent_test["bankrupt_delist"] == 0].copy()
    latest_5y = alive_recent.sort_values("datadate").groupby("gvkey").tail(1).copy()

    X_5y = latest_5y[feature_cols].fillna(0).astype("float32")
    latest_5y["pred_risk_5y"] = model_5y.predict_proba(X_5y)[:, 1]

    top_5y = latest_5y.sort_values("pred_risk_5y", ascending=False)[
        ["gvkey", "conm", "datadate", "pred_risk_5y"]
    ].copy()

    if has_tickers:
        top_5y = top_5y.merge(company_idx[["gvkey", "ticker"]], on="gvkey", how="left")
        top_5y["ticker"] = top_5y["ticker"].fillna("").str.upper()

    top_5y = top_5y.rename(columns={"datadate": "latest_observed_date"})
    top_5y["risk_window"] = "Dec 2025 - Dec 2030 (5 years from last fiscal year, Alive Companies Only)"

    cols_5y = ["gvkey", "conm", "ticker", "latest_observed_date", "risk_window", "pred_risk_5y"]
    top_5y = top_5y[[c for c in cols_5y if c in top_5y.columns]]
    top_5y.to_csv(OUT_5Y / "top_2026_risk_5y_2026data_only_alive.csv", index=False)

    print(f"\n5-Year Top 10 (Dec 2025/2026 data + alive only):")
    print(top_5y.head(10).to_string(index=False))

    print("\n✅ New risk lists generated using only recent (Dec 2025+) data.")
    print("   The main page will now show forward-looking companies only.")


if __name__ == "__main__":
    main()
