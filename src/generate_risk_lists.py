#!/usr/bin/env python3
"""
Regenerate the Top Risk Lists using the unified hazard model.

Run AFTER `save_best_models.py` finishes:
    python src/generate_risk_lists.py

Scores each company twice (horizon=12, horizon=60) with the single calibrated
model, enforces monotonicity as a safety net, and saves CSV lists.
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


def score_at_horizon(model, X_base, feature_cols, horizon_months):
    """Score companies at a specific horizon using the unified model."""
    X = X_base[feature_cols].fillna(0).astype("float32").copy()
    X["horizon_months"] = np.float32(horizon_months)
    return model.predict_proba(X)[:, 1]


def enrich_with_tickers(df, company_idx):
    """Attach ticker and industry from the company index."""
    if company_idx is None:
        return df
    merge_cols = ["gvkey", "ticker"]
    if "industry" in company_idx.columns:
        merge_cols.append("industry")
    df = df.merge(company_idx[merge_cols], on="gvkey", how="left")
    df["ticker"] = df["ticker"].fillna("").str.upper()
    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("")
    return df


def main():
    print("Loading unified calibrated model...")

    model = joblib.load(MODELS_DIR / "unified_calibrated.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Base feature cols = everything except horizon_months (we add it per-score)
    base_features = [c for c in feature_cols if c != "horizon_months"]

    test = pd.read_parquet(DATA_PROC / "monthly_test.parquet")
    test["gvkey"] = test["gvkey"].astype(str).str.strip()

    # Recent-data filter
    recent_mask = test["datadate"] >= "2025-12-01"
    recent_test = test[recent_mask].copy()
    if len(recent_test) < 80:
        print("Not enough Dec 2025+ data, relaxing to mid-2025+.")
        recent_test = test[test["datadate"] >= "2025-06-01"].copy()

    # Company index for tickers
    try:
        company_idx = pd.read_parquet(MODELS_DIR / "company_index.parquet")
        company_idx["gvkey"] = company_idx["gvkey"].astype(str).str.strip()
    except Exception:
        company_idx = None

    print(f"Scoring {recent_test['gvkey'].nunique():,} companies at both horizons...")

    # === Latest observation per company ===
    latest = recent_test.sort_values("datadate").groupby("gvkey").tail(1).copy()

    # Score at both horizons
    latest["pred_risk_12m"] = score_at_horizon(model, latest, base_features, 12)
    latest["pred_risk_5y_raw"] = score_at_horizon(model, latest, base_features, 60)

    # Monotonicity safety net (should be ~100% natural with unified model)
    latest["pred_risk_5y"] = np.maximum(latest["pred_risk_5y_raw"], latest["pred_risk_12m"])
    mono_pct = (latest["pred_risk_5y_raw"] >= latest["pred_risk_12m"] - 1e-6).mean() * 100
    print(f"Monotonicity (5y >= 12m): {mono_pct:.1f}%")

    # === 12-Month list ===
    top_12m = latest.sort_values("pred_risk_12m", ascending=False)[
        ["gvkey", "conm", "datadate", "pred_risk_12m"]
    ].copy()
    top_12m = top_12m.rename(columns={"pred_risk_12m": "pred_risk_2026", "datadate": "latest_observed_date"})
    top_12m["risk_horizon"] = "Dec 2025 - Dec 2026 (12 months from last fiscal year)"
    top_12m = enrich_with_tickers(top_12m, company_idx)

    cols_12m = ["gvkey", "conm", "ticker", "industry", "latest_observed_date", "risk_horizon", "pred_risk_2026"]
    top_12m = top_12m[[c for c in cols_12m if c in top_12m.columns]]
    top_12m.to_csv(OUT_12M / "top_2026_companies.csv", index=False)

    print(f"\n12-Month Top 10:")
    print(top_12m.head(10).to_string(index=False))

    # === 5-Year list (alive companies only) ===
    alive = latest[latest.get("bankrupt_delist", 0) == 0].copy()
    top_5y = alive.sort_values("pred_risk_5y", ascending=False)[
        ["gvkey", "conm", "datadate", "pred_risk_5y"]
    ].copy()
    top_5y = top_5y.rename(columns={"datadate": "latest_observed_date"})
    top_5y["risk_window"] = "Dec 2025 - Dec 2030 (5 years from last fiscal year, Alive Companies Only)"
    top_5y = enrich_with_tickers(top_5y, company_idx)

    cols_5y = ["gvkey", "conm", "ticker", "industry", "latest_observed_date", "risk_window", "pred_risk_5y"]
    top_5y = top_5y[[c for c in cols_5y if c in top_5y.columns]]
    top_5y.to_csv(OUT_5Y / "top_2026_risk_5y_2026data_only_alive.csv", index=False)

    print(f"\n5-Year Top 10:")
    print(top_5y.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
