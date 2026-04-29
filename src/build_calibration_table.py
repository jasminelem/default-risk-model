#!/usr/bin/env python3
"""
Build backtesting calibration tables using the unified hazard model.

IMPORTANT: Only uses observations with COMPLETE follow-up windows so that
actual default rates are not biased downward by right-censoring.
  - 12m table: uses all training data (all obs have 12m+ of follow-up)
  - 5y table:  uses only training obs with datadate <= 2020-12-31
               (ensuring a full 60-month window through end of data)

Output: outputs/calibration_table_12m.json, outputs/calibration_table_5y.json

Usage:
    python src/build_calibration_table.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PD_BINS = [0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90, 1.01]
PD_LABELS = [
    "0-1%", "1-3%", "3-5%", "5-10%", "10-20%",
    "20-30%", "30-50%", "50-75%", "75-90%", "90-100%",
]


def build_table(predicted_pds, actuals):
    df_eval = pd.DataFrame({"predicted_pd": predicted_pds, "actual": actuals})
    df_eval["bucket"] = pd.cut(
        df_eval["predicted_pd"], bins=PD_BINS, labels=PD_LABELS, right=False
    )

    rows = []
    for label in PD_LABELS:
        group = df_eval[df_eval["bucket"] == label]
        if len(group) == 0:
            continue
        count = int(len(group))
        actual_defaults = int(group["actual"].sum())
        actual_rate = float(group["actual"].mean())
        mean_pred = float(group["predicted_pd"].mean())
        ratio = round(actual_rate / mean_pred, 3) if mean_pred > 1e-8 else None
        rows.append({
            "bucket": label,
            "count": count,
            "actual_defaults": actual_defaults,
            "actual_default_rate": round(actual_rate, 6),
            "mean_predicted_pd": round(mean_pred, 6),
            "ratio": ratio,
        })
    return rows


def main():
    print("Loading unified model and historical data...")

    model = joblib.load(MODELS_DIR / "unified_calibrated.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    base_features = [c for c in feature_cols if c != "horizon_months"]

    train_path = DATA_PROC / "monthly_train.parquet"
    if not train_path.exists():
        print("Training data not found.")
        return
    train = pd.read_parquet(train_path)
    train["datadate"] = pd.to_datetime(train["datadate"])
    print(f"  Loaded training data: {len(train):,} rows ({train['datadate'].min().date()} to {train['datadate'].max().date()})")

    X_base = train[base_features].fillna(0).astype("float32")

    # 12-month: all training obs have complete 12m follow-up
    if "default_in_next_12m" in train.columns:
        X_12m = X_base.copy()
        X_12m["horizon_months"] = np.float32(12)
        preds_12m = model.predict_proba(X_12m)[:, 1]
        actuals_12m = train["default_in_next_12m"].values

        print(f"\n12m: scoring {len(train):,} obs (all have complete 12m follow-up)")
        print(f"  Mean predicted: {preds_12m.mean():.4%}  Actual default rate: {actuals_12m.mean():.4%}")

        table_12m = build_table(preds_12m, actuals_12m)
        out_path = OUTPUT_DIR / "calibration_table_12m.json"
        with open(out_path, "w") as f:
            json.dump(table_12m, f, indent=2)
        print(f"  Saved -> {out_path}")
        for row in table_12m:
            print(f"    {row['bucket']:>8s}  n={row['count']:>7,}  "
                  f"pred={row['mean_predicted_pd']:.4%}  actual={row['actual_default_rate']:.4%}  "
                  f"ratio={row['ratio']}")

    # 5-year: restrict to obs with complete 60-month follow-up
    if "default_in_next_5y" in train.columns:
        cutoff = train["datadate"].max() - pd.DateOffset(years=5)
        mask = train["datadate"] <= cutoff
        train_5y = train[mask]
        X_5y = X_base[mask].copy()
        X_5y["horizon_months"] = np.float32(60)
        preds_5y = model.predict_proba(X_5y)[:, 1]
        actuals_5y = train_5y["default_in_next_5y"].values

        print(f"\n5y: scoring {len(train_5y):,} obs (datadate <= {cutoff.date()}, complete 5y follow-up)")
        print(f"  Mean predicted: {preds_5y.mean():.4%}  Actual default rate: {actuals_5y.mean():.4%}")

        table_5y = build_table(preds_5y, actuals_5y)
        out_path = OUTPUT_DIR / "calibration_table_5y.json"
        with open(out_path, "w") as f:
            json.dump(table_5y, f, indent=2)
        print(f"  Saved -> {out_path}")
        for row in table_5y:
            print(f"    {row['bucket']:>8s}  n={row['count']:>7,}  "
                  f"pred={row['mean_predicted_pd']:.4%}  actual={row['actual_default_rate']:.4%}  "
                  f"ratio={row['ratio']}")

    print("\nCalibration tables generated (using only observations with complete follow-up windows).")


if __name__ == "__main__":
    main()
