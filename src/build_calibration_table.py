#!/usr/bin/env python3
"""
Build backtesting calibration tables using the unified hazard model.

Uses the VALIDATION set (out-of-sample data from 2022-2023) which has
complete follow-up windows for both 12-month and 5-year horizons.
The test set (2024-2026) is too recent for reliable calibration since
most firms haven't had time to default yet.

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


def print_table(table):
    for row in table:
        print(f"    {row['bucket']:>8s}  n={row['count']:>7,}  "
              f"pred={row['mean_predicted_pd']:.4%}  actual={row['actual_default_rate']:.4%}  "
              f"ratio={row['ratio']}")


def main():
    print("Loading unified model...")

    model = joblib.load(MODELS_DIR / "unified_model.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    base_features = [c for c in feature_cols if c != "horizon_months"]

    # Use VALIDATION set: out-of-sample (not in training), and recent enough
    # to have complete follow-up for 12-month outcomes (2022-2023 data with
    # outcomes observable through 2024).
    val_path = DATA_PROC / "monthly_val.parquet"
    if not val_path.exists():
        print("Validation data not found.")
        return
    val = pd.read_parquet(val_path)
    val["datadate"] = pd.to_datetime(val["datadate"])
    print(f"  Validation set: {len(val):,} rows ({val['datadate'].min().date()} to {val['datadate'].max().date()})")

    X_base = val[base_features].fillna(0).astype("float32")

    # 12-month calibration
    target_12m = "default_in_next_12m"
    if target_12m in val.columns:
        X_12m = X_base.copy()
        X_12m["horizon_months"] = np.float32(12)
        preds = model.predict_proba(X_12m)[:, 1]
        actuals = val[target_12m].values

        print(f"\n12-Month Calibration (on validation set, not used in training):")
        print(f"  {len(val):,} obs, predicted mean={preds.mean():.4%}, actual rate={actuals.mean():.4%}")

        table = build_table(preds, actuals)
        out_path = OUTPUT_DIR / "calibration_table_12m.json"
        with open(out_path, "w") as f:
            json.dump(table, f, indent=2)
        print_table(table)
        print(f"  Saved -> {out_path}")

    # 5-year calibration (only obs with complete 60-month follow-up)
    target_5y = "default_in_next_60m" if "default_in_next_60m" in val.columns else "default_in_next_5y"
    if target_5y in val.columns:
        cutoff = val["datadate"].max() - pd.DateOffset(years=5)
        mask = val["datadate"] <= cutoff
        val_5y = val[mask]

        if len(val_5y) > 0:
            X_5y = X_base[mask].copy()
            X_5y["horizon_months"] = np.float32(60)
            preds = model.predict_proba(X_5y)[:, 1]
            actuals = val_5y[target_5y].values

            print(f"\n5-Year Calibration (val obs with complete 5y follow-up):")
            print(f"  {len(val_5y):,} obs (datadate <= {cutoff.date()}), predicted mean={preds.mean():.4%}, actual rate={actuals.mean():.4%}")

            table = build_table(preds, actuals)
            out_path = OUTPUT_DIR / "calibration_table_5y.json"
            with open(out_path, "w") as f:
                json.dump(table, f, indent=2)
            print_table(table)
            print(f"  Saved -> {out_path}")
        else:
            print("\n5-Year: No validation observations with complete 5y follow-up.")

    print("\nDone. Calibration uses validation set (out-of-sample, complete follow-up).")


if __name__ == "__main__":
    main()
