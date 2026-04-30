#!/usr/bin/env python3
"""
Recalibrate the unified hazard model using out-of-sample validation data.

The existing CalibratedClassifierCV was fitted during training, so its
probability mapping is learned on in-sample data. This script fits fresh
IsotonicRegression calibrators on the validation set (data the model has
never seen), producing properly calibrated probabilities.

Saves:
  models/recal_12m.joblib  - IsotonicRegression for 12m horizon
  models/recal_5y.joblib   - IsotonicRegression for 5y horizon

Usage:
    python src/recalibrate.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("Loading model and validation data...")

    raw_model = joblib.load(MODELS_DIR / "unified_model.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    base_features = [c for c in feature_cols if c != "horizon_months"]

    val = pd.read_parquet(DATA_PROC / "monthly_val.parquet")
    print(f"  Val set: {len(val):,} rows ({val['datadate'].min().date()} to {val['datadate'].max().date()})")

    X_base = val[base_features].fillna(0).astype("float32")

    for horizon, target, months in [("12m", "default_in_next_12m", 12), ("5y", "default_in_next_5y", 60)]:
        print(f"\n{'='*50}")
        print(f"  {horizon} recalibration (horizon_months={months})")
        print(f"{'='*50}")

        X = X_base.copy()
        X["horizon_months"] = np.float32(months)

        y = val[target].values
        raw_probs = raw_model.predict_proba(X)[:, 1]

        print(f"  Raw model mean pred: {raw_probs.mean():.4%}  actual: {y.mean():.4%}")
        print(f"  Raw Brier:  {brier_score_loss(y, raw_probs):.6f}")
        print(f"  Raw AUC:    {roc_auc_score(y, raw_probs):.4f}")

        # Fit isotonic regression: maps raw_probs -> calibrated probs
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(raw_probs, y)

        cal_probs = iso.predict(raw_probs)
        print(f"\n  After recalibration:")
        print(f"  Cal mean pred: {cal_probs.mean():.4%}  actual: {y.mean():.4%}  (ratio: {cal_probs.mean()/y.mean():.2f}x)")
        print(f"  Cal Brier:  {brier_score_loss(y, cal_probs):.6f}")
        print(f"  Cal AUC:    {roc_auc_score(y, cal_probs):.4f}")
        print(f"  Cal LogLoss:{log_loss(y, np.clip(cal_probs, 1e-8, 1-1e-8)):.4f}")

        # Decile check
        df_check = pd.DataFrame({"pred": cal_probs, "actual": y})
        df_check["decile"] = pd.qcut(df_check["pred"], 10, labels=False, duplicates="drop")
        dec = df_check.groupby("decile").agg(
            n=("actual", "count"),
            mean_pred=("pred", "mean"),
            actual_rate=("actual", "mean"),
        ).reset_index()
        print(f"\n  Decile calibration:")
        for _, r in dec.iterrows():
            ratio = r["actual_rate"] / r["mean_pred"] if r["mean_pred"] > 1e-8 else 0
            print(f"    D{int(r['decile'])}  n={int(r['n']):>5,}  pred={r['mean_pred']:.3%}  actual={r['actual_rate']:.3%}  ratio={ratio:.2f}x")

        out_path = MODELS_DIR / f"recal_{horizon}.joblib"
        joblib.dump(iso, out_path)
        print(f"\n  Saved -> {out_path}")

    print("\nRecalibration complete.")


if __name__ == "__main__":
    main()
