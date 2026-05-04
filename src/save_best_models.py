#!/usr/bin/env python3
"""
Train a single unified discrete-time hazard model for credit default prediction.

A single XGBoost classifier learns P(default | features, horizon) by treating
`horizon_months` as a feature. Each firm-year observation is duplicated:
  - once with horizon_months=12  and label=default_in_next_12m
  - once with horizon_months=60  and label=default_in_next_5y

This guarantees monotonicity (P(5y) >= P(1y)) by construction and eliminates
the need for two separate models, two calibrators, and post-hoc enforcement.

Run:
    python src/save_best_models.py

Outputs:
    models/unified_model.joblib          -- raw XGBoost (for SHAP)
    models/unified_calibrated.joblib     -- CalibratedClassifierCV (for predictions)
    models/feature_cols.json             -- feature list (includes horizon_months)
    models/best_params.json              -- best Optuna hyperparameters
"""

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

DROP_COLS = [
    "gvkey", "permno", "month_end", "datadate", "conm", "cik",
    "annual_date", "ratio_date", "qtr_date",
    "dlstdt", "linkdt", "linkenddt", "linktype", "linkprim",
    "fyear",
    "bankrupt_event", "bankrupt_delist", "fjc_match_method",
    "fjc_default_chapter", "bankrupt_event_date", "delist_bankrupt_date",
    "default_date", "year_month",
    "fedfunds", "gs10", "baa", "aaa",
    "rating", "ratingdate",
    # Raw market features dropped -- replaced by Merton DD
    "mktcap_dec", "ret_12m", "ret_vol_12m", "excess_ret_12m",
    "equity_to_liabilities_mkt", "prcc_f", "bm",
    # All horizon targets (become labels via stacking, never features)
    "default_in_next_5y",
] + [f"default_in_next_{h}m" for h in HORIZONS]


def get_base_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric feature columns (excluding targets, IDs, etc.)."""
    return [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]


def stack_horizons(df: pd.DataFrame, feature_cols: list):
    """Stack each row N times (one per horizon), adding horizon_months as a feature."""
    X_base = df[feature_cols].fillna(0).astype("float32")
    Xs, ys = [], []
    for h in HORIZONS:
        X_h = X_base.copy()
        X_h["horizon_months"] = np.float32(h)
        target_col = f"default_in_next_{h}m"
        if target_col in df.columns:
            ys.append(df[target_col].astype(int))
        else:
            ys.append(pd.Series(0, index=df.index))
        Xs.append(X_h)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def main():
    print("=== Training unified discrete-time hazard model ===\n")

    train = pd.read_parquet(DATA_PROC / "monthly_train.parquet")
    val = pd.read_parquet(DATA_PROC / "monthly_val.parquet")

    base_features = get_base_feature_cols(train)
    print(f"Base features: {len(base_features)}")

    X_train, y_train = stack_horizons(train, base_features)
    X_val, y_val = stack_horizons(val, base_features)

    # feature_cols now includes horizon_months
    feature_cols = list(X_train.columns)

    # Monotone constraint: force P(default) to be non-decreasing in horizon_months.
    # +1 = prediction must increase as feature increases, 0 = unconstrained.
    mono_constraints = tuple(1 if c == "horizon_months" else 0 for c in feature_cols)

    print(f"Stacked train: {len(X_train):,} rows ({len(HORIZONS)}x {len(train):,})")
    print(f"Stacked val:   {len(X_val):,} rows")
    print(f"Combined positive rate: {y_train.mean():.4%}")
    print(f"Monotone constraint on horizon_months: enabled (+1)")

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # --- Optuna HPO ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 350, 850),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 6.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
            "monotone_constraints": mono_constraints,
            "scale_pos_weight": scale_pos,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print("\nRunning Optuna (40 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    best_params["monotone_constraints"] = mono_constraints
    best_params["scale_pos_weight"] = scale_pos
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    print(f"\nBest validation AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # --- Train final models ---
    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    # 1) Raw model (for SHAP)
    print("\nTraining raw model on train+val...")
    raw_model = xgb.XGBClassifier(**best_params)
    raw_model.fit(X_all, y_all)

    # 2) Calibrated model (5-fold isotonic)
    print("Fitting CalibratedClassifierCV (5-fold isotonic)...")
    base = xgb.XGBClassifier(**best_params)
    calibrated_model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    calibrated_model.fit(X_all, y_all)

    # Sanity check: verify monotonicity across all horizons on val
    n_base = len(val)
    print(f"\nMonotonicity + calibration check on val:")
    prev_preds = None
    for h in HORIZONS:
        h_mask = X_val["horizon_months"] == h
        preds = calibrated_model.predict_proba(X_val[h_mask])[:, 1]
        mono_str = ""
        if prev_preds is not None:
            mono_pct = (preds >= prev_preds - 1e-6).mean() * 100
            mono_str = f"  mono={mono_pct:.0f}%"
        print(f"  {h:2d}m: mean={preds.mean():.4%}, max={preds.max():.4%}{mono_str}")
        prev_preds = preds

    # --- Save ---
    joblib.dump(raw_model, MODELS_DIR / "unified_model.joblib")
    joblib.dump(calibrated_model, MODELS_DIR / "unified_calibrated.joblib")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save params (convert numpy types and tuples for JSON compatibility)
    save_params = {k: (list(v) if isinstance(v, tuple) else float(v) if hasattr(v, 'item') else v)
                   for k, v in best_params.items()}
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(save_params, f, indent=2)

    print(f"\nSaved:")
    print(f"  models/unified_model.joblib       (raw, for SHAP)")
    print(f"  models/unified_calibrated.joblib   (calibrated, for predictions)")
    print(f"  models/feature_cols.json           ({len(feature_cols)} features incl. horizon_months)")
    print(f"  models/best_params.json")
    print(f"\nNext: python src/generate_risk_lists.py")


if __name__ == "__main__":
    main()
