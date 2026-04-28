#!/usr/bin/env python3
"""
Save the best-parameter XGBoost models for both 12-month and 5-year horizons.

Run this once (or whenever you want to refresh the production models):
    python src/save_best_models.py

It will:
- Load the monthly train/val splits
- Run a short Optuna search (30 trials) for each target to find best params
- Train final models on train+val
- Save:
    models/12m_model.joblib
    models/5y_model.joblib
    models/feature_cols.json
    models/best_params_12m.json
    models/best_params_5y.json
"""

import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DROP_COLS = [
    "gvkey", "permno", "month_end", "datadate", "conm", "cik",
    "annual_date", "ratio_date", "qtr_date",
    "dlstdt", "linkdt", "linkenddt", "linktype", "linkprim",
    "fyear",
    "bankrupt_event", "bankrupt_delist", "fjc_match_method",
    "fjc_default_chapter", "bankrupt_event_date", "delist_bankrupt_date",
    "default_date", "year_month",
    "fedfunds", "gs10", "baa", "aaa",
    # Raw rating string is excluded; we keep rating_numeric, speculative_grade, downgrade_1y, rating_age_years as real features
    "rating", "ratingdate",
    # Both targets must always be dropped — never use one as a feature for the other model
    "default_in_next_12m",
    "default_in_next_5y"
]


def get_feature_cols(df: pd.DataFrame, target: str) -> list:
    drop = DROP_COLS + [target]
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def run_optuna_and_train(target: str, n_trials: int = 30):
    print(f"\n=== Training best model for {target} ===")

    train = pd.read_parquet(DATA_PROC / "monthly_train.parquet")
    val = pd.read_parquet(DATA_PROC / "monthly_val.parquet")

    feature_cols = get_feature_cols(train, target)

    X_train = train[feature_cols].fillna(0).astype("float32")
    y_train = train[target].astype(int)
    X_val = val[feature_cols].fillna(0).astype("float32")
    y_val = val[target].astype(int)

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

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
            "scale_pos_weight": scale_pos,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    print(f"Best {target} validation AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # Final model on train + val
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_all, y_all)

    # Persist
    model_path = MODELS_DIR / f"{target.replace('_', '')}_model.joblib"
    joblib.dump(final_model, model_path)

    params_path = MODELS_DIR / f"best_params_{target.replace('_', '')}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Saved model  -> {model_path}")
    print(f"Saved params -> {params_path}")

    return final_model, feature_cols, best_params


def main():
    print("Building production-grade best-parameter models for Credit Risk UI...")

    # 12-month model
    model_12m, feature_cols, params_12m = run_optuna_and_train("default_in_next_12m", n_trials=30)

    # 5-year model
    model_5y, _, params_5y = run_optuna_and_train("default_in_next_5y", n_trials=30)

    # Save the common feature_cols (they are identical)
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n✅ Best-parameter models saved successfully.")
    print("   Models + feature list + best_params are in ./models/")
    print("\n   → To (re)build the searchable company list + tickers, run separately:")
    print("        python src/build_company_index.py")
    print("\n   This is now a lightweight script and does not retrain models.")


if __name__ == "__main__":
    main()
