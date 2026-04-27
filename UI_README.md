# Pulse Credit Risk UI

Beautiful Google/ChatGPT-style interface for the dual-horizon (12-month + 5-year) corporate default risk models.

## One-time setup (build the best models)

```bash
cd ~/Documents/Projects/capiq-credit-risk
pip install -r requirements.txt

# 1. Train the two best-parameter models (12m + 5y) — this is the slow step
python src/save_best_models.py

# 2. Build the company search index + pull real tickers (lightweight, fast)
python src/build_company_index.py
```

`save_best_models.py` trains the Optuna-tuned models and saves:
- `models/defaultinnext12m_model.joblib`
- `models/defaultinnext5y_model.joblib`
- `models/best_params_*.json`
- `models/feature_cols.json`

`build_company_index.py` is now a **separate lightweight script** that only builds the searchable company list + attaches real tickers from WRDS. You can run it anytime without retraining the models.

## Run the UI

```bash
python -m app.main
# or
uvicorn app.main:app --reload --port 8000
```

**Refreshing the company list or tickers later** (no model retraining needed):
```bash
python src/build_company_index.py
```

Open http://localhost:8000

- Type any company name or GVKEY in the huge centered search bar
- Real-time typeahead suggestions appear instantly
- Click any result → instantly see:
  - Two large probability cards (12m risk Dec 2025–Dec 2026 and 5y risk Dec 2025–Dec 2030)
  - Color-coded risk level (Low / Medium / High)
  - SHAP feature importance bars for both models (what actually drives the prediction)

## Architecture

- Backend: FastAPI + SHAP TreeExplainer (models loaded once at startup)
- Frontend: Pure Tailwind + vanilla JS + Chart.js (zero build step)
- All company search is in-memory from the pre-built index
- Every prediction returns full SHAP values for explainability

The two models use the exact best hyperparameters discovered by Optuna and the same rigorous feature set as the notebooks (Altman components, leverage, liquidity, macro spreads, CRSP market signals, etc.).
