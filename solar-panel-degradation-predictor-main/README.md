# Solar Panel Degradation Predictor Using Machine Learning

**Academic project — VTU, BE CSE (AI&ML), Rajeev Institute of Technology, Hassan**
Phase-I abstract: predicts solar panel efficiency degradation from environmental and
operational data using regression-based machine learning, with predictive maintenance
and energy forecasting as end goals.

This document is the single source of truth for an AI coding agent building Phase-II.
Read it fully before writing any code.

---

## 1. Scope of Phase-II (what to actually build)

Per the Phase-I system architecture (Fig 6.2), the full system has these layers:

- **User layer**: registration, login, logout, session management
- **Application layer**: dataset upload, preprocessing, ML model training, prediction
- **Data layer**: user database, dataset storage, model storage, prediction results
- **Performance analysis & visualization**: graphs/charts of predictions vs actuals
- **Report generation**: exportable PDF/Excel prediction reports
- **Maintenance recommendation**: rule-based suggestions from prediction results
- **Security layer**: authentication, session handling, input validation

**All of the above must be built for Phase-II** — this is a full working web application,
not just a notebook. The ML model is one module inside a larger app, not the whole
deliverable.

**Pragmatic note on the database:** Phase-I's requirements chapter explicitly allows
either "MySQL or CSV-based storage" / "MySQL or local storage." **Use SQLite** unless
you specifically want to install and run a MySQL server — SQLite requires no separate
server, ships with Python, and is fully acceptable per the Phase-I spec as "local
storage." Mention this substitution explicitly in the final report/demo as a practical
implementation choice.

---

## 2. Dataset

**Source file:** `data/train.csv` — 20,000 rows, 17 original columns, target = `efficiency`

| Column | Notes |
|---|---|
| id | row identifier, dropped before modeling |
| temperature | ambient temperature |
| irradiance | solar irradiance — strongest predictor of efficiency |
| humidity | had corrupted text entries ('error'/'unknown'/'badval') |
| panel_age | years — used for the degradation trend |
| maintenance_count | number of past maintenance events |
| soiling_ratio | dirt/soiling level on panel surface |
| voltage, current | electrical readings |
| module_temperature | panel surface temperature |
| cloud_coverage | percentage, should be 0–100 |
| wind_speed | had corrupted text entries |
| pressure | had corrupted text entries |
| string_id | categorical, 4 unique values (A1/B2/C3/D4) |
| error_code | categorical, E00/E01/E02/missing (~30% missing) |
| installation_type | categorical, fixed/tracking/dual-axis/missing (~25% missing) |
| efficiency | **target**, continuous, roughly 0.1–0.99 |

### Cleaning already performed (do not redo — see `notebooks/eda_cleaning.ipynb`)
1. Corrupted numeric columns (`humidity`, `wind_speed`, `pressure`) converted with
   `pd.to_numeric(errors='coerce')`.
2. Missing numeric values (~5% each, random pattern) → median imputation. Missing
   categoricals → filled with the literal string `"missing"`.
3. Physically impossible outliers clipped: `irradiance` ≥ 0, `temperature` ≤ 90,
   `cloud_coverage` ≤ 100, `voltage` ≤ 100.
4. 631 rows had `efficiency == 0` exactly — investigated, found statistically identical
   to normal rows on every feature and on `error_code` distribution → concluded to be a
   data-logging artifact, not real failures → treated as missing, median-imputed.
5. Categoricals one-hot encoded (`error_code`, `installation_type`, `string_id`,
   `drop_first=True`). `id` dropped.

**Output:** `data/train_cleaned.csv` — 20,000 rows × 22 columns, zero missing values,
ready for modeling.

### EDA findings — correlation with `efficiency`
| Feature | Correlation | Note |
|---|---|---|
| irradiance | +0.74 | dominant predictor |
| soiling_ratio | +0.37 | moderate |
| current | +0.35 | moderate |
| voltage | +0.21 | weak-moderate |
| panel_age | -0.24 | degradation signal |
| temperature/humidity/cloud_coverage/wind_speed/pressure/maintenance_count | ~0 to -0.08 | negligible |

`temperature` and `irradiance` correlate at 0.90 with each other (multicollinearity) —
relevant for Linear Regression, not for tree-based models.

Degradation curve (`efficiency` vs `panel_age`): real but noisy downward trend, ~0.57 at
age 0 → ~0.48 at age 35. High variance at every age — panel_age alone is weak, hence a
multivariate model is justified.

---

## 3. Machine Learning Requirements

- **Train/test split** on `data/train_cleaned.csv`.
- **Train and compare 3 models**: Linear Regression, Random Forest Regressor, Gradient
  Boosting Regressor.
- **Evaluate** with RMSE, MAE, R² on the held-out test set; pick the best model.
- **Feature importance** (Random Forest / Gradient Boosting) — should confirm irradiance
  dominance from the EDA.
- **Save the trained (best) model** to `models/` via `joblib`, to be loaded by the web
  app at prediction time — do not retrain on every request.

---

## 4. Web Application Requirements

### 4.1 Tech stack
- **Backend**: Python, Flask
- **Frontend**: HTML/CSS/JavaScript (server-rendered templates — Jinja2 is fine, no need
  for a separate JS framework)
- **Database**: SQLite via SQLAlchemy (see pragmatic note in section 1)
- **Auth**: Flask-Login for sessions, Werkzeug for password hashing
- **Reports**: `reportlab` or `fpdf2` for PDF export, `openpyxl`/pandas for Excel export
- **ML**: scikit-learn, joblib to load the saved model

### 4.2 Functional requirements (from Phase-I §4.2)
- User registration and login (unique users, hashed passwords, session-based auth)
- Dataset upload (CSV) tied to the logged-in user
- Trigger preprocessing on an uploaded dataset (reuse the same cleaning pipeline as
  `notebooks/eda_cleaning.ipynb`, refactored into a reusable function/module)
- Run prediction using the saved trained model on new/uploaded data or on a manually
  entered single row of input values
- Display results: predicted efficiency, degradation trend, chart of prediction vs
  history if available
- Generate and download a prediction report as **PDF and Excel**
- Generate a **maintenance recommendation** from the prediction (see 4.4 below)
- Securely store: user accounts, uploaded datasets metadata, prediction results
- Secure logout

### 4.3 Non-functional requirements (from Phase-I §4.3)
Performance (fast response on reasonable data sizes), accuracy (rely on the evaluated
best model), reliability (no crashes on malformed input — validate uploads), usability
(simple, clear UI/UX), scalability (should not choke on a larger dataset later),
maintainability (modular code, no giant single-file app), integrity (uploaded data and
predictions must not be corrupted or silently altered).

### 4.4 Maintenance recommendation logic
Since the Phase-I report doesn't specify exact rules, use this simple, explainable
rule-based logic (documented in code comments so it can be defended in a viva):

| Condition | Recommendation |
|---|---|
| predicted efficiency < 0.40 | "Urgent maintenance required — efficiency critically low" |
| predicted efficiency 0.40–0.55 **and** soiling_ratio > 0.75 | "Schedule panel cleaning — high soiling detected" |
| panel_age > 25 **and** predicted efficiency < 0.55 | "Inspect for age-related degradation" |
| none of the above | "No immediate maintenance needed — monitor periodically" |

Thresholds are derived from the EDA (dataset mean efficiency ≈ 0.51–0.53, so 0.40 is
roughly the bottom of the distribution). Keep this logic in its own small module
(`app/utils/maintenance_rules.py`) so it's easy to explain and adjust.

### 4.5 Pages/routes needed
- `/register`, `/login`, `/logout`
- `/dashboard` — landing page after login, links to upload/predict/reports
- `/upload` — upload a CSV dataset
- `/predict` — form for manual single-row prediction, or run prediction on an uploaded dataset
- `/results/<id>` — view a specific prediction result with chart + maintenance recommendation
- `/reports/<id>/pdf` and `/reports/<id>/excel` — download endpoints

---

## 5. Non-goals / constraints
- Do not re-run the data cleaning logic differently than what's documented above — reuse
  it, refactored into a shared module the Flask app can call, rather than duplicating code.
- Keep ML training/evaluation code separate from the web app (`ml/` folder), with the web
  app only *loading* the saved model, not retraining per request.
- Code must be organized into clear modules/files, commented with *why*, not just *what*
  — this needs to be explained in a viva/presentation.
- Don't introduce packages beyond what's listed in section 4.1 unless clearly justified.

---

## 6. Folder Structure
```
solar-panel-degradation-predictor/
├── data/
│   ├── train.csv
│   └── train_cleaned.csv
├── notebooks/
│   └── eda_cleaning.ipynb
├── ml/
│   ├── preprocessing.py        # shared cleaning pipeline (used by notebook AND app)
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   └── (saved model files, e.g. best_model.joblib)
├── app/
│   ├── app.py                  # Flask entry point
│   ├── config.py
│   ├── models_db.py            # SQLAlchemy models: User, Dataset, PredictionResult
│   ├── routes/
│   │   ├── auth.py
│   │   ├── dataset.py
│   │   ├── predict.py
│   │   └── reports.py
│   ├── templates/
│   │   ├── base.html, login.html, register.html, dashboard.html,
│   │   │   upload.html, predict.html, results.html
│   ├── static/
│   │   ├── css/  └── js/
│   └── utils/
│       ├── maintenance_rules.py
│       └── report_generator.py
├── instance/
│   └── app.db                  # SQLite database file (created at runtime)
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```

---

## 7. Running Locally — Complete Setup Guide

> **Starting point:** a fresh clone or unzip of this repository. No database, no
> model files, no Python packages installed. Follow every step in order.

### Step 0 — Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check: `python --version`. Download from python.org if needed. |
| Git (optional) | Only needed if cloning. Not needed if you unzipped the archive. |
| No MySQL / no extra server | SQLite is used by default and ships with Python. |

### Step 1 — Create a virtual environment

```bash
# Enter the project folder
cd Major_Project

# Create the environment (creates a .venv/ folder)
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate
```

You should now see `(.venv)` in your prompt.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, SQLAlchemy, scikit-learn, fpdf2, openpyxl, and all other
packages listed in `requirements.txt`.

### Step 3 — Configure environment variables

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```

Open `.env` in a text editor. The only variable you **must** change before any
public deployment is `SECRET_KEY`. For local testing the default is fine.

The three variables that are actually read by the app:

| Variable | Read by | Default if omitted |
|---|---|---|
| `SECRET_KEY` | `app/config.py` | `dev-key-change-in-production` |
| `FLASK_ENV` | `run.py` | `development` |
| `DATABASE_URL` | `app/config.py` | `sqlite:///app.db` (file auto-created at `instance/app.db`) |

### Step 4 — Train the ML models

```bash
python ml/train_model.py
```

Expected output:
```
[1/3] Loaded cleaned dataset: (20000, 22)
[2/3] Training 3 models...
      [1] Linear Regression (baseline, interpretable)
      [2] Random Forest (100 trees, max_depth=20)
      [3] Gradient Boosting (200 trees, learning_rate=0.05, max_depth=5)
[3/3] Models saved to models/
      OK linear_regression.joblib
      OK random_forest.joblib
      OK gradient_boosting.joblib
      OK feature_columns.joblib  (21 columns)
```

This creates four files in `models/`. The web app loads them at request time and
never retrains. If you want the evaluation plots and metrics table first, also run:

```bash
python ml/evaluate_model.py   # saves PNG charts to outputs/
```

### Step 5 — Start the web application

```bash
python run.py
```

Expected output:
```
Starting Flask development server (environment: development)
Visit http://localhost:5000/register to create an account
Database: sqlite:///app.db
 * Running on http://127.0.0.1:5000
```

The SQLite database (`instance/app.db`) and upload folder (`uploads/`) are created
automatically on first startup.

### Step 6 — Use the application

Open **http://localhost:5000/register** in your browser and follow this flow:

1. **Register** — create a username and password (minimum 6 characters).
2. **Login** — you land on the Dashboard.
3. **Predict** (nav bar) — fill in all 15 feature fields and submit. You get:
   - Predicted efficiency (0–1 scale)
   - Colour-coded maintenance recommendation
   - Bar chart of key input features
4. **Download PDF / Excel** — on the result page, download the full report.
5. **Upload** (nav bar) — upload a raw CSV (same columns as `data/train.csv`).
   The same cleaning pipeline runs automatically and batch predictions are returned.
6. **History** — lists every prediction your account has made.
7. **Logout** — session is cleared.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure the venv is activated and `pip install -r requirements.txt` was run inside it. |
| Old server behavior persists after restart / port stays in use | On Windows, Flask's debug reloader can leave orphaned processes behind after Ctrl+C. Run `Get-Process python \| Stop-Process -Force` then confirm with `Get-Process python` (should return nothing) before restarting with `python run.py`. |
| `models/best_model.joblib not found` | Run `python ml/train_model.py` first. |
| `Address already in use` on port 5000 | Change `port=5000` in `run.py` or kill the other process. |
| App starts but login redirects forever | Ensure `SECRET_KEY` in `.env` is set and not an empty string. |

---

## 8. Phase-II Architecture — Mapping to Phase-I Fig 6.2

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full component-to-layer mapping
and an explicit list of anything from Fig 6.2 that was not implemented.

