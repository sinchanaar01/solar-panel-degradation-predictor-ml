# Solar Panel Degradation Predictor

[![CI](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml)

**Live demo:** _Add the deployed URL here._

An academic Flask application that estimates solar-panel efficiency from
environmental and operating conditions. It combines a trained regression model
with explainable maintenance rules, per-user prediction history, CSV batch
predictions, and downloadable PDF/Excel reports.

## Features

- Account registration and login with password hashing, CSRF protection,
  per-IP rate limits, secure sessions, and an audit trail.
- Single-panel predictions and CSV batch predictions with shared preprocessing
  and fixed categorical encodings.
- Default simple mode plus an advanced 15-feature mode for single-panel input.
- Training-time numeric imputation, percentile-range warnings, an age-based
  efficiency forecast, and the top eight saved-model feature importances.
- Maintenance recommendations based on predicted efficiency, cleanliness, and age.
- English, Kannada, Hindi, Telugu, and Tamil UI with a saved per-user language
  preference; PDF and Excel export language can be selected independently.
- User-owned result/history pages and PDF, Excel, and prediction-CSV downloads.
- Three compared regressors: Linear Regression, Random Forest, and Gradient
  Boosting.

## Tech stack

Python 3.11, Flask 2.3, Flask-SQLAlchemy, Flask-Login, Flask-WTF,
Flask-Limiter, SQLite, pandas, scikit-learn, joblib, Chart.js, fpdf2 with
HarfBuzz text shaping, and openpyxl.

## Model and dataset

The source dataset is `data/train.csv` (20,000 rows). Training converts damaged
numeric entries to missing values, clips the established physically invalid
limits, imputes feature values, and uses fixed categorical levels. The 631 rows
whose target efficiency was zero are excluded rather than assigned a fabricated
target, leaving **19,369 rows** in `data/train_cleaned.csv`. An 80/20 split with
`random_state=42` produced 15,495 training rows and 3,874 test rows.
The full-precision comparison is saved to `outputs/model_metrics.csv`.

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.781448 | 0.049932 | 0.037952 |
| Random Forest | 0.808268 | 0.046768 | 0.035557 |
| Gradient Boosting (selected) | **0.817902** | **0.045578** | **0.034575** |

The baseline model, trained before removing the invalid target rows, scored
R² 0.790008, RMSE 0.048034, and MAE 0.036167 on its 4,000-row test set. The
test-set sizes differ because the cleaned training set is smaller.

The feature `soiling_ratio` is treated as a **cleanliness index**: the dataset
shows efficiency increasing with the value, so 1.0 indicates clean and lower
values indicate more dust.

The browser and generated PDF reports use bundled Noto fonts for all supported
Indic scripts. Font families and their OFL notices are included in
`app/static/fonts/`. Kannada, Hindi, Telugu, and Tamil wording is draft text
pending native-speaker review.

## Setup

Use Python 3.11 for the pinned dependency versions.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Generate a secret and replace the `SECRET_KEY` placeholder in `.env`:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

For local schema changes that add the account email and language preference,
the existing `instance/app.db` is recreated; doing so deletes existing local
accounts and their saved results.

For local development, set `FLASK_ENV=development` in `.env`; the app creates
a random secret at startup if `SECRET_KEY` is omitted. Production requires an
explicit non-default secret and should be served over HTTPS. Set
`COOKIE_SECURE=true` when HTTPS is enabled. The default database is the local
SQLite file `instance/app.db`.

Train and evaluate the model (the checked-in model artifacts are already
available):

```powershell
python -m ml.preprocessing
python -m ml.train_model
python -m ml.evaluate_model
```

Training and evaluation use optional plotting packages. Install them separately
when regenerating model artifacts:

```powershell
python -m pip install -r requirements-train.txt
```

Start the application:

```powershell
python run.py
```

The development server prints its local URL. Register, then sign in to access
the dashboard.

## Usage

1. Create an account that meets the displayed password policy and log in.
2. Choose **Predict** for a manual input. Simple mode is the default; advanced
   mode exposes all 15 model features.
3. For CSV uploads, provide the 12 numeric features; `string_id`, `error_code`,
   and `installation_type` are optional. An `id` and `efficiency` target are
   not required. Headers with common unit suffixes or aliases are normalized,
   and comma-, semicolon-, and tab-delimited files are supported. Blank or
   unknown category values use documented defaults and produce a notice;
   arbitrary panel identifiers are retained as `panel_id` in the download.
4. Review the predicted efficiency, recommendation, age-sweep forecast, global
   feature importances, and any out-of-training-range warning.
5. Download a single-result PDF/Excel report or a batch predictions CSV from
   the user-owned result page.
6. Use the navigation language menu to set a UI preference. On a result page,
   choose any of the five report languages before downloading PDF or Excel.

Maintenance rules are evaluated in priority order: efficiency below 0.40 is
critical; efficiency 0.40–0.55 with cleanliness index below 0.55 triggers a
panel-cleaning recommendation; older low-efficiency panels are flagged for
inspection.

## Screenshots

Capture these six desktop-width screenshots yourself:

| Page | Screenshot |
|---|---|
| Login (English) | [login-en-desktop.png](docs/screenshots/login-en-desktop.png) |
| Predict (English) | [predict-en-desktop.png](docs/screenshots/predict-en-desktop.png) |
| Predict (Kannada) | [predict-kn-desktop.png](docs/screenshots/predict-kn-desktop.png) |
| Results (English) | [results-en-desktop.png](docs/screenshots/results-en-desktop.png) |
| Results (Tamil) | [results-ta-desktop.png](docs/screenshots/results-ta-desktop.png) |
| History (English) | [history-en-desktop.png](docs/screenshots/history-en-desktop.png) |

The [complete screenshot checklist](docs/screenshots/README.md) includes all
five languages and both desktop and 375px viewport filenames.

## Limitations

- The model learns patterns from the supplied dataset and should not be treated
  as an engineering guarantee for a real installation.
- The age forecast holds other inputs constant and extrapolates only within the
  model's 0–35 year input range; tree predictions are non-smooth.
- The forecast's linear fit summarizes a non-smooth tree-model curve and is not
  a guarantee that field performance changes linearly with age.
- CSV uploads are limited by the configured request-size limit and required
  feature schema.
- Unknown categorical values are mapped to the relevant default group, so
  predictions for those rows rely mainly on the numeric inputs.
- SQLite and in-memory rate-limit storage are appropriate for a local review
  deployment, not a horizontally scaled production service.
- The percentile warning signals distribution shift; it is not a calibrated
  prediction interval.
- Translated Kannada, Hindi, Telugu, and Tamil text is draft wording and should
  be reviewed by native speakers.

## Tests

Run the acceptance suite with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Docker

Create a strong key, export it as `SECRET_KEY`, then build and run:

```powershell
$env:SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
docker compose up --build
```

The app is available at `http://localhost:8000/`; health is checked at
`/health`. Docker Compose uses named volumes for the SQLite instance database
and uploads. Back these volumes up. Free-tier deployments without persistent
storage may lose SQLite records and uploads on redeploy.
The local Compose default disables secure cookies for plain HTTP; set
`COOKIE_SECURE=true` when serving through HTTPS.
