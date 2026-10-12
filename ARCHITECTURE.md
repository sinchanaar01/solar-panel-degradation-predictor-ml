# Application Architecture

## Overview

The application is a server-rendered Flask site backed by SQLite and a saved
scikit-learn regression model. The web and training pipelines share the
categorical vocabulary and preprocessing contract in `ml/preprocessing.py`.

## Request and data flow

| Concern | Implementation |
|---|---|
| App setup | `app/app.py` app factory wires configuration, SQLAlchemy, Flask-Login, CSRF, rate limiting, and blueprints |
| Health probe | `GET /health` returns static JSON without querying or writing to the database |
| Localization | `app/utils/translations.py` owns the five-language catalog and safe English fallback; Jinja exposes the active language and allowlist |
| Configuration | `app/config.py` reads environment settings, rejects missing/default production secrets, and creates a random development key when needed |
| Proxy support | `ProxyFix` trusts one forwarded proxy hop only when `TRUST_PROXY=true`; otherwise forwarded headers are ignored |
| Authentication | `app/routes/auth.py` validates passwords, authenticates users, validates local redirects, and logs security events |
| Persistent records | `app/models_db.py` defines `User`, `Dataset`, `PredictionResult`, and `AuditLog` |
| Feature cleaning | `ml/preprocessing.py` defines required inputs, fixed categorical levels, training cleaning, and inference cleaning |
| Artifact loading | `ml/artifacts.py` caches the model, feature schema, training medians, and percentile ranges by artifact path |
| Training | `ml/train_model.py` fits three regressors and saves feature columns, numeric medians, and 1st/99th percentile ranges |
| Evaluation | `ml/evaluate_model.py` compares held-out RMSE, MAE, and R², saves `outputs/model_metrics.csv`, and saves model plots |
| Manual prediction | `app/routes/predict.py` validates simple-mode and advanced-mode inputs, invokes `ml/predict.py`, saves results, and uses the recommendation engine's severity |
| Age forecast | `ml/degradation.py` sweeps panel age from 0 to 35 while holding remaining inputs fixed; a linear fit summarizes the non-smooth tree-model predictions |
| CSV prediction | `app/routes/dataset.py` normalizes headers, validates numeric inputs, applies optional categorical defaults, predicts in bounded chunks, reports actual-target metrics and training overlap, and provides an owned predictions-CSV download |
| Recommendation | `app/utils/maintenance_rules.py` evaluates prioritized efficiency, cleanliness, and age rules |
| Reports | `app/routes/reports.py` checks result ownership and streams user-language-selected PDF/Excel from `app/utils/report_generator.py`; bundled Noto fonts shape Indic PDF text |
| Browser charts | `app/templates/results.html` renders the age curve, current-age and 0.40 markers, and top eight global feature importances with Chart.js |
| Deployment | Render uses the two-worker Procfile Gunicorn command; PythonAnywhere uses its WSGI entry point; Docker uses Python 3.11 slim and a non-root Gunicorn user |

## Preprocessing and model artifacts

`data/train.csv` has 20,000 rows. Training discards the 631 rows with a
zero-efficiency target, leaving **19,369 rows** (15,495 training and 3,874 test
rows with the reproducible 80/20 split). Missing feature numerics are imputed,
categorical missing values use the `missing` vocabulary where supported, and
fixed-level manual dummy encoding is used for training, single predictions,
and CSV predictions. Numeric columns are required for upload; absent or blank
categorical columns use the documented defaults, and unknown category values
are safely mapped to those defaults with a notice. The single-panel form keeps
strict categorical validation.

Batch uploads are limited to 50,000 records and inferred in 5,000-row chunks.
If `efficiency` is included, non-zero numeric target values are compared to
predictions with MAE, RMSE, and R²; invalid and zero values are counted as
ignored. Upload result pages show up to ten output rows and a scatter plot
sampled to at most 500 actual/predicted pairs. Feature overlap is hashed
against only the reproducible 80% model-training split, so a held-out example
is not incorrectly reported as training data.

The feature `soiling_ratio` is treated as a cleanliness index because the data
shows efficiency increasing with the value; 1.0 is clean and lower values are
dustier. Rule 2 therefore recommends cleaning when the index is below 0.55 and
predicted efficiency is between 0.40 and 0.55.

Training writes these model artifacts:

| Artifact | Purpose |
|---|---|
| `models/best_model.joblib` | Selected Gradient Boosting model |
| `models/feature_columns.joblib` | Inference column names and order |
| `models/training_medians.joblib` | Feature medians learned from training rows |
| `models/training_ranges.joblib` | Numeric feature 1st/99th percentile bounds |

The app process caches these artifacts by path. Forecast metrics are calculated
when a result or report is viewed and are not stored in the database.

Measured results on the held-out test set:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.781448 | 0.049932 | 0.037952 |
| Random Forest | 0.808268 | 0.046768 | 0.035557 |
| Gradient Boosting | **0.817902** | **0.045578** | **0.034575** |

The baseline Gradient Boosting metrics were R² 0.790008, RMSE 0.048034, and
MAE 0.036167 on 4,000 test rows before zero-target rows were removed.

## Security controls and remaining gaps

| Control or capability | Status | Implementation |
|---|---|---|
| Password hashing and password-strength policy | Implemented | Werkzeug hashes; registration requires length, case, digit, non-common value, and username inequality |
| CSRF on POST forms | Implemented | Global Flask-WTF protection, including logout |
| Login/register request limits | Implemented | Flask-Limiter in-memory backend, 5/minute login and 10/hour registration per IP |
| Audit logging | Implemented | User ID (nullable for anonymous failures), action, timestamp, and IP |
| Session/cookie controls | Implemented | HTTP-only, SameSite=Lax; secure-cookie flag is environment-configurable and defaults on in production |
| Secret handling | Implemented | Random key for development; production rejects absent or known default keys |
| Open redirect protection | Implemented | `next` only accepts a local single-leading-slash path |
| Ownership checks | Implemented | Prediction, dataset-result, and report lookups are scoped to the signed-in user |
| Input validation and upload constraints | Implemented | Numeric ranges, categorical allowlist, CSV schema checks, extension whitelist, filename sanitization, and upload size cap |
| Persistent rate-limit storage | Not implemented | Limiter uses in-memory storage; restart resets counters and multiple workers do not share them |
| Production database and deployment | Partially implemented | Startup creates missing tables non-destructively; Render, PythonAnywhere, and Docker instructions are provided. SQLite and uploads still need persistent storage and backups for durability |
| Model calibration and external validation | Not implemented | Metrics reflect only this dataset's held-out split; no real-world calibration claim is made |

## Storage relationships

- `User` owns prediction records and uploaded dataset metadata.
- `User.language` stores the signed-in user's preference; logged-out visitors keep
  their choice in the session only.
- `Dataset` stores its source file path, row count, aggregate prediction values,
  and generated predictions CSV path; result pages and downloads are owner-scoped.
- `PredictionResult` stores the manual input JSON, predicted efficiency, and
  recommendation.
- `AuditLog.user_id` may be null so failed authentication attempts can be
  recorded even when no account is known.

## Localization and reports

The interface and exports support English, Kannada, Hindi, Telugu, and Tamil.
PDF output selects the appropriate bundled Noto family and enables HarfBuzz
shaping; the SIL Open Font License notices are alongside the font files.
Translated Indic wording remains draft text pending native-speaker review.
