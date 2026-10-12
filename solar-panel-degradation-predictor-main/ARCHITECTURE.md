# Phase-II Architecture — Mapping to Phase-I Fig 6.2

This document maps every component built in Phase-II back to the system
architecture described in the Phase-I report (Fig 6.2). Any item from Fig 6.2
that was **not** implemented is called out explicitly under the Gaps section at
the bottom.

---

## Layer mapping

### 1. User Layer

> Phase-I description: registration, login, logout, session management.

| What was built | Files |
|---|---|
| User registration with server-side validation (empty username, password length, duplicate check) | `app/routes/auth.py` → `register()` |
| Password hashing: Werkzeug PBKDF2-SHA256 (salted, iterative — no plaintext ever stored) | `app/models_db.py` → `User.set_password()` / `User.check_password()` |
| Login: password verified against stored hash; session created by Flask-Login | `app/routes/auth.py` → `login()` |
| Session management: HTTP-only cookie, SameSite=Lax, 1-hour lifetime | `app/config.py` → `SESSION_COOKIE_*` |
| Logout: session cleared; redirect to login | `app/routes/auth.py` → `logout()` |
| Unauthenticated redirect: `@login_required` on all protected routes | every route in `app/routes/` |
| User database table: `id`, `username` (UNIQUE), `password_hash`, `created_at` | `app/models_db.py` → `User` |

---

### 2. Application Layer — Dataset Upload

> Phase-I description: dataset upload, tied to logged-in user.

| What was built | Files |
|---|---|
| File upload form; accepts only `.csv` (extension whitelist) | `app/routes/dataset.py` → `upload()`, `app/templates/upload.html` |
| Filename sanitised with `werkzeug.secure_filename` before writing to disk | `app/routes/dataset.py` |
| Files stored in per-user subdirectory (`uploads/<user_id>/`) | `app/routes/dataset.py` |
| Upload size capped at 16 MB (`MAX_CONTENT_LENGTH`) | `app/config.py` |
| Dataset metadata stored in DB (`datasets` table) | `app/models_db.py` → `Dataset` |

---

### 3. Application Layer — Preprocessing

> Phase-I description: preprocessing pipeline (shared between notebook and web app).

| What was built | Files |
|---|---|
| Reusable `clean_and_encode_dataset()` function — single source of truth for both training and upload | `ml/preprocessing.py` |
| Steps: coerce corrupted numerics, clip outliers, impute median, fill missing categoricals, OHE, drop `id` | `ml/preprocessing.py` (each step commented with reason) |
| Called by the upload route on every user-uploaded CSV | `app/routes/dataset.py` |
| Originally in `notebooks/eda_cleaning.ipynb`; refactored into the module | Phase-I notebook reference preserved; module extracted |

---

### 4. Application Layer — ML Model Training

> Phase-I description: train regression model(s), evaluate, select best.

| What was built | Files |
|---|---|
| Three models trained and compared: Linear Regression, Random Forest, Gradient Boosting | `ml/train_model.py` |
| 80/20 train/test split, `random_state=42` for reproducibility | `ml/train_model.py` |
| Evaluation metrics: RMSE, MAE, R² on held-out test set | `ml/evaluate_model.py` |
| Plots: predicted-vs-actual, residuals, feature importance for both tree models | `ml/evaluate_model.py` → `outputs/*.png` |
| Best model (Gradient Boosting, R²=0.790) saved to `models/best_model.joblib` | `ml/train_model.py` |
| Feature column list saved alongside model at `models/feature_columns.joblib` so inference never reads CSV | `ml/train_model.py` |
| Web app loads saved model; never retrains per request | `ml/predict.py` |

---

### 5. Application Layer — Prediction

> Phase-I description: run prediction on uploaded or manually entered data.

| What was built | Files |
|---|---|
| Manual single-row prediction form: 12 numeric + 3 categorical inputs, HTML5 + server-side validation | `app/routes/predict.py` → `predict()`, `app/templates/predict.html` |
| Batch prediction on uploaded CSV via `predict_batch()` | `ml/predict.py`, `app/routes/dataset.py` |
| OHE encoding at inference time matches training (same `drop_first=True` logic); column alignment via `reindex()` | `ml/predict.py` → `build_input_row()` |
| Prediction result displayed with efficiency score, maintenance recommendation, Chart.js bar chart | `app/templates/results.html` |
| Past predictions listed in history view | `app/routes/predict.py` → `history()`, `app/templates/history.html` |

---

### 6. Data Layer — Storage

> Phase-I description: user database, dataset storage, prediction results.

| What was built | Files |
|---|---|
| SQLite database via SQLAlchemy (auto-created on startup; no separate server) | `instance/app.db`, `app/models_db.py` |
| `users` table | `app/models_db.py` → `User` |
| `datasets` table: filename, file_path, uploaded_at, rows_count, FK→user | `app/models_db.py` → `Dataset` |
| `prediction_results` table: input_data (JSON), predicted_efficiency, maintenance_recommendation, created_at, FK→user, FK→dataset (nullable) | `app/models_db.py` → `PredictionResult` |
| Database URI overridable via `DATABASE_URL` env var (MySQL/Postgres drop-in) | `app/config.py` |

---

### 7. Performance Analysis & Visualisation

> Phase-I description: graphs/charts of predictions vs actuals.

| What was built | Files |
|---|---|
| Static training-time plots (PNG, saved to `outputs/`): predicted-vs-actual scatter (3 models), residuals, feature importance comparison | `ml/evaluate_model.py` |
| Live per-result bar chart in browser using Chart.js (CDN, no extra package): shows 6 key input features | `app/templates/results.html` |
| Feature importance confirms irradiance dominance (rank #1, importance 0.674) matching EDA finding (+0.74 correlation) | `ml/evaluate_model.py` output |

---

### 8. Report Generation

> Phase-I description: exportable PDF/Excel prediction reports.

| What was built | Files |
|---|---|
| PDF report (fpdf2): branded header, colour-coded efficiency score, maintenance recommendation box, two-column feature table, model note | `app/utils/report_generator.py` → `generate_pdf()` |
| Excel report (openpyxl): two sheets — Summary (metadata + colour-coded values) and Input Data (one column per feature, frozen header) | `app/utils/report_generator.py` → `generate_excel()` |
| Download endpoints (`/reports/<id>/pdf`, `/reports/<id>/excel`): `@login_required`, ownership check (404 on other users' results), streamed as attachment | `app/routes/reports.py` |

---

### 9. Maintenance Recommendation

> Phase-I description: rule-based suggestions from prediction results.

| What was built | Files |
|---|---|
| 4-rule priority logic (thresholds from EDA, documented in code comments): | `app/utils/maintenance_rules.py` |
| Rule 1: efficiency < 0.40 → "Urgent maintenance required — efficiency critically low" | |
| Rule 2: efficiency 0.40–0.55 AND soiling_ratio > 0.75 → "Schedule panel cleaning — high soiling detected" | |
| Rule 3: panel_age > 25 AND efficiency < 0.55 → "Inspect for age-related degradation" | |
| Rule 4 (default) → "No immediate maintenance needed — monitor periodically" | |
| Recommendation shown on results page, stored in DB, included in both PDF and Excel reports | |

---

### 10. Security Layer

> Phase-I description: authentication, session handling, input validation.

| What was built | Files |
|---|---|
| Password hashing: PBKDF2-SHA256 (Werkzeug default), salted automatically | `app/models_db.py` |
| Session cookies: HTTP-only, SameSite=Lax | `app/config.py` |
| Session timeout: 1 hour (`PERMANENT_SESSION_LIFETIME`) | `app/config.py` |
| Route protection: `@login_required` on all non-auth routes | all routes |
| Report ownership check: `filter_by(user_id=current_user.id)` prevents horizontal privilege escalation | `app/routes/reports.py`, `app/routes/predict.py` |
| Upload security: extension whitelist (`.csv` only), `secure_filename`, per-user storage directory | `app/routes/dataset.py` |
| Upload size limit: 16 MB (`MAX_CONTENT_LENGTH`) | `app/config.py` |
| Input validation: numeric range checks and categorical allowlist on prediction form (server-side, not just HTML5) | `app/routes/predict.py` → `predict()` |
| Wrong-password message deliberately generic ("Invalid username or password") — doesn't reveal which field is wrong | `app/routes/auth.py` |

---

## Gaps — items from Fig 6.2 NOT implemented

| Fig 6.2 item | Status | Notes |
|---|---|---|
| "Remember me" / persistent login | **Not implemented** | `login_user(remember=False)` — the checkbox and `remember=True` path were omitted as out of scope for an academic demo. Trivial to add. |
| Email-based registration / password reset | **Not implemented** | Phase-I Fig 6.2 shows only username/password auth with no email field. This was not specified. |
| Application Layer, Module 8 — System Maintenance Module | **Not implemented** | Fig 6.2 explicitly includes a maintenance/operations module covering system monitoring/logging, backup and data recovery, dataset/model update workflows, and reliability assurance. None of that exists as a first-class module in this app. For an academic Phase-II demo it is reasonable to omit heavy operational tooling, but a production version would need structured application logs, health checks, scheduled backups, restore procedures, model/dataset versioning, and documented update/runbook processes. |
| External Services — Email / Notifications | **Not implemented** | Fig 6.2 shows external notification capability, but the app does not send emails, OTPs, report mails, or user notifications of any kind. Reasonable to omit for a self-contained academic demo because it requires third-party credentials and delivery infrastructure. A production version would need SMTP or an email API provider, template management, retry handling, and secure secret storage for provider credentials. |
| External Services — Cloud Backup | **Not implemented** | Fig 6.2 shows cloud backup integration, but there is no sync of the SQLite database, uploaded CSVs, reports, or model artifacts to any remote storage provider. Acceptable to omit in a local academic prototype, but a production version would need automated off-site backups, retention policy, restore validation, and secure storage such as S3/Azure Blob/GCS. |
| External Services — System Alerts | **Not implemented** | Fig 6.2 includes system alerts, but this app has no alerting channel for failures such as repeated login failures, upload processing errors, disk/database issues, or model/report generation failures. Reasonable to omit for a single-machine academic deployment. A production version would need alert thresholds, delivery channels (email/SMS/Slack), and monitoring integration to surface operational failures in real time. |
| Production WSGI server config | **Not implemented as running code** | `gunicorn` command shown in README §7 but no `Procfile` or systemd unit was created. Appropriate for an academic submission; not a gap in the architecture. |
| Rate limiting / brute-force protection on login | **Not implemented** | Not specified in Phase-I §4.3 non-functional requirements. A comment in `auth.py` flags it as a production hardening step. |
| CSRF tokens on forms | **Partially implemented** | Flask's session-cookie SameSite=Lax provides meaningful CSRF protection for same-site form submissions. A full token-per-form scheme (e.g. Flask-WTF) was not added as it was not listed in Phase-I §4.1 tech stack. |
| Security Layer — Data Encryption at Rest | **Not implemented** | Apart from password hashing, there is no encryption at rest for the SQLite database, uploaded CSV files, generated reports, or model artifacts. This is a real gap relative to the Fig 6.2 security box. It is understandable in an academic local deployment where infrastructure is intentionally minimal, but a production system would need encrypted database storage, encrypted backup artifacts, OS/disk encryption, and key-management procedures. |
| Security Layer — Secure Communication (HTTPS/TLS) | **Not implemented** | The app runs on Flask's development server over plain HTTP (`http://127.0.0.1:5000`) and there is no TLS termination, certificate handling, or reverse proxy configuration. That is acceptable for local demo/testing only, but it is a real omission relative to the security architecture. A production version would need HTTPS via a reverse proxy/load balancer, certificate issuance/rotation, and secure-cookie settings enabled. |
| Security Layer — Audit & Logging | **Not implemented** | Beyond default request logging from the development server / Werkzeug, there is no audit trail for key user actions such as registration, login/logout, dataset upload, prediction generation, report download, or failed-access attempts. This is a real gap for accountability and incident analysis. A production version would need structured audit logs, timestamps, actor IDs, event types, retention policy, and tamper-resistant storage or centralised log shipping. |

---

## ML model selection summary

| Model | R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | 0.7531 | 0.0521 | 0.0394 | Baseline; multicollinearity between irradiance and temperature (r=0.90) inflates coefficients but still acceptable |
| Random Forest | 0.7791 | 0.0493 | 0.0372 | Handles multicollinearity naturally; clear feature importance |
| **Gradient Boosting** | **0.7900** | **0.0480** | **0.0362** | **Selected** — best on all three metrics; sequential error correction; irradiance importance 0.674 matches EDA |

**Gradient Boosting** is saved as `models/best_model.joblib` and loaded by the web app.

---

## Database substitution note

Phase-I §4.2 explicitly allows "MySQL or CSV-based / local storage." This implementation
uses **SQLite** via SQLAlchemy. SQLite requires no separate server process, ships with
Python, and the `DATABASE_URL` environment variable in `.env` can be changed to a MySQL
or PostgreSQL URL without touching any application code — SQLAlchemy abstracts the
difference entirely.
