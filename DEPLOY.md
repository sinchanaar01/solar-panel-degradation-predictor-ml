# Deployment

## Render

1. Push this project to a GitHub repository and create a Render **Web Service**
   from it. Select Python as the runtime.
2. Set the build command to:

   ```sh
   pip install -r requirements.txt
   ```

3. Set the start command to:

   ```sh
   gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT "app.app:create_app('production')"
   ```

   The included `Procfile` has this same command.
   One worker keeps Flask-Limiter's in-memory rate-limit counters consistent
   across requests; multiple workers would maintain separate counters.
4. Add these environment variables in Render:
   - `SECRET_KEY`: generate a unique value with
     `python -c "import secrets; print(secrets.token_hex(32))"`.
   - `FLASK_ENV`: `production`.
   - `COOKIE_SECURE`: `true`.
   - `TRUST_PROXY`: `true` only when the service is behind the platform's
     trusted reverse proxy. The service expects a single proxy hop.
   - `DATABASE_URL`: optional. If omitted, SQLite is created at
     `instance/app.db` under the project root.
5. Deploy. Check `/health` for a JSON `{"status":"ok"}` response. Register a
   fresh account and verify a prediction and report download.

**Important:** free-tier instances may have ephemeral storage. SQLite data,
uploaded CSV files, and generated prediction files in the local filesystem can
be wiped on redeploy or restart. For durable use, configure a persistent disk
or an external production database and object storage. The app does not migrate
an existing SQLite schema automatically.

## PythonAnywhere

1. Create a Python 3.11 web app and clone this repository into your account.
2. In a Bash console, install the dependencies into the virtual environment
   configured for the web app:

   ```sh
   cd ~/solar-panel-degradation-predictor
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set the web app's virtualenv path to
   `/home/<your-username>/solar-panel-degradation-predictor/.venv`.
4. In the PythonAnywhere **Web** tab, open the WSGI configuration file and use:

   ```python
   import os
   import sys

   project = "/home/<your-username>/solar-panel-degradation-predictor"
   if project not in sys.path:
       sys.path.insert(0, project)

   os.environ["FLASK_ENV"] = "production"
   os.environ["SECRET_KEY"] = "replace-with-a-generated-secret"
   os.environ["COOKIE_SECURE"] = "true"
   os.environ["TRUST_PROXY"] = "false"

   from app.app import create_app
   application = create_app("production")
   ```

   Replace the placeholder secret through the Web tab's environment variables
   facility when available; do not commit a real secret in this file.
5. Set the web app's working directory to the project root, then press
   **Reload** in the Web tab. PythonAnywhere serves the WSGI `application`;
   it does not use Render's Procfile start command.
6. Visit `/health`, register, and check prediction and report flows.

PythonAnywhere account storage persists according to the account plan, but local
SQLite and uploads are still tied to the account filesystem and are not a
substitute for database backups.
