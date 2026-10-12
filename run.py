#!/usr/bin/env python
"""
Application entry point for running the Flask development server.

Usage:
    python run.py

For production, use a production WSGI server:
    gunicorn -w 4 -b 0.0.0.0:5000 "app.app:create_app('production')"
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.app import create_app, db

# Determine config environment
config_name = os.environ.get('FLASK_ENV', 'production')
startup_url = 'http://127.0.0.1:5000/register'

# Create app
app = create_app(config_name)

if __name__ == '__main__':
    # For development only — use production WSGI server in production
    print(f"Starting Flask development server (environment: {config_name})")
    print(f"Visit {startup_url} to create an account")
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")

    app.run(host='127.0.0.1', port=5000, debug=app.config['DEBUG'])
