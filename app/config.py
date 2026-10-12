"""
Flask application configuration.

Separates configuration logic from app initialization.
Different configs for dev/production can be added here.
"""

import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = PROJECT_ROOT / 'instance'


class Config:
    """Base configuration."""
    
    # Database URI.
    # Reads DATABASE_URL from the environment if set; otherwise defaults to
    # SQLite stored at instance/app.db (no server required).
    # SQLite is used per the README pragmatic note: "Use SQLite unless you
    # specifically want to install and run a MySQL server — SQLite requires no
    # separate server, ships with Python, and is fully acceptable per Phase-I
    # spec as 'local storage'."
    # To switch to MySQL: DATABASE_URL=mysql+pymysql://user:pass@host/dbname
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{(INSTANCE_DIR / 'app.db').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flask-Login
    REMEMBER_COOKIE_HTTPONLY = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour session timeout
    SECRET_KEY = None
    ENVIRONMENT = 'development'
    
    # File upload settings
    UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    @staticmethod
    def init_app(app):
        secret_key = os.environ.get('SECRET_KEY', '').strip()
        defaults = {
            'dev-key-change-in-production',
            'prod-key-must-be-set',
            'change-me',
            'replace-with-a-generated-secret',
            'your-secret-key',
            'replace-me',
            'secret',
        }
        if app.config['ENVIRONMENT'] == 'production':
            if not secret_key or secret_key.casefold() in defaults:
                raise RuntimeError(
                    'Production requires a non-default SECRET_KEY environment variable.'
                )
        elif not secret_key:
            secret_key = secrets.token_hex(32)

        app.config['SECRET_KEY'] = secret_key
        secure_default = app.config['ENVIRONMENT'] == 'production'
        secure_cookie = os.environ.get('COOKIE_SECURE')
        if secure_cookie is None:
            secure_cookie = secure_default
        else:
            secure_cookie = secure_cookie.strip().lower() in {'1', 'true', 'yes', 'on'}
        app.config['SESSION_COOKIE_SECURE'] = secure_cookie
        app.config['REMEMBER_COOKIE_SECURE'] = secure_cookie
        app.config['TRUST_PROXY'] = (
            os.environ.get('TRUST_PROXY', '').strip().lower()
            in {'1', 'true', 'yes', 'on'}
        )


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    ENVIRONMENT = 'development'


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    ENVIRONMENT = 'testing'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # In-memory DB for tests


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    ENVIRONMENT = 'production'


# Config selection by environment
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
