"""
Flask application factory and entry point.

Initializes the Flask app with:
  - SQLAlchemy database
  - Flask-Login session management
  - Authentication routes (auth blueprint)
  - Database tables
"""

import os
from pathlib import Path
from flask import Flask, jsonify, redirect, render_template, session, url_for
from flask_login import LoginManager
from werkzeug.middleware.proxy_fix import ProxyFix
from app.config import config_by_name
from app.models_db import db, User
from app.extensions import csrf, limiter
from app.routes.auth import auth_bp
from app.routes.dataset import dataset_bp
from app.routes.predict import predict_bp
from app.utils.translations import SUPPORTED_LANGS, t
from ml.preprocessing import get_training_feature_hashes


def create_app(config_name='production'):
    """
    Application factory function.
    
    Args:
        config_name (str): Configuration to load ('development', 'production', 'testing')
    
    Returns:
        Flask: Configured Flask application
    """
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = config_by_name.get(config_name, config_by_name['default'])
    app.config.from_object(config)
    config.init_app(app)
    if app.config['TRUST_PROXY']:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,
            x_proto=1,
            x_host=1,
            x_port=1,
            x_prefix=1,
        )
    app.jinja_env.globals['t'] = t

    @app.context_processor
    def inject_language_context():
        language = session.get('lang', 'en')
        if language not in SUPPORTED_LANGS:
            language = 'en'
        return {
            'current_lang': language,
            'SUPPORTED_LANGS': SUPPORTED_LANGS,
        }
    
    # Initialize extensions
    db.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'  # Redirect to /login if not authenticated
    login_manager.login_message = None
    
    @login_manager.user_loader
    def load_user(user_id):
        """
        User loader callback for Flask-Login.
        
        Called when Flask-Login needs to retrieve a user from the session.
        user_id is the User.id from the session cookie.
        """
        return db.session.get(User, int(user_id))
    
    # Ensure uploads folder exists at startup
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    uploads_dir = Path(app.config['UPLOAD_FOLDER']).resolve()
    app.config['UPLOAD_FOLDER'] = uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.config['TRAINING_FEATURE_HASHES'] = get_training_feature_hashes()

    # Register blueprints
    # Each blueprint groups related routes; prefix kept empty so URLs stay clean
    app.register_blueprint(auth_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(predict_bp)

    from app.routes.reports import reports_bp
    app.register_blueprint(reports_bp)

    @app.get('/health')
    def health():
        return jsonify(status='ok'), 200

    @app.errorhandler(429)
    def rate_limit_exceeded(_error):
        return render_template('error.html', status_code=429,
                               message=t('rate_limit_error')), 429

    @app.errorhandler(400)
    def bad_request(_error):
        return render_template('error.html', status_code=400,
                               message=t('bad_request_error')), 400

    @app.errorhandler(404)
    def not_found(_error):
        return render_template('error.html', status_code=404,
                               message=t('not_found_error')), 404

    @app.errorhandler(500)
    def internal_error(_error):
        db.session.rollback()
        return render_template('error.html', status_code=500,
                               message=t('server_error')), 500

    # Create database tables
    with app.app_context():
        db.create_all()
        from sqlalchemy import inspect, text
        dataset_columns = {
            column['name'] for column in inspect(db.engine).get_columns('datasets')
        }
        dataset_migrations = {
            'actual_mae': 'FLOAT',
            'actual_rmse': 'FLOAT',
            'actual_r2': 'FLOAT',
            'actual_ignored': 'INTEGER',
            'has_actual_efficiency': 'BOOLEAN NOT NULL DEFAULT 0',
            'upload_notices': 'TEXT',
            'overlap_percentage': 'FLOAT',
            'scatter_data': 'TEXT',
        }
        for column, sql_type in dataset_migrations.items():
            if column not in dataset_columns:
                db.session.execute(text(
                    f'ALTER TABLE datasets ADD COLUMN {column} {sql_type}'
                ))
        db.session.commit()
    
    # Root route (redirect to dashboard or login based on auth state)
    @app.route('/')
    def index():
        from flask_login import current_user
        if current_user.is_authenticated:
            return redirect(url_for('auth.dashboard'))
        return redirect(url_for('auth.login'))
    
    return app


if __name__ == '__main__':
    # Development server
    # For production, use a production WSGI server (gunicorn, uWSGI, etc.)
    app = create_app(os.environ.get('FLASK_ENV', 'production'))
    app.run(debug=app.config['DEBUG'], host='127.0.0.1', port=5000)
