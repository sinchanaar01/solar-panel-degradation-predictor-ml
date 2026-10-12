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
from flask import Flask, redirect, url_for
from flask_login import LoginManager
from app.config import config_by_name
from app.models_db import db, User
from app.routes.auth import auth_bp
from app.routes.dataset import dataset_bp
from app.routes.predict import predict_bp


def create_app(config_name='development'):
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
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'  # Redirect to /login if not authenticated
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        """
        User loader callback for Flask-Login.
        
        Called when Flask-Login needs to retrieve a user from the session.
        user_id is the User.id from the session cookie.
        """
        return User.query.get(int(user_id))
    
    # Ensure uploads folder exists at startup
    uploads_dir = Path(app.config['UPLOAD_FOLDER'])
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Register blueprints
    # Each blueprint groups related routes; prefix kept empty so URLs stay clean
    app.register_blueprint(auth_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(predict_bp)

    # Reports blueprint registered separately (Phase 4)
    try:
        from app.routes.reports import reports_bp
        app.register_blueprint(reports_bp)
    except ImportError:
        pass  # Phase 4 not yet built; skip gracefully

    # Create database tables
    with app.app_context():
        db.create_all()
    
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
    app = create_app('development')
    app.run(debug=True, host='127.0.0.1', port=5000)
