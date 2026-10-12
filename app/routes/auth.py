"""
Authentication routes for Flask-Login integration.

Routes:
  GET/POST /register — user registration form and handler
  GET/POST /login — user login form and handler
  POST /logout — logout and redirect
"""

from urllib.parse import urlsplit
import re

from flask import (
    Blueprint, current_app, render_template, request, redirect, url_for, flash,
    session,
)
from flask_login import login_user, logout_user, login_required, current_user
from app.extensions import limiter
from app.models_db import AuditLog, PredictionResult, db, User
from app.utils.translations import SUPPORTED_LANGS, t


auth_bp = Blueprint('auth', __name__)
COMMON_PASSWORDS = {
    'password', 'password123', '12345678', 'qwerty123', 'qwerty12',
    'abc12345', 'letmein123', 'admin123', 'welcome1',
}
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


def _audit(action, user_id=None):
    db.session.add(AuditLog(
        user_id=user_id,
        action=action,
        ip=request.remote_addr,
    ))
    db.session.commit()


def _safe_next(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/'):
        return None
    if parsed.path.startswith('//') or parsed.path.startswith('/\\'):
        return None
    return value


@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit('10 per hour', methods=['POST'])
def register():
    """
    User registration route.
    
    GET: Display registration form
    POST: Process form submission
    
    Security notes:
      - Passwords are hashed via Werkzeug.generate_password_hash (PBKDF2-SHA256, salted)
      - Usernames must be unique (checked in model constraint)
      - No plaintext passwords stored or logged
      - Form validation on both client and server side
    """
    
    # Redirect if already logged in
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        
        # Validation
        errors = []
        if not username:
            errors.append(t('username_required'))
        if not email:
            errors.append(t('email_required'))
        elif not EMAIL_PATTERN.fullmatch(email):
            errors.append(t('email_invalid'))
        if len(password) < 8:
            errors.append(t('password_min'))
        if not any(char.isupper() for char in password):
            errors.append(t('password_upper'))
        if not any(char.islower() for char in password):
            errors.append(t('password_lower'))
        if not any(char.isdigit() for char in password):
            errors.append(t('password_digit'))
        if username and password.casefold() == username.casefold():
            errors.append(t('password_username'))
        if password.casefold() in COMMON_PASSWORDS:
            errors.append(t('password_common'))
        if password != password_confirm:
            errors.append(t('password_mismatch'))
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            errors.append(t('username_taken'))
        if email and User.query.filter_by(email=email).first():
            errors.append(t('email_taken'))
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)  # Hash password
        
        try:
            db.session.add(user)
            db.session.flush()
            db.session.add(AuditLog(
                user_id=user.id, action='register', ip=request.remote_addr
            ))
            db.session.commit()
            flash(t('registration_success'), 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception('Registration failed')
            flash(t('registration_failed'), 'error')
            return redirect(url_for('auth.register'))
    
    return render_template('register.html')


@auth_bp.route('/set-language/<lang>', methods=['POST'])
def set_language(lang):
    if lang not in SUPPORTED_LANGS:
        flash(t('invalid_language'), 'error')
        return redirect(url_for('auth.login'))

    session['lang'] = lang
    if current_user.is_authenticated:
        current_user.language = lang
        db.session.commit()
        flash(t('saved_language', lang=lang), 'success')
    destination = _safe_next(request.form.get('next') or request.referrer)
    if destination:
        return redirect(destination)
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
    return redirect(url_for('auth.login'))


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit('5 per minute', methods=['POST'])
def login():
    """
    User login route with Flask-Login session management.
    
    GET: Display login form
    POST: Authenticate user and create session
    
    Session management:
      - Flask-Login creates secure session cookie after successful auth
      - Cookie is HTTP-only (not accessible to JavaScript)
      - Session timeout via PERMANENT_SESSION_LIFETIME (default 1 hour)
      - Session tied to User.id via session loader (see app.py)
    """
    
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
    
    if request.method == 'POST':
        identifier = request.form.get(
            'identifier', request.form.get('username', '')
        ).strip()
        password = request.form.get('password', '')
        
        # Validate inputs
        if not identifier or not password:
            flash(t('passwords_required'), 'error')
            _audit('login_failure')
            return redirect(url_for('auth.login'))
        
        # Email matching is case-insensitive because registration stores lowercase.
        user = User.query.filter_by(username=identifier).first()
        if user is None:
            user = User.query.filter_by(email=identifier.lower()).first()
        
        # Verify password (check_password handles hash comparison)
        if user and user.check_password(password):
            # Success: create session
            login_user(user, remember=False)  # Set remember=True for "remember me" feature
            session['lang'] = (
                user.language if user.language in SUPPORTED_LANGS else 'en'
            )
            _audit('login_success', user.id)
            flash(t('login_success', username=user.username), 'success')
            
            # Redirect to next page or dashboard
            next_page = _safe_next(request.values.get('next'))
            if next_page:
                return redirect(next_page)
            return redirect(url_for('auth.dashboard'))
        else:
            # Failed: invalid username or password
            # Do NOT reveal which one is wrong (security best practice)
            _audit('login_failure', user.id if user else None)
            flash(t('login_invalid'), 'error')
            return redirect(url_for('auth.login'))
    
    return render_template('login.html')


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """
    User logout route.
    
    Clears session and redirects to login.
    Requires user to be authenticated (@login_required).
    """
    username = current_user.username
    _audit('logout', current_user.id)
    logout_user()
    flash(t('logged_out_goodbye', username=username), 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/dashboard')
@login_required
def dashboard():
    """
    Dashboard page (landing page after login).
    
    Shows user welcome message and links to key features:
      - Upload dataset
      - Make prediction
      - View past predictions
      - Generate reports
    
    @login_required ensures only authenticated users can access.
    current_user is available in template automatically.
    """
    latest_prediction = (
        PredictionResult.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionResult.created_at.desc())
        .first()
    )
    return render_template(
        'dashboard.html',
        latest_prediction=latest_prediction,
    )
