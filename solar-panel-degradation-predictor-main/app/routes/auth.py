"""
Authentication routes for Flask-Login integration.

Routes:
  GET/POST /register — user registration form and handler
  GET/POST /login — user login form and handler
  GET /logout — logout and redirect
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from app.models_db import db, User


auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
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
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        
        # Validation
        errors = []
        if not username:
            errors.append('Username is required.')
        if not password:
            errors.append('Password is required.')
        if password != password_confirm:
            errors.append('Passwords do not match.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            errors.append(f'Username "{username}" is already taken.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        user = User(username=username)
        user.set_password(password)  # Hash password
        
        try:
            db.session.add(user)
            db.session.commit()
            flash(f'Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
            return redirect(url_for('auth.register'))
    
    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
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
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Validate inputs
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('auth.login'))
        
        # Look up user by username
        user = User.query.filter_by(username=username).first()
        
        # Verify password (check_password handles hash comparison)
        if user and user.check_password(password):
            # Success: create session
            login_user(user, remember=False)  # Set remember=True for "remember me" feature
            flash(f'Logged in as {username}.', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('auth.dashboard'))
        else:
            # Failed: invalid username or password
            # Do NOT reveal which one is wrong (security best practice)
            flash('Invalid username or password.', 'error')
            return redirect(url_for('auth.login'))
    
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """
    User logout route.
    
    Clears session and redirects to login.
    Requires user to be authenticated (@login_required).
    """
    username = current_user.username
    logout_user()
    flash(f'Logged out. Goodbye, {username}!', 'success')
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
    return render_template('dashboard.html')
