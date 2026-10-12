"""
SQLAlchemy database models for the Solar Panel Degradation Predictor app.

Models:
  - User: registered users with hashed passwords
  - Dataset: uploaded CSV files tied to users
  - PredictionResult: predictions with maintenance recommendations tied to users
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    User model for authentication.
    
    Flask-Login requires:
      - is_authenticated: property (automatic via UserMixin)
      - is_active: property (automatic via UserMixin)
      - is_anonymous: property (automatic via UserMixin)
      - get_id(): method (automatic via UserMixin)
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    predictions = db.relationship('PredictionResult', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Hash and store password. Called during registration/password change."""
        # Werkzeug uses PBKDF2 with SHA256 by default (method='pbkdf2:sha256')
        # Hash includes salt automatically, strong against rainbow tables
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify plaintext password against stored hash. Called during login."""
        return check_password_hash(self.password_hash, password)


class Dataset(db.Model):
    """
    Metadata for uploaded CSV datasets.
    
    Each dataset belongs to a user and stores info about the upload
    (not the actual CSV data, which is stored on disk).
    """
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)  # Disk location of uploaded CSV
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    rows_count = db.Column(db.Integer)  # Number of rows in the CSV (for info display)
    
    # Relationships
    predictions = db.relationship('PredictionResult', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Dataset {self.filename} (user={self.user_id})>'


class PredictionResult(db.Model):
    """
    Stores a prediction result: inputs, output, and maintenance recommendation.
    
    Tied to a user (for multi-user isolation) and optionally a dataset (if predicted on uploaded data).
    """
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)  # NULL if manual single-row prediction
    
    # Input features as JSON (for reproducibility and record-keeping)
    # Example: {"irradiance": 850, "temperature": 35, "humidity": 0.65, ...}
    input_data = db.Column(db.Text, nullable=False)  # JSON string
    
    # Prediction output
    predicted_efficiency = db.Column(db.Float, nullable=False)
    
    # Maintenance recommendation (from maintenance_rules.py logic)
    # Example: "Schedule panel cleaning — high soiling detected"
    maintenance_recommendation = db.Column(db.Text, nullable=False)
    
    # Timestamp for audit trail
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<PredictionResult {self.id} (efficiency={self.predicted_efficiency:.4f})>'
    
    def get_input_data(self):
        """Parse input_data JSON string back to dict."""
        return json.loads(self.input_data)
    
    def set_input_data(self, data_dict):
        """Store input data as JSON string."""
        self.input_data = json.dumps(data_dict)
