"""
Prediction routes: manual single-row form and per-result detail view.

/predict  (GET)  — show the input form with all 15 feature fields
/predict  (POST) — validate inputs, run model, store result, redirect to results
/results/<id>    — show a stored PredictionResult with chart + recommendation
"""

import json
from pathlib import Path

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app
)
from flask_login import login_required, current_user

from app.models_db import db, PredictionResult
from app.utils.maintenance_rules import get_maintenance_recommendation

predict_bp = Blueprint('predict', __name__)


# Feature definitions used to render the form.
# Numeric inputs come with (min, max, step, example_value).
# Categorical inputs list their allowed choices.
NUMERIC_FEATURES = [
    ('temperature',       'Temperature (°C)',         -10,  90,   0.1,  35.0),
    ('irradiance',        'Irradiance (W/m²)',           0, 1200,  1,   600.0),
    ('humidity',          'Humidity (%)',                0,  100,  0.1,  60.0),
    ('panel_age',         'Panel Age (years)',           0,   50,  0.5,  10.0),
    ('maintenance_count', 'Maintenance Count',           0,   50,  1,     2.0),
    ('soiling_ratio',     'Soiling Ratio (0–1)',         0,    1,  0.01,  0.3),
    ('voltage',           'Voltage (V)',                 0,  100,  0.1,  45.0),
    ('current',           'Current (A)',                 0,   30,  0.1,   8.0),
    ('module_temperature','Module Temperature (°C)',     0,   90,  0.1,  40.0),
    ('cloud_coverage',    'Cloud Coverage (%)',          0,  100,  0.1,  20.0),
    ('wind_speed',        'Wind Speed (m/s)',            0,   50,  0.1,   3.5),
    ('pressure',          'Pressure (hPa)',            900, 1100,  0.1, 1013.0),
]

CATEGORICAL_FEATURES = [
    ('string_id', 'String ID', ['A1', 'B2', 'C3', 'D4']),
    ('error_code', 'Error Code', ['E00', 'E01', 'E02', 'missing']),
    ('installation_type', 'Installation Type', ['fixed', 'tracking', 'dual-axis', 'missing']),
]


@predict_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """
    GET: Render the prediction form pre-filled with example values.
    POST: Validate inputs, run model, store PredictionResult, redirect to /results/<id>.

    Design note: all numeric inputs are validated on the server side even
    though HTML5 min/max constraints are also set, because browser validation
    can be bypassed.
    """
    if request.method == 'POST':
        raw = {}
        errors = []

        # Collect and validate numeric inputs
        for name, label, min_val, max_val, step, _ in NUMERIC_FEATURES:
            val_str = request.form.get(name, '').strip()
            try:
                val = float(val_str)
                if val < min_val or val > max_val:
                    errors.append(f'{label}: value {val} is outside [{min_val}, {max_val}].')
                raw[name] = val
            except ValueError:
                errors.append(f'{label}: "{val_str}" is not a valid number.')
                raw[name] = 0.0

        # Collect categorical inputs (validated against allowed values)
        for name, label, choices in CATEGORICAL_FEATURES:
            val = request.form.get(name, '').strip()
            if val not in choices:
                errors.append(f'{label}: "{val}" is not a valid choice ({choices}).')
            raw[name] = val

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('predict.html',
                                   numeric_features=NUMERIC_FEATURES,
                                   categorical_features=CATEGORICAL_FEATURES,
                                   form_data=request.form)

        # Run prediction
        try:
            from ml.predict import predict_single

            # Build absolute paths so the module works regardless of cwd
            base = Path(current_app.root_path).parent
            model_path = str(base / 'models' / 'best_model.joblib')
            data_path = str(base / 'data' / 'train_cleaned.csv')

            efficiency = predict_single(raw, model_path=model_path, data_path=data_path)
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'error')
            return render_template('predict.html',
                                   numeric_features=NUMERIC_FEATURES,
                                   categorical_features=CATEGORICAL_FEATURES,
                                   form_data=request.form)

        # Get maintenance recommendation using raw inputs and predicted efficiency
        recommendation, severity = get_maintenance_recommendation(
            predicted_efficiency=efficiency,
            soiling_ratio=raw['soiling_ratio'],
            panel_age=raw['panel_age']
        )

        # Store result in DB
        result = PredictionResult(user_id=current_user.id)
        result.set_input_data(raw)
        result.predicted_efficiency = efficiency
        result.maintenance_recommendation = recommendation
        db.session.add(result)
        db.session.commit()

        flash(f'Prediction complete: efficiency = {efficiency:.4f}', 'success')
        return redirect(url_for('predict.results', result_id=result.id))

    return render_template('predict.html',
                           numeric_features=NUMERIC_FEATURES,
                           categorical_features=CATEGORICAL_FEATURES,
                           form_data={})


@predict_bp.route('/results/<int:result_id>')
@login_required
def results(result_id):
    """
    Show a stored PredictionResult.

    Only the owning user can view their own results (user_id check).
    Returns 404 if the result does not exist or belongs to another user.
    """
    result = PredictionResult.query.filter_by(
        id=result_id,
        user_id=current_user.id
    ).first_or_404()

    input_data = result.get_input_data()

    # Determine severity for template styling
    efficiency = result.predicted_efficiency
    if efficiency < 0.40:
        severity = 'critical'
    elif efficiency < 0.55:
        severity = 'warning'
    else:
        severity = 'ok'

    # Build chart data: key numeric features for display
    chart_labels = ['irradiance', 'soiling_ratio', 'panel_age',
                    'humidity', 'temperature', 'cloud_coverage']
    chart_values = [input_data.get(k, 0) for k in chart_labels]

    return render_template(
        'results.html',
        result=result,
        input_data=input_data,
        severity=severity,
        chart_labels=json.dumps(chart_labels),
        chart_values=json.dumps(chart_values),
        reports_ready=True   # Phase 4 reports blueprint is now registered
    )


@predict_bp.route('/history')
@login_required
def history():
    """List all past predictions for the logged-in user, newest first."""
    past = (PredictionResult.query
            .filter_by(user_id=current_user.id)
            .order_by(PredictionResult.created_at.desc())
            .all())
    return render_template('history.html', predictions=past)
