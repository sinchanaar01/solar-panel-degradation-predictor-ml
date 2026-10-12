"""Manual prediction, result display, and per-user prediction history."""

import math

from flask import (
    Blueprint, current_app, flash, redirect, render_template, request, url_for,
)
from flask_login import current_user, login_required

from app.models_db import AuditLog, db, PredictionResult
from app.utils.maintenance_rules import get_maintenance_recommendation
from app.utils.translations import t
from ml.artifacts import load_artifacts
from ml.preprocessing import NUMERIC_FEATURES as MODEL_NUMERIC_FEATURES

predict_bp = Blueprint('predict', __name__)

NUMERIC_FEATURES = [
    ('temperature', 'temperature', 0, 90, 0.1, 25.0),
    ('irradiance', 'irradiance', 0, 1600, 1, 500.0),
    ('humidity', 'humidity', 0, 100, 0.1, 50.0),
    ('panel_age', 'panel_age', 0, 35, 0.5, 10.0),
    ('maintenance_count', 'maintenance_count', 0, 15, 1, 4.0),
    ('soiling_ratio', 'soiling_label',
     0.4, 1.0, 0.01, 0.8),
    ('voltage', 'voltage', 0, 100, 0.1, 12.0),
    ('current', 'current', 0, 8, 0.1, 1.5),
    ('module_temperature', 'module_temperature', 0, 65, 0.1, 30.0),
    ('cloud_coverage', 'cloud_coverage', 0, 100, 0.1, 25.0),
    ('wind_speed', 'wind_speed', 0, 15, 0.1, 7.0),
    ('pressure', 'pressure', 950, 1060, 0.1, 1013.0),
]

CATEGORICAL_FEATURES = [
    ('string_id', 'string_id', ['A1', 'B2', 'C3', 'D4']),
    ('error_code', 'error_code', ['E00', 'E01', 'E02', 'missing']),
    ('installation_type', 'installation_type',
     ['fixed', 'tracking', 'dual-axis', 'missing']),
]
SIMPLE_NUMERIC_FIELDS = ('panel_age', 'maintenance_count')
CLEANLINESS_CHOICES = {
    '0.9': 'clean_option',
    '0.7': 'light_dust_option',
    '0.5': 'heavy_dust_option',
}
CATEGORICAL_DISPLAY_LABELS = {
    'string_id': {
        'A1': 'string_a1',
        'B2': 'string_b2',
        'C3': 'string_c3',
        'D4': 'string_d4',
    },
    'error_code': {
        'E00': 'error_none',
        'E01': 'error_e01',
        'E02': 'error_e02',
        'missing': 'opt_missing',
    },
    'installation_type': {
        'fixed': 'opt_fixed',
        'tracking': 'opt_tracking',
        'dual-axis': 'opt_dual_axis',
        'missing': 'opt_missing',
    },
    'cleanliness': {
        '0.9': 'clean_option',
        '0.7': 'light_dust_option',
        '0.5': 'heavy_dust_option',
    },
}


def _translated_feature_name(name):
    for column, choices in CATEGORICAL_DISPLAY_LABELS.items():
        prefix = f'{column}_'
        if name.startswith(prefix):
            value = name[len(prefix):]
            if value in choices:
                return f'{t(column)} — {t(choices[value])}'
    return t(name)


def _render_form(form_data=None):
    return render_template(
        'predict.html',
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        option_labels=CATEGORICAL_DISPLAY_LABELS,
        form_data=form_data or {},
    )


def _validate_numeric(value, label, minimum, maximum, errors, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(t('enter_valid_number', label=t(label)))
        return None
    if not math.isfinite(number):
        errors.append(t('enter_finite_number', label=t(label)))
    elif number < minimum or number > maximum:
        errors.append(t(
            'value_out_of_range', label=t(label), value=number,
            minimum=minimum, maximum=maximum,
        ))
    elif integer and not number.is_integer():
        errors.append(t('whole_number_required', label=t(label)))
    if integer and math.isfinite(number) and number.is_integer():
        return int(number)
    return number


@predict_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return _render_form()

    mode = request.form.get('mode', 'simple')
    errors = []
    if mode not in {'simple', 'advanced'}:
        errors.append(t('invalid_mode'))
    artifacts = load_artifacts()
    medians = artifacts['training_medians']
    raw = dict(medians)

    if mode == 'simple':
        raw.update({'string_id': 'A1', 'error_code': 'E00'})
        features_by_name = {item[0]: item for item in NUMERIC_FEATURES}
        for name in SIMPLE_NUMERIC_FIELDS:
            _, label, minimum, maximum, _, _ = features_by_name[name]
            value = _validate_numeric(
                request.form.get(name), label, minimum, maximum, errors,
                integer=name == 'maintenance_count',
            )
            if value is not None:
                raw[name] = value

        cleanliness = request.form.get('cleanliness')
        if cleanliness not in CLEANLINESS_CHOICES:
            errors.append(t('invalid_choice', label=t('cleanliness_index')))
        else:
            raw['soiling_ratio'] = float(cleanliness)

        installation = request.form.get('installation_type', '')
        allowed_installations = dict(
            (name, choices) for name, _, choices in CATEGORICAL_FEATURES
        )['installation_type']
        if installation not in allowed_installations:
            errors.append(t('invalid_choice', label=t('installation_type')))
        raw['installation_type'] = installation
    else:
        for name, label, minimum, maximum, _, _ in NUMERIC_FEATURES:
            value = _validate_numeric(
                request.form.get(name), label, minimum, maximum, errors,
                integer=name == 'maintenance_count',
            )
            if value is not None:
                raw[name] = value
        for name, label, choices in CATEGORICAL_FEATURES:
            value = request.form.get(name, '')
            if value not in choices:
                errors.append(t('invalid_choice', label=t(label)))
            raw[name] = value

    if errors:
        for error in errors:
            flash(error, 'error')
        return _render_form(request.form)

    raw['prediction_mode'] = mode
    try:
        from ml.predict import predict_single
        efficiency = predict_single(raw)
    except (ValueError, FileNotFoundError) as exc:
        current_app.logger.info('Prediction input rejected: %s', exc)
        flash(t('prediction_failed'), 'error')
        return _render_form(request.form)
    except Exception:
        current_app.logger.exception('Prediction failed')
        flash(t('prediction_failed'), 'error')
        return _render_form(request.form)

    recommendation, _severity, _recommendation_key = get_maintenance_recommendation(
        predicted_efficiency=efficiency,
        soiling_ratio=raw['soiling_ratio'],
        panel_age=raw['panel_age'],
    )
    result = PredictionResult(user_id=current_user.id)
    result.set_input_data(raw)
    result.predicted_efficiency = efficiency
    result.maintenance_recommendation = recommendation
    db.session.add(result)
    db.session.add(AuditLog(
        user_id=current_user.id, action='prediction', ip=request.remote_addr
    ))
    db.session.commit()
    flash(t('prediction_complete', efficiency=f'{efficiency:.4f}'), 'success')
    return redirect(url_for('predict.results', result_id=result.id))


@predict_bp.route('/results/<int:result_id>')
@login_required
def results(result_id):
    result = PredictionResult.query.filter_by(
        id=result_id, user_id=current_user.id
    ).first_or_404()
    input_data = result.get_input_data()

    _, severity, recommendation_key = get_maintenance_recommendation(
        predicted_efficiency=result.predicted_efficiency,
        soiling_ratio=input_data['soiling_ratio'],
        panel_age=input_data['panel_age'],
    )
    artifacts = load_artifacts()
    ranges = artifacts['training_ranges']
    outside_training_range = any(
        input_data[feature] < ranges[feature][0]
        or input_data[feature] > ranges[feature][1]
        for feature in MODEL_NUMERIC_FEATURES
    )

    from ml.degradation import forecast_degradation
    forecast = forecast_degradation(input_data)
    importances = getattr(artifacts['model'], 'feature_importances_', None)
    if importances is None:
        raise RuntimeError('The saved model does not provide feature importances.')
    top_features = sorted(
        zip(artifacts['feature_columns'], importances),
        key=lambda item: item[1],
        reverse=True,
    )[:8]

    return render_template(
        'results.html',
        result=result,
        input_data=input_data,
        severity=severity,
        option_labels=CATEGORICAL_DISPLAY_LABELS,
        recommendation_text=t(recommendation_key),
        recommendation_key=recommendation_key,
        status_key={
            'urgent': 'status_urgent',
            'cleaning': 'status_watch',
            'age_inspect': 'status_watch',
            'ok': 'status_good',
        }[recommendation_key],
        outside_training_range=outside_training_range,
        prediction_mode=input_data.get('prediction_mode', 'advanced'),
        forecast=forecast,
        importance_labels=[_translated_feature_name(name) for name, _ in top_features],
        importance_values=[float(value) for _, value in top_features],
        reports_ready=True,
    )


@predict_bp.route('/history')
@login_required
def history():
    predictions = (
        PredictionResult.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionResult.created_at.desc())
        .all()
    )
    recommendation_keys = {
        item.id: get_maintenance_recommendation(
            item.predicted_efficiency,
            item.get_input_data()['soiling_ratio'],
            item.get_input_data()['panel_age'],
        )[2]
        for item in predictions
    }
    return render_template(
        'history.html',
        predictions=predictions,
        recommendation_keys=recommendation_keys,
    )
