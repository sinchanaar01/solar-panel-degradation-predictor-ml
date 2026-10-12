"""Upload, validate, predict, and download user-owned CSV datasets."""

from pathlib import Path
from uuid import uuid4
import json
import re

import numpy as np
import pandas as pd
from html import escape
from flask import (
    Blueprint, abort, current_app, flash, redirect, render_template, request,
    send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.models_db import AuditLog, Dataset, db
from app.utils.maintenance_rules import get_maintenance_recommendation
from app.utils.translations import t
from ml.preprocessing import (
    CATEGORICAL_LEVELS,
    MissingColumnsError,
    NUMERIC_FEATURES,
    UPLOAD_CATEGORY_DEFAULTS,
    count_training_feature_overlaps,
    clean_for_upload,
    normalize_columns,
)
from ml.predict import predict_batch

dataset_bp = Blueprint('dataset', __name__)
ALLOWED_EXTENSIONS = {'csv'}
MAX_UPLOAD_ROWS = 50_000
PREDICTION_CHUNK_SIZE = 5_000
PREVIEW_ROW_COUNT = 10


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_upload_csv(path):
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            with path.open(encoding=encoding, newline='') as source:
                sample = source.read(4096)
            separator = None if any(
                delimiter in sample for delimiter in (',', ';', '\t')
            ) else ','
            return pd.read_csv(
                path, sep=separator, engine='python', encoding=encoding
            )
        except UnicodeDecodeError:
            if encoding == 'latin-1':
                raise
    raise UnicodeDecodeError('utf-8', b'', 0, 1, 'Could not decode CSV')


def _short_values(values, maximum=140):
    text = ', '.join(str(value)[:24] for value in values)
    if len(text) > maximum:
        text = text[:maximum - 3].rstrip(', ') + '...'
    return text


def _short_message(message):
    if len(escape(message)) <= 300:
        return message
    shortened = message
    while shortened and len(escape(shortened.rstrip() + '...')) > 300:
        shortened = shortened[:-1]
    return shortened.rstrip() + '...'


def _panel_identifiers(normalized_rows):
    for column in ('panel_id', 'panel', 'id'):
        if column in normalized_rows.columns:
            return normalized_rows[column]
    if 'string_id' in normalized_rows.columns:
        values = normalized_rows['string_id'].fillna('').astype(str).str.strip()
        valid = {
            value.casefold()
            for value in CATEGORICAL_LEVELS['string_id']
        }
        if (~values.str.casefold().isin(valid) & values.ne('')).any():
            return normalized_rows['string_id']
    return None


def _render_upload_notice(notice):
    if notice['type'] == 'missing':
        columns = ', '.join(
            f"{column} ({UPLOAD_CATEGORY_DEFAULTS[column]})"
            for column in notice['columns']
        )
        return _short_message(t(
            'upload_optional_defaults_notice', columns=columns
        ))
    if notice['type'] == 'unknown':
        examples = ', '.join(notice['examples'])
        if notice['remaining']:
            examples += f", {t('and_more_count', count=notice['remaining'])}"
        return _short_message(t(
            'upload_unknown_defaults_notice',
            column=notice['column'],
            count=notice['count'],
            examples=examples,
            default=notice['default'],
        ))
    if notice['type'] == 'overlap':
        return t(
            'upload_overlap_warning',
            percentage=f"{notice['percentage']:.1f}",
        )
    return t('upload_failed')


def _build_result_rows(raw_rows, normalized_rows, cleaned, predictions,
                       recommendations, severities):
    actual_source = next(
        (column for column in normalized_rows.columns if column == 'efficiency'),
        None,
    )
    actual = None
    if actual_source is not None:
        actual_column = raw_rows.columns[
            list(normalized_rows.columns).index(actual_source)
        ]
        actual = pd.to_numeric(raw_rows[actual_column], errors='coerce')
        actual = actual.reset_index(drop=True)

    result = pd.DataFrame(index=range(len(raw_rows)))
    panel_ids = _panel_identifiers(normalized_rows)
    if panel_ids is not None:
        result['panel_id'] = panel_ids.reset_index(drop=True)
    result['predicted_efficiency'] = np.round(predictions, 4)
    result['status'] = severities
    result['recommendation'] = recommendations
    if actual is not None:
        result['actual_efficiency'] = actual
        result['error'] = np.asarray(predictions) - actual.to_numpy()

    excluded = set()
    if actual_source is not None:
        excluded.add(raw_rows.columns[
            list(normalized_rows.columns).index(actual_source)
        ])
    result_panel = 'panel_id' in result.columns
    if result_panel:
        for index, column in enumerate(raw_rows.columns):
            normalized_name = normalized_rows.columns[index]
            if normalized_name in {'panel_id', 'panel', 'id'}:
                excluded.add(column)
    for column in raw_rows.columns:
        if column not in excluded:
            result[column] = raw_rows[column].reset_index(drop=True)
    return result, actual


def _actual_metrics(predictions, actual):
    if actual is None:
        return None, 0, None
    target = pd.to_numeric(actual, errors='coerce')
    valid = target.notna() & target.ne(0) & np.isfinite(target)
    valid &= np.isfinite(predictions)
    ignored = int((~valid).sum())
    if not valid.any():
        return None, ignored, None
    y_true = target.loc[valid].to_numpy(dtype='float64')
    y_pred = np.asarray(predictions)[valid.to_numpy()]
    metrics = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2 else None
        ),
    }
    indexes = np.flatnonzero(valid.to_numpy())
    if len(indexes) > 500:
        indexes = indexes[np.linspace(
            0, len(indexes) - 1, 500, dtype=int
        )]
    scatter = [
        {'actual': float(target.iloc[index]), 'predicted': float(predictions[index])}
        for index in indexes
    ]
    return metrics, ignored, scatter


@dataset_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        flash(t('file_selected_error'), 'error')
        return redirect(url_for('dataset.upload'))
    if not _allowed_file(uploaded.filename):
        flash(t('only_csv_error'), 'error')
        return redirect(url_for('dataset.upload'))

    safe_name = secure_filename(uploaded.filename)
    unique_name = f'{uuid4().hex}_{safe_name}'
    user_dir = Path(current_app.config['UPLOAD_FOLDER']) / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    upload_path = user_dir / unique_name
    uploaded.save(str(upload_path))

    try:
        raw_rows = _read_upload_csv(upload_path)
        if raw_rows.empty:
            raise ValueError('empty_csv')
        if len(raw_rows) > MAX_UPLOAD_ROWS:
            raise ValueError('too_many_rows')
        normalized_rows = normalize_columns(raw_rows)
        cleaned, notices = clean_for_upload(normalized_rows)
        project_root = Path(current_app.root_path).parent
        model_path = project_root / 'models' / 'best_model.joblib'
        data_path = project_root / 'data' / 'train_cleaned.csv'
        prediction_chunks = []
        for offset in range(0, len(cleaned), PREDICTION_CHUNK_SIZE):
            prediction_chunks.append(predict_batch(
                cleaned.iloc[offset:offset + PREDICTION_CHUNK_SIZE],
                model_path=str(model_path),
                data_path=str(data_path),
                already_encoded=True,
            ))
        predictions = np.concatenate(prediction_chunks)

        recommendation_data = [
            get_maintenance_recommendation(
                float(prediction),
                float(cleaned.iloc[index]['soiling_ratio']),
                float(cleaned.iloc[index]['panel_age']),
            )
            for index, prediction in enumerate(predictions)
        ]
        recommendations = [item[0] for item in recommendation_data]
        severities = [item[1] for item in recommendation_data]
        results, actual = _build_result_rows(
            raw_rows, normalized_rows, cleaned, predictions,
            recommendations, severities,
        )
        actual_metrics, actual_ignored, scatter_data = _actual_metrics(
            predictions, actual
        )
        overlap_count = count_training_feature_overlaps(
            cleaned, current_app.config['TRAINING_FEATURE_HASHES']
        )
        overlap_percentage = 100.0 * overlap_count / len(cleaned)
        if overlap_percentage > 50:
            notices.append({
                'type': 'overlap',
                'percentage': overlap_percentage,
            })

        mean_eff = float(predictions.mean())
        min_eff = float(predictions.min())
        max_eff = float(predictions.max())
        count_critical = int((predictions < 0.40).sum())

        dataset = Dataset(
            user_id=current_user.id,
            filename=safe_name,
            file_path=str(upload_path),
            rows_count=len(raw_rows),
            mean_eff=mean_eff,
            min_eff=min_eff,
            max_eff=max_eff,
            count_critical=count_critical,
            actual_mae=(actual_metrics['mae'] if actual_metrics else None),
            actual_rmse=(actual_metrics['rmse'] if actual_metrics else None),
            actual_r2=(actual_metrics['r2'] if actual_metrics else None),
            actual_ignored=actual_ignored,
            has_actual_efficiency=actual is not None,
            upload_notices=json.dumps(notices),
            overlap_percentage=overlap_percentage,
            scatter_data=json.dumps(scatter_data) if scatter_data else None,
        )
        db.session.add(dataset)
        db.session.flush()
        predictions_path = user_dir / f'predictions_{dataset.id}.csv'
        results.to_csv(predictions_path, index=False)
        dataset.predictions_path = str(predictions_path)
        db.session.add(AuditLog(
            user_id=current_user.id, action='upload', ip=request.remote_addr
        ))
        db.session.commit()
    except MissingColumnsError as exc:
        db.session.rollback()
        current_app.logger.info('CSV upload is missing numeric columns: %s', exc.missing)
        missing = _short_values(exc.missing, maximum=180)
        found_values = list(exc.found[:20])
        if len(exc.found) > 20:
            found_values.append('...')
        found = _short_values(found_values, maximum=80)
        flash(_short_message(t(
            'missing_required_columns',
            columns=missing,
            found=found or t('none_found'),
        )), 'error')
        return redirect(url_for('dataset.upload'))
    except (ValueError, FileNotFoundError, pd.errors.ParserError,
            pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        db.session.rollback()
        current_app.logger.info('CSV upload rejected: %s', exc)
        message = t('upload_failed')
        if str(exc) == 'empty_csv':
            message = t('empty_csv_error')
        elif str(exc) == 'too_many_rows':
            message = t('upload_too_many_rows', count=len(raw_rows))
        flash(_short_message(message), 'error')
        return redirect(url_for('dataset.upload'))
    except Exception:
        db.session.rollback()
        current_app.logger.exception('CSV upload processing failed')
        flash(t('upload_failed'), 'error')
        return redirect(url_for('dataset.upload'))

    flash(t(
        'upload_summary', count=len(raw_rows), mean=f'{mean_eff:.3f}',
        critical=count_critical,
    ), 'success')
    return redirect(url_for('dataset.upload_results', dataset_id=dataset.id))


@dataset_bp.get('/upload/template')
@login_required
def download_upload_template():
    template_path = (
        Path(current_app.root_path).parent / 'data' / 'sample_upload.csv'
    )
    if not template_path.is_file():
        abort(404)
    db.session.add(AuditLog(
        user_id=current_user.id, action='upload_template_download',
        ip=request.remote_addr,
    ))
    db.session.commit()
    return send_file(
        template_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_upload.csv',
    )


@dataset_bp.get('/upload/holdout-sample')
@login_required
def download_holdout_sample():
    sample_path = (
        Path(current_app.root_path).parent / 'data' / 'holdout_demo_200.csv'
    )
    if not sample_path.is_file():
        abort(404)
    db.session.add(AuditLog(
        user_id=current_user.id, action='holdout_sample_download',
        ip=request.remote_addr,
    ))
    db.session.commit()
    return send_file(
        sample_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='holdout_demo_200.csv',
    )


@dataset_bp.route('/upload/results/<int:dataset_id>')
@login_required
def upload_results(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()
    predictions_path = Path(dataset.predictions_path or '')
    if not predictions_path.is_file():
        abort(404)
    preview = pd.read_csv(
        predictions_path, nrows=PREVIEW_ROW_COUNT
    ).fillna('')
    preview_records = preview.to_dict(orient='records')
    for row in preview_records:
        for column, value in row.items():
            if column not in {'panel_id', 'status', 'recommendation'}:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    row[column] = f'{number:.3f}'
    stored_notices = json.loads(dataset.upload_notices or '[]')
    notices = [_render_upload_notice(notice) for notice in stored_notices]
    scatter = json.loads(dataset.scatter_data or '[]')
    return render_template(
        'upload_results.html',
        dataset=dataset,
        preview_columns=list(preview.columns),
        preview_rows=preview_records,
        upload_notices=notices,
        scatter_data=scatter,
        has_actual_metrics=dataset.has_actual_efficiency,
    )


@dataset_bp.route('/upload/<int:dataset_id>/download')
@login_required
def download_predictions(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()
    path = Path(dataset.predictions_path or '')
    if not path.is_file():
        abort(404)
    db.session.add(AuditLog(
        user_id=current_user.id, action='prediction_download',
        ip=request.remote_addr,
    ))
    db.session.commit()
    return send_file(
        path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'predictions_{dataset.id}.csv',
    )
