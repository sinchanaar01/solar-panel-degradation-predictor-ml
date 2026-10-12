"""Upload, validate, predict, and download user-owned CSV datasets."""

from pathlib import Path
from uuid import uuid4

import pandas as pd
from html import escape
from flask import (
    Blueprint, abort, current_app, flash, redirect, render_template, request,
    send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.models_db import AuditLog, Dataset, db
from app.utils.maintenance_rules import get_maintenance_recommendation
from app.utils.translations import t
from ml.preprocessing import (
    CATEGORICAL_LEVELS,
    MissingColumnsError,
    UPLOAD_CATEGORY_DEFAULTS,
    clean_for_upload,
    normalize_columns,
)
from ml.predict import predict_batch

dataset_bp = Blueprint('dataset', __name__)
ALLOWED_EXTENSIONS = {'csv'}


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


def _show_upload_notices(notices):
    for notice in notices:
        if notice['type'] == 'missing':
            columns = ', '.join(
                f"{column} ({default})"
                for column, default in (
                    (name, UPLOAD_CATEGORY_DEFAULTS[name])
                    for name in notice['columns']
                )
            )
            flash(_short_message(t(
                'upload_optional_defaults_notice', columns=columns
            )), 'info')
            continue

        examples = ', '.join(notice['examples'])
        if notice['remaining']:
            examples += f", {t('and_more_count', count=notice['remaining'])}"
        flash(_short_message(t(
            'upload_unknown_defaults_notice',
            column=notice['column'],
            count=notice['count'],
            examples=examples,
            default=UPLOAD_CATEGORY_DEFAULTS[notice['column']],
        )), 'info')


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
        normalized_rows = normalize_columns(raw_rows)
        cleaned, notices = clean_for_upload(normalized_rows)
        panel_ids = _panel_identifiers(normalized_rows)
        project_root = Path(current_app.root_path).parent
        model_path = project_root / 'models' / 'best_model.joblib'
        data_path = project_root / 'data' / 'train_cleaned.csv'
        predictions = predict_batch(
            cleaned,
            model_path=str(model_path),
            data_path=str(data_path),
            already_encoded=True,
        )

        recommendations = [
            get_maintenance_recommendation(
                float(prediction),
                float(cleaned.iloc[index]['soiling_ratio']),
                float(cleaned.iloc[index]['panel_age']),
            )[0]
            for index, prediction in enumerate(predictions)
        ]
        results = raw_rows.copy()
        if panel_ids is not None:
            results['panel_id'] = panel_ids.to_numpy()
        results['predicted_efficiency'] = predictions
        results['recommendation'] = recommendations
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
        found = _short_values(found_values, maximum=115)
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
        flash(_short_message(message), 'error')
        return redirect(url_for('dataset.upload'))
    except Exception:
        db.session.rollback()
        current_app.logger.exception('CSV upload processing failed')
        flash(t('upload_failed'), 'error')
        return redirect(url_for('dataset.upload'))

    _show_upload_notices(notices)
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


@dataset_bp.route('/upload/results/<int:dataset_id>')
@login_required
def upload_results(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()
    predictions_path = Path(dataset.predictions_path or '')
    if not predictions_path.is_file():
        abort(404)
    preview = pd.read_csv(predictions_path, nrows=100).fillna('')
    return render_template(
        'upload_results.html',
        dataset=dataset,
        preview_columns=list(preview.columns),
        preview_rows=preview.to_dict(orient='records'),
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
