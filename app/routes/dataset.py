"""Upload, validate, predict, and download user-owned CSV datasets."""

from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import (
    Blueprint, abort, current_app, flash, redirect, render_template, request,
    send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.models_db import AuditLog, Dataset, db
from app.utils.maintenance_rules import get_maintenance_recommendation
from app.utils.translations import t
from ml.preprocessing import clean_for_inference
from ml.predict import predict_batch

dataset_bp = Blueprint('dataset', __name__)
ALLOWED_EXTENSIONS = {'csv'}


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        raw_rows = pd.read_csv(upload_path)
        if raw_rows.empty:
            raise ValueError('empty_csv')
        cleaned = clean_for_inference(raw_rows)
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
    except (ValueError, FileNotFoundError, pd.errors.ParserError,
            UnicodeDecodeError) as exc:
        db.session.rollback()
        current_app.logger.info('CSV upload rejected: %s', exc)
        message = t('upload_failed')
        if str(exc) == 'empty_csv':
            message = t('empty_csv_error')
        elif str(exc).startswith('Missing required column(s): '):
            columns = str(exc).split(': ', 1)[1].rstrip('.')
            message = t('missing_required_columns', columns=columns)
        elif str(exc).startswith('Unknown ') and ' value(s): ' in str(exc):
            category, remaining = str(exc)[len('Unknown '):].split(
                ' value(s): ', 1
            )
            values, allowed = remaining.split('. Allowed values: ', 1)
            message = t(
                'unknown_category',
                column=category,
                values=values,
                allowed=allowed.rstrip('.'),
            )
        flash(message, 'error')
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


@dataset_bp.route('/upload/results/<int:dataset_id>')
@login_required
def upload_results(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()
    return render_template('upload_results.html', dataset=dataset)


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
