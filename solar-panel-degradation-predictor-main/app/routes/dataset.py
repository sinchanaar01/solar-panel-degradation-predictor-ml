"""
Dataset upload route.

Allows a logged-in user to upload a CSV file, runs the same cleaning
pipeline used during training (ml/preprocessing.py), runs batch
predictions on all cleaned rows, and stores a summary in the DB.
"""

import os
import json
from pathlib import Path
from werkzeug.utils import secure_filename
import pandas as pd

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, send_from_directory
)
from flask_login import login_required, current_user

from app.models_db import db, Dataset, PredictionResult
from app.utils.maintenance_rules import get_maintenance_recommendation

dataset_bp = Blueprint('dataset', __name__)

ALLOWED_EXTENSIONS = {'csv'}


def _allowed_file(filename):
    """Check that the uploaded file has a .csv extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@dataset_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """
    GET: Show the upload form.
    POST: Accept a CSV file, clean it, run predictions, store metadata.

    Security:
      - File extension validated before saving (whitelist .csv only)
      - Filename sanitised with werkzeug.secure_filename before use on disk
      - Files stored in a per-user subdirectory to avoid collisions
      - MAX_CONTENT_LENGTH in config limits upload size to 16 MB
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('dataset.upload'))

        file = request.files['file']
        if not file.filename:
            flash('No file selected.', 'error')
            return redirect(url_for('dataset.upload'))

        if not _allowed_file(file.filename):
            flash('Only CSV files are accepted.', 'error')
            return redirect(url_for('dataset.upload'))

        # Build user-specific upload directory
        upload_root = Path(current_app.config['UPLOAD_FOLDER'])
        user_dir = upload_root / str(current_user.id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Secure the filename to prevent path traversal attacks
        safe_name = secure_filename(file.filename)
        save_path = user_dir / safe_name
        file.save(str(save_path))

        # Run cleaning pipeline + batch predictions
        try:
            from ml.preprocessing import clean_and_encode_dataset
            from ml.predict import predict_batch

            df_cleaned = clean_and_encode_dataset(str(save_path))
            row_count = len(df_cleaned)

            # Drop target if accidentally present in upload
            if 'efficiency' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['efficiency'])

            # Batch predict
            preds = predict_batch(df_cleaned)
            mean_eff = float(preds.mean())
            min_eff = float(preds.min())
            max_eff = float(preds.max())
            count_critical = int((preds < 0.40).sum())

            # Store predictions CSV alongside upload
            pred_path = user_dir / f"predictions_{safe_name}"
            pd.DataFrame({'predicted_efficiency': preds}).to_csv(str(pred_path), index=False)

        except Exception as e:
            flash(f'Processing failed: {str(e)}', 'error')
            return redirect(url_for('dataset.upload'))

        # Store Dataset metadata in DB
        dataset = Dataset(
            user_id=current_user.id,
            filename=safe_name,
            file_path=str(save_path),
            rows_count=row_count,
        )
        db.session.add(dataset)
        db.session.commit()

        flash(
            f'Dataset uploaded and processed: {row_count} rows, '
            f'mean predicted efficiency {mean_eff:.3f}, '
            f'{count_critical} panels critically low.',
            'success'
        )
        return redirect(url_for('dataset.upload_results',
                                dataset_id=dataset.id,
                                mean_eff=round(mean_eff, 4),
                                min_eff=round(min_eff, 4),
                                max_eff=round(max_eff, 4),
                                count_critical=count_critical))

    return render_template('upload.html')


@dataset_bp.route('/upload/results')
@login_required
def upload_results():
    """Show a summary page after a dataset upload + batch prediction."""
    dataset_id = request.args.get('dataset_id', type=int)
    mean_eff = request.args.get('mean_eff', type=float)
    min_eff = request.args.get('min_eff', type=float)
    max_eff = request.args.get('max_eff', type=float)
    count_critical = request.args.get('count_critical', type=int, default=0)

    dataset = None
    if dataset_id:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()

    return render_template(
        'upload_results.html',
        dataset=dataset,
        mean_eff=mean_eff,
        min_eff=min_eff,
        max_eff=max_eff,
        count_critical=count_critical
    )
