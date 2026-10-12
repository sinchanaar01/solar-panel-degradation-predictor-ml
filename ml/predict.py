"""Prediction helpers using the canonical preprocessing feature schema."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.artifacts import PROJECT_ROOT, load_artifacts
from ml.preprocessing import (
    NUMERIC_FEATURES,
    clean_for_inference,
)


def _load_feature_columns(model_path='models/best_model.joblib',
                          data_path='data/train_cleaned.csv'):
    return load_artifacts(
        model_path=model_path, data_path=data_path
    )['feature_columns']


def build_input_row(raw_dict, feature_columns, model_dir=None):
    """Validate and encode one raw feature mapping in trained column order."""
    medians_path = (
        Path(model_dir) / 'training_medians.joblib' if model_dir else None
    )
    encoded = clean_for_inference(
        pd.DataFrame([raw_dict]), medians_path=medians_path
    )
    return encoded.reindex(columns=feature_columns, fill_value=0)


def predict_single(raw_dict, model_path='models/best_model.joblib',
                   data_path='data/train_cleaned.csv'):
    artifacts = load_artifacts(model_path=model_path, data_path=data_path)
    model_dir = Path(model_path).parent
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    row = build_input_row(
        raw_dict, artifacts['feature_columns'], model_dir=model_dir
    )
    return float(np.clip(artifacts['model'].predict(row)[0], 0.0, 1.0))


def predict_batch(df_raw, model_path='models/best_model.joblib',
                  data_path='data/train_cleaned.csv', already_encoded=False):
    """Encode and predict unlabeled raw rows using training-time medians."""
    artifacts = load_artifacts(model_path=model_path, data_path=data_path)
    if already_encoded:
        required_encoded = set(artifacts['feature_columns'])
        missing = sorted(required_encoded - set(df_raw.columns))
        if missing:
            raise ValueError(
                'Missing required encoded feature column(s): '
                + ', '.join(missing) + '.'
            )
        missing_numeric = [column for column in NUMERIC_FEATURES
                           if column not in df_raw.columns]
        if missing_numeric:
            raise ValueError(
                'Missing required column(s): ' + ', '.join(missing_numeric) + '.'
            )
        encoded = df_raw.copy()
    else:
        encoded = clean_for_inference(
            df_raw,
            medians_path=(
                (Path(model_path).parent if Path(model_path).is_absolute()
                 else PROJECT_ROOT / Path(model_path).parent)
                / 'training_medians.joblib'
            ),
        )
    aligned = encoded.reindex(columns=artifacts['feature_columns'], fill_value=0)
    return np.clip(artifacts['model'].predict(aligned), 0.0, 1.0)
