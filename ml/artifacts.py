"""Cached access to the artifacts produced together during model training."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'models'


@lru_cache(maxsize=8)
def _load_cached(model_path, data_path, model_dir):
    model_path = Path(model_path)
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    model = joblib.load(model_path)
    feature_path = model_dir / 'feature_columns.joblib'
    if feature_path.is_file():
        feature_columns = joblib.load(feature_path)
    elif data_path.is_file():
        feature_columns = list(
            pd.read_csv(data_path).drop(columns=['efficiency']).columns
        )
    else:
        raise FileNotFoundError(
            f'Feature columns were not found at {feature_path}; train the model first.'
        )

    medians_path = model_dir / 'training_medians.joblib'
    ranges_path = model_dir / 'training_ranges.joblib'
    if not medians_path.is_file() or not ranges_path.is_file():
        raise FileNotFoundError(
            f'Training preprocessing artifacts are missing from {model_dir}; '
            'train the model first.'
        )
    return {
        'model': model,
        'feature_columns': feature_columns,
        'training_medians': joblib.load(medians_path),
        'training_ranges': joblib.load(ranges_path),
    }


def load_artifacts(model_path=None, data_path=None, model_dir=None):
    """Load model and preprocessing artifacts once per configured path."""
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / 'best_model.joblib'
    data_path = Path(data_path) if data_path else PROJECT_ROOT / 'data' / 'train_cleaned.csv'
    model_dir = Path(model_dir) if model_dir else model_path.parent
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    return _load_cached(
        str(model_path.resolve()),
        str(data_path.resolve()),
        str(model_dir.resolve()),
    )
