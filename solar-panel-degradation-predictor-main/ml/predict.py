"""
Prediction helper for single-row and batch predictions.

Used by the Flask web app to call the saved Gradient Boosting model
without retraining. Handles the categorical encoding step so callers
can pass raw feature values (e.g. string_id='A1') rather than OHE booleans.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Categorical columns and their dropped baseline (must match training pipeline)
# drop_first=True alphabetically drops the first level of each category.
# Encoding reference:
#   error_code: E00 dropped  → E01, E02, missing
#   installation_type: dual-axis dropped → fixed, missing, tracking
#   string_id: A1 dropped  → B2, C3, D4
CATEGORICAL_COLUMNS = ['error_code', 'installation_type', 'string_id']


def _load_feature_columns(model_path='models/best_model.joblib',
                          data_path='data/train_cleaned.csv'):
    """
    Load the exact feature column names (in order) that were used when
    the model was trained.

    Primary source: models/feature_columns.joblib, saved by train_model.py
    at training time. This is the authoritative list — it is always in sync
    with the model because it is written in the same training run.

    Fallback: derive column names from train_cleaned.csv. This path is
    only used if the .joblib file is absent (e.g. pre-existing install).
    A loud warning is printed so the discrepancy is never silent.

    Why not hardcode the list here? If the model is ever retrained on a
    different feature set (e.g. a new categorical added), a hardcoded list
    would silently pass wrong-shaped input to the model. Loading from the
    saved file guarantees the names match the trained artifact.
    """
    # Derive the feature_columns path from wherever the model lives
    model_dir = Path(model_path).parent
    saved_path = model_dir / 'feature_columns.joblib'

    if saved_path.exists():
        return joblib.load(saved_path)

    # Fallback — should not normally be reached in a properly trained setup
    import warnings
    warnings.warn(
        f"feature_columns.joblib not found at {saved_path}. "
        "Falling back to deriving columns from train_cleaned.csv. "
        "Re-run ml/train_model.py to generate the canonical column list.",
        RuntimeWarning,
        stacklevel=3,
    )
    df = pd.read_csv(data_path)
    return list(df.drop(columns=['efficiency']).columns)


def build_input_row(raw_dict, feature_columns):
    """
    Convert a dict of raw user inputs to a 1-row DataFrame aligned
    to the model's expected feature columns.

    Steps:
      1. Put raw values in a single-row DataFrame.
      2. One-hot encode the categorical columns with drop_first=True
         (same as the training pipeline in ml/preprocessing.py).
      3. Reindex to model's feature columns, filling missing OHE
         columns with 0 (= the dropped baseline category).

    Args:
        raw_dict (dict): e.g. {'temperature': 35, 'irradiance': 800,
                               ..., 'string_id': 'A1', 'error_code': 'E00',
                               'installation_type': 'fixed'}
        feature_columns (list[str]): ordered list from _load_feature_columns()

    Returns:
        pd.DataFrame: single-row, correctly ordered and encoded
    """
    df = pd.DataFrame([raw_dict])

    # Apply same OHE encoding as preprocessing pipeline
    cat_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present, drop_first=True)

    # Align to model's column schema (fill_value=0 = baseline category)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict_single(raw_dict,
                   model_path='models/best_model.joblib',
                   data_path='data/train_cleaned.csv'):
    """
    Run a single-row prediction from raw (pre-encoding) feature values.

    Args:
        raw_dict (dict): raw feature values (categorical as strings)
        model_path (str): path to saved joblib model
        data_path (str): path to cleaned CSV (for feature column reference)

    Returns:
        float: predicted efficiency (clipped to [0.0, 1.0])
    """
    model = joblib.load(model_path)
    feature_columns = _load_feature_columns(model_path=model_path, data_path=data_path)
    input_row = build_input_row(raw_dict, feature_columns)

    prediction = float(model.predict(input_row)[0])
    # Clip to valid efficiency range as a safety guard
    return float(np.clip(prediction, 0.0, 1.0))


def predict_batch(df_raw,
                  model_path='models/best_model.joblib',
                  data_path='data/train_cleaned.csv'):
    """
    Run predictions on a DataFrame of raw (pre-encoding) rows.
    Used after uploading a CSV dataset.

    Args:
        df_raw (pd.DataFrame): raw rows with same columns as train.csv
        model_path (str): path to saved joblib model
        data_path (str): path to cleaned CSV (feature column reference)

    Returns:
        np.ndarray: array of predicted efficiencies
    """
    model = joblib.load(model_path)
    feature_columns = _load_feature_columns(model_path=model_path, data_path=data_path)

    # OHE the categorical columns
    cat_present = [c for c in CATEGORICAL_COLUMNS if c in df_raw.columns]
    df_encoded = df_raw.copy()
    if cat_present:
        df_encoded = pd.get_dummies(df_encoded, columns=cat_present, drop_first=True)

    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(df_encoded)
    return np.clip(predictions, 0.0, 1.0)


if __name__ == '__main__':
    # Smoke test with a sample row (average values from EDA)
    sample = {
        'temperature': 35.0,
        'irradiance': 600.0,
        'humidity': 60.0,
        'panel_age': 10.0,
        'maintenance_count': 2.0,
        'soiling_ratio': 0.5,
        'voltage': 45.0,
        'current': 8.0,
        'module_temperature': 38.0,
        'cloud_coverage': 20.0,
        'wind_speed': 3.5,
        'pressure': 101.0,
        'string_id': 'A1',
        'error_code': 'E00',
        'installation_type': 'fixed',
    }
    result = predict_single(sample)
    print(f"Sample prediction: {result:.4f}")
