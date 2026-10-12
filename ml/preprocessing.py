"""Shared training and inference cleaning for solar panel input features."""

from pathlib import Path
from functools import lru_cache
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CATEGORICAL_LEVELS = {
    'string_id': ('A1', 'B2', 'C3', 'D4'),
    'error_code': ('E00', 'E01', 'E02', 'missing'),
    'installation_type': ('dual-axis', 'fixed', 'missing', 'tracking'),
}
CATEGORICAL_COLUMNS = tuple(CATEGORICAL_LEVELS)
NUMERIC_FEATURES = (
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure',
)
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLUMNS
UPLOAD_CATEGORY_DEFAULTS = {
    'string_id': 'A1',
    'error_code': 'missing',
    'installation_type': 'missing',
}
UPLOAD_CATEGORY_BASELINES = {
    'string_id': 'A1',
    'error_code': 'E00',
    'installation_type': 'dual-axis',
}
_EXTERNAL_STRING_ID_PATTERN = re.compile(
    r'STR-(?:00[1-9]|0[1-9][0-9]|100)', re.IGNORECASE
)
COLUMN_ALIASES = {
    'soiling': 'soiling_ratio',
    'cleanliness': 'soiling_ratio',
    'cleanliness_index': 'soiling_ratio',
    'module_temp': 'module_temperature',
    'panel_temperature': 'module_temperature',
    'cloud_cover': 'cloud_coverage',
    'wind': 'wind_speed',
    'age': 'panel_age',
    'maintenance': 'maintenance_count',
    'maintenance_counts': 'maintenance_count',
    'string': 'string_id',
    'installation': 'installation_type',
    'error': 'error_code',
}
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'


class MissingColumnsError(ValueError):
    """Required numeric columns were absent from an uploaded data frame."""

    def __init__(self, missing, found):
        self.missing = tuple(missing)
        self.found = tuple(found)
        super().__init__('missing_required_columns')


@lru_cache(maxsize=2)
def get_training_feature_hashes(dataset_path=None):
    """Hash numeric rows used to fit the model, excluding held-out rows."""
    path = Path(dataset_path) if dataset_path else (
        MODEL_DIR.parent / 'data' / 'train_cleaned.csv'
    )
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    training = pd.read_csv(path, usecols=list(NUMERIC_FEATURES))
    train_indices, _ = train_test_split(
        np.arange(len(training)), test_size=0.20, random_state=42
    )
    values = training.loc[train_indices, NUMERIC_FEATURES].to_numpy(
        dtype='float64'
    )
    return frozenset(
        hashlib.sha256(np.round(row, 4).tobytes()).digest()
        for row in values
    )


def count_training_feature_overlaps(features, training_hashes):
    """Return the number of feature rows matching the model's train split."""
    values = features.loc[:, NUMERIC_FEATURES].to_numpy(dtype='float64')
    return sum(
        hashlib.sha256(np.round(row, 4).tobytes()).digest() in training_hashes
        for row in values
    )


def is_supported_string_identifier(value):
    """Recognize model levels and the accepted external panel-ID format."""
    if pd.isna(value):
        return False
    normalized = str(value).strip()
    model_levels = {
        level.casefold() for level in CATEGORICAL_LEVELS['string_id']
    }
    return (
        normalized.casefold() in model_levels
        or _EXTERNAL_STRING_ID_PATTERN.fullmatch(normalized) is not None
    )


def normalize_columns(df):
    """Normalize common CSV header styles and known feature aliases."""
    normalized = []
    for column in df.columns:
        name = str(column).strip().lower()
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
        name = re.sub(r'[\s-]+', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        normalized.append(COLUMN_ALIASES.get(name, name))

    duplicate_names = sorted({
        name for name in normalized if normalized.count(name) > 1
    })
    if duplicate_names:
        raise ValueError('duplicate_columns')
    result = df.copy()
    result.columns = normalized
    return result


def _clean_numeric_and_clip(df):
    cleaned = df.copy()
    for col in NUMERIC_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    cleaned['irradiance'] = cleaned['irradiance'].clip(lower=0)
    cleaned['cloud_coverage'] = cleaned['cloud_coverage'].clip(upper=100)
    cleaned['temperature'] = cleaned['temperature'].clip(upper=90)
    cleaned['voltage'] = cleaned['voltage'].clip(upper=100)
    return cleaned


def encode_categoricals(df):
    """Encode all rows against the fixed training vocabulary and baseline."""
    encoded = df.drop(columns=list(CATEGORICAL_COLUMNS), errors='ignore').copy()
    for column, levels in CATEGORICAL_LEVELS.items():
        values = df[column].fillna('missing').astype(str).str.strip()
        level_lookup = {level.casefold(): level for level in levels}
        canonical = values.map(
            lambda value: level_lookup.get(value.casefold(), value)
        )
        unknown = sorted(set(canonical) - set(levels))
        if unknown:
            examples = ', '.join(value[:24] for value in unknown[:5])
            remaining = len(unknown) - 5
            if remaining > 0:
                examples += f', and {remaining} more'
            raise ValueError(
                f"Unknown {column} value(s): {examples}. "
                f"Allowed values: {', '.join(levels)}."
            )
        for level in levels[1:]:
            encoded[f'{column}_{level}'] = (
                canonical == level
            ).astype('int8')
    return encoded


def _check_required_columns(df, required=REQUIRED_FEATURES):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise MissingColumnsError(missing, df.columns)


def _clean_inference_frame(df, medians_path=None):
    normalized = normalize_columns(df)
    _check_required_columns(normalized, NUMERIC_FEATURES)
    cleaned = normalized.copy()
    for column, default in UPLOAD_CATEGORY_DEFAULTS.items():
        if column not in cleaned.columns:
            cleaned[column] = default
    cleaned = _clean_numeric_and_clip(cleaned.loc[:, REQUIRED_FEATURES])

    path = Path(medians_path) if medians_path else MODEL_DIR / 'training_medians.joblib'
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.is_file():
        raise FileNotFoundError(
            f'Training medians were not found at {path}; train the model first.'
        )
    from ml.artifacts import load_artifacts
    medians = load_artifacts(
        model_path=path.parent / 'best_model.joblib',
        model_dir=path.parent,
    )['training_medians']
    for column in NUMERIC_FEATURES:
        if column not in medians:
            raise ValueError(f'Training medians are missing numeric feature {column}.')
        cleaned[column] = cleaned[column].fillna(medians[column])
    return cleaned


def clean_and_encode_dataset(input_path, output_path=None):
    """Clean labeled training data, removing invalid zero-efficiency targets."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f'Dataset not found at {input_path}')

    df = pd.read_csv(input_path)
    _check_required_columns(df)
    df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
    df = df.loc[df['efficiency'].notna() & df['efficiency'].ne(0)].copy()
    df = _clean_numeric_and_clip(df)

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna('missing')

    for column in NUMERIC_FEATURES:
        df[column] = df[column].fillna(df[column].median())
    df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
    df = df.loc[df['efficiency'].notna()]

    df = df.drop(columns=['id'], errors='ignore')
    encoded = encode_categoricals(df)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(output_path, index=False)
        print(f'Cleaned dataset saved to {output_path}')
    return encoded


def clean_for_inference(df, medians_path=None):
    """Clean unlabeled raw inputs with medians learned from the training split."""
    normalized = normalize_columns(df)
    _check_required_columns(normalized)
    cleaned = _clean_inference_frame(normalized, medians_path=medians_path)
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna('missing')
    return encode_categoricals(cleaned)


def clean_for_upload(df, medians_path=None):
    """Prepare batch inputs with optional categories and bounded notices."""
    normalized = normalize_columns(df)
    _check_required_columns(normalized, NUMERIC_FEATURES)
    notices = []
    cleaned = normalized.copy()
    dummy_rebuilt = {}
    has_encoded_categories = any(
        f'{column}_{level}'.casefold() in cleaned.columns
        for column, levels in CATEGORICAL_LEVELS.items()
        for level in levels[1:]
    )
    for column, levels in CATEGORICAL_LEVELS.items():
        if column in cleaned.columns:
            continue
        dummy_columns = {
            level: f'{column}_{level}'.casefold()
            for level in levels[1:]
            if f'{column}_{level}'.casefold() in cleaned.columns
        }
        if not dummy_columns:
            if has_encoded_categories:
                cleaned[column] = UPLOAD_CATEGORY_BASELINES[column]
            continue
        baseline = UPLOAD_CATEGORY_BASELINES[column]
        values = pd.Series(baseline, index=cleaned.index, dtype='object')
        for level, dummy_column in dummy_columns.items():
            active = pd.to_numeric(
                cleaned[dummy_column], errors='coerce'
            ).fillna(0).eq(1)
            if (active & values.ne(baseline)).any():
                raise ValueError('invalid_dummy_categories')
            values.loc[active] = level
        cleaned[column] = values
        dummy_rebuilt[column] = tuple(dummy_columns.values())

    missing_categorical = [
        column for column in CATEGORICAL_COLUMNS
        if column not in cleaned.columns
    ]
    if missing_categorical:
        notices.append({
            'type': 'missing',
            'columns': missing_categorical,
        })

    for column, default in UPLOAD_CATEGORY_DEFAULTS.items():
        if column not in cleaned.columns:
            cleaned[column] = default
        else:
            values = cleaned[column]
            blank = values.isna() | values.astype(str).str.strip().eq('')
            cleaned.loc[blank, column] = default

    cleaned = _clean_inference_frame(cleaned, medians_path=medians_path)
    for column, levels in CATEGORICAL_LEVELS.items():
        values = cleaned[column].fillna(UPLOAD_CATEGORY_DEFAULTS[column])
        values = values.astype(str).str.strip()
        lookup = {level.casefold(): level for level in levels}
        canonical = values.map(
            lambda value: lookup.get(value.casefold())
        )
        unknown_mask = canonical.isna()
        if unknown_mask.any():
            unknown_values = values.loc[unknown_mask]
            fallback = UPLOAD_CATEGORY_DEFAULTS[column]
            unique_values = list(dict.fromkeys(unknown_values.tolist()))
            external_identifier_rows = (
                int(unknown_values.map(
                    is_supported_string_identifier
                ).sum())
                if column == 'string_id' else 0
            )
            examples = [
                re.sub(r'[^A-Za-z0-9._-]', '_', value)[:18]
                for value in unique_values[:5]
            ]
            notices.append({
                'type': 'unknown',
                'column': column,
                'count': int(unknown_mask.sum()),
                'examples': examples,
                'remaining': max(0, len(unique_values) - len(examples)),
                'default': fallback,
                'external_identifier_rows': external_identifier_rows,
            })
            canonical.loc[unknown_mask] = fallback
        cleaned[column] = canonical

    for column, dummy_columns in dummy_rebuilt.items():
        cleaned = cleaned.drop(columns=list(dummy_columns), errors='ignore')

    return encode_categoricals(cleaned), notices


if __name__ == '__main__':
    result = clean_and_encode_dataset('data/train.csv', 'data/train_cleaned.csv')
    print(f'Dataset shape: {result.shape}')
    print(f'Missing values: {result.isnull().sum().sum()}')
