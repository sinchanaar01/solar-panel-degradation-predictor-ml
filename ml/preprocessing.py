"""Shared training and inference cleaning for solar panel input features."""

from pathlib import Path

import pandas as pd


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
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'


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
        values = df[column].fillna('missing')
        unknown = sorted(set(values.astype(str)) - set(levels))
        if unknown:
            raise ValueError(
                f"Unknown {column} value(s): {', '.join(unknown)}. "
                f"Allowed values: {', '.join(levels)}."
            )
        for level in levels[1:]:
            encoded[f'{column}_{level}'] = (values == level).astype('int8')
    return encoded


def _check_required_columns(df):
    missing = [column for column in REQUIRED_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            'Missing required column(s): ' + ', '.join(missing) + '.'
        )


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
    _check_required_columns(df)
    cleaned = _clean_numeric_and_clip(df.loc[:, REQUIRED_FEATURES])

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

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna('missing')
    return encode_categoricals(cleaned)


if __name__ == '__main__':
    result = clean_and_encode_dataset('data/train.csv', 'data/train_cleaned.csv')
    print(f'Dataset shape: {result.shape}')
    print(f'Missing values: {result.isnull().sum().sum()}')
