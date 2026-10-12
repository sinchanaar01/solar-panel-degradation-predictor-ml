"""Export human-readable rows from the model's reproducible held-out split."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ml.preprocessing import (
    CATEGORICAL_LEVELS,
    NUMERIC_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _decode_categories(encoded):
    result = encoded.loc[:, NUMERIC_FEATURES].copy()
    for column, levels in CATEGORICAL_LEVELS.items():
        values = pd.Series(levels[0], index=encoded.index, dtype='object')
        active_count = pd.Series(0, index=encoded.index, dtype='int8')
        for level in levels[1:]:
            dummy = f'{column}_{level}'
            active = encoded[dummy].eq(1)
            active_count += active.astype('int8')
            values.loc[active] = level
        if active_count.gt(1).any():
            raise ValueError(f'Multiple active dummy levels for {column}.')
        result[column] = values
    result['efficiency'] = encoded['efficiency']
    return result


def make_holdout(data_path=None, models_dir=None, outputs_dir=None):
    root = PROJECT_ROOT
    data_file = Path(data_path) if data_path else root / 'data' / 'train_cleaned.csv'
    model_dir = Path(models_dir) if models_dir else root / 'models'
    output_dir = Path(outputs_dir) if outputs_dir else root / 'data'
    if not data_file.is_absolute():
        data_file = root / data_file
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    encoded = pd.read_csv(data_file)
    x = encoded.drop(columns=['efficiency'])
    y = encoded['efficiency']
    _, test_indices = train_test_split(
        x.index, test_size=0.20, random_state=42
    )
    x_test = x.loc[test_indices]
    y_test = y.loc[test_indices]

    model = joblib.load(model_dir / 'best_model.joblib')
    predictions = model.predict(x_test)
    measured_r2 = float(r2_score(y_test, predictions))
    saved_metrics = pd.read_csv(root / 'outputs' / 'model_metrics.csv')
    best_name = saved_metrics.loc[saved_metrics['r2'].idxmax(), 'model']
    saved_r2 = float(
        saved_metrics.loc[saved_metrics['model'].eq(best_name), 'r2'].iloc[0]
    )
    print(f'Saved evaluation R2 ({best_name}): {saved_r2:.8f}')
    print(f'Holdout model R2: {measured_r2:.8f}')
    print(f'Absolute R2 difference: {abs(saved_r2 - measured_r2):.8f}')
    if abs(saved_r2 - measured_r2) > 0.01:
        raise RuntimeError(
            'Held-out R2 differs from saved evaluation by more than 0.01.'
        )

    human_readable = _decode_categories(encoded.loc[test_indices])
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / 'holdout_test.csv'
    demo_path = output_dir / 'holdout_demo_200.csv'
    human_readable.to_csv(test_path, index=False)
    demo_count = min(200, len(human_readable))
    human_readable.sample(n=demo_count, random_state=42).to_csv(
        demo_path, index=False
    )
    print(f'Wrote {len(human_readable)} held-out rows to {test_path}')
    print(f'Wrote {demo_count} demo rows to {demo_path}')
    return {
        'saved_r2': saved_r2,
        'holdout_r2': measured_r2,
        'test_path': test_path,
        'demo_path': demo_path,
    }


if __name__ == '__main__':
    make_holdout()
