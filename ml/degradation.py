"""Age-sweep efficiency forecast for a fixed set of operating conditions."""

import numpy as np
import pandas as pd

from ml.predict import predict_batch, predict_single


def forecast_degradation(raw_inputs):
    """Predict efficiency from age 0–35; linear fit smooths the tree curve."""
    current_age = float(raw_inputs['panel_age'])
    age_values = list(range(36))
    sweep = pd.DataFrame([
        {**raw_inputs, 'panel_age': age}
        for age in age_values
    ])
    efficiencies = predict_batch(sweep)
    curve = [
        {'age': age, 'efficiency': float(value)}
        for age, value in zip(age_values, efficiencies)
    ]

    at_or_after = [
        (point['age'], point['efficiency'])
        for point in curve
        if point['age'] >= current_age
    ]
    if len(at_or_after) >= 2:
        slope = float(np.polyfit(
            [age for age, _ in at_or_after],
            [value for _, value in at_or_after],
            1,
        )[0])
    else:
        slope = 0.0

    current_efficiency = predict_single(raw_inputs)
    if current_efficiency < 0.40:
        years_to_threshold = 0.0
    else:
        crossing = next(
            (
                point['age'] - current_age
                for point in curve
                if point['age'] > current_age and point['efficiency'] < 0.40
            ),
            None,
        )
        years_to_threshold = float(crossing) if crossing is not None else None

    return {
        'curve': curve,
        'current_age': current_age,
        'current_efficiency': current_efficiency,
        'loss_per_year': slope,
        'years_to_threshold': years_to_threshold,
        'years_to_threshold_text': (
            'not within model range (≤35 yrs)'
            if years_to_threshold is None
            else f'{years_to_threshold:.1f}'
        ),
    }
