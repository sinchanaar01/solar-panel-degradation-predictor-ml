"""
Rule-based maintenance recommendation engine.

Rules are derived directly from README §4.4 with thresholds grounded
in the EDA findings (dataset mean efficiency ~0.51–0.53).

This module is intentionally small and pure-Python so it is easy to
explain, test, and adjust during a viva/presentation.
"""


def get_maintenance_recommendation(predicted_efficiency, soiling_ratio, panel_age):
    """
    Return a maintenance recommendation string based on predicted efficiency,
    soiling ratio, and panel age.

    Rules (evaluated in priority order):

    1. Efficiency < 0.40
       → "Urgent maintenance required — efficiency critically low"
       Rationale: 0.40 is roughly the bottom of the efficiency distribution
       (mean ~0.51–0.53 from EDA); below this the panel output is
       severely degraded and requires immediate action.

    2. Efficiency 0.40–0.55 AND soiling_ratio < 0.55
       → "Schedule panel cleaning — heavy soiling detected"
       Rationale: dataset efficiency rises with soiling_ratio, so it acts
       as a cleanliness index rather than a dirt measurement.

    3. panel_age > 25  AND  efficiency < 0.55
       → "Inspect for age-related degradation"
       Rationale: panel_age shows a weak but real downward trend in
       efficiency (EDA: -0.24 correlation). At >25 years with sub-0.55
       efficiency the combination suggests physical degradation worth
       inspecting rather than just cleaning.

    4. Otherwise
       → "No immediate maintenance needed — monitor periodically"

    Args:
        predicted_efficiency (float): model output, 0–1
        soiling_ratio (float): cleanliness index, 0.4 (dusty) to 1.0 (clean)
        panel_age (float): age in years

    Returns:
        tuple[str, str]: (recommendation text, severity level)
                         severity: 'critical' | 'warning' | 'info' | 'ok'
    """

    if predicted_efficiency < 0.40:
        # Rule 1: critically low — immediate action regardless of other factors
        return (
            "Urgent maintenance required — efficiency critically low",
            "critical",
            "urgent",
        )

    if 0.40 <= predicted_efficiency <= 0.55 and soiling_ratio < 0.55:
        # The dataset shows efficiency rising with this cleanliness index.
        return (
            "Schedule panel cleaning — heavy soiling detected",
            "warning",
            "cleaning",
        )

    if panel_age > 25 and predicted_efficiency < 0.55:
        # Rule 3: old panel + below-average efficiency → physical inspection
        return (
            "Inspect for age-related degradation",
            "warning",
            "age_inspect",
        )

    # Rule 4: all thresholds clear
    return (
        "No immediate maintenance needed — monitor periodically",
        "ok",
        "ok",
    )
