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

    2. Efficiency 0.40–0.55  AND  soiling_ratio > 0.75
       → "Schedule panel cleaning — high soiling detected"
       Rationale: soiling_ratio > 0.75 indicates heavy dirt accumulation
       that is measurably reducing output. Cleaning is a low-cost, high-
       impact intervention at this level.

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
        soiling_ratio (float): soiling level, 0–1
        panel_age (float): age in years

    Returns:
        tuple[str, str]: (recommendation text, severity level)
                         severity: 'critical' | 'warning' | 'info' | 'ok'
    """

    if predicted_efficiency < 0.40:
        # Rule 1: critically low — immediate action regardless of other factors
        return (
            "Urgent maintenance required — efficiency critically low",
            "critical"
        )

    if 0.40 <= predicted_efficiency <= 0.55 and soiling_ratio > 0.75:
        # Rule 2: moderate efficiency + high soiling → cleaning likely fixes it
        return (
            "Schedule panel cleaning — high soiling detected",
            "warning"
        )

    if panel_age > 25 and predicted_efficiency < 0.55:
        # Rule 3: old panel + below-average efficiency → physical inspection
        return (
            "Inspect for age-related degradation",
            "warning"
        )

    # Rule 4: all thresholds clear
    return (
        "No immediate maintenance needed — monitor periodically",
        "ok"
    )
