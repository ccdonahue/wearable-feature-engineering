
import pandas as pd

DEFAULT_VARIABLES = [
    "sleep_duration_min",
    "sleep_efficiency_pct",
    "hrv_rmssd_ms",
    "steps",
    "activity_minutes"
]


def calculate_summary_features(
    df,
    variables=None,
    participant_col="participant_id"
):
    """
    Calculate participant-level summary features.

    Parameters
    ----------
    df : pandas.DataFrame
        Longitudinal wearable observations.
    variables : list, optional
        Variables to summarize.
    participant_col : str
        Participant identifier column.

    Returns
    -------
    pandas.DataFrame
        One row per participant with summary features.
    """

    if variables is None:
        variables = DEFAULT_VARIABLES

    required = [participant_col] + list(variables)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        df.groupby(participant_col)[variables]
        .agg(["mean", "median", "std", "count"])
    )

    summary.columns = [
        f"{variable}_{statistic}"
        for variable, statistic in summary.columns
    ]

    return summary.reset_index()
