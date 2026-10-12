
import numpy as np
import pandas as pd


def calculate_temporal_features(
    df,
    variables=None,
    participant_col="participant_id",
    date_col="date"
):
    """
    Calculate participant-level linear trends over time.

    Slopes represent change in the original measurement
    units per calendar day.

    Missing values are excluded independently for each
    variable. At least two valid observations on distinct
    dates are required.
    """

    if variables is None:
        variables = [
            "sleep_duration_min",
            "hrv_rmssd_ms",
            "steps",
            "activity_minutes"
        ]

    required = [participant_col, date_col] + list(variables)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    results = []

    for participant, group in data.groupby(participant_col):

        result = {participant_col: participant}

        days = (
            group[date_col] - group[date_col].min()
        ).dt.total_seconds() / 86400

        for variable in variables:

            valid = days.notna() & group[variable].notna()

            if valid.sum() >= 2 and days[valid].nunique() >= 2:

                slope = np.polyfit(
                    days[valid],
                    group.loc[valid, variable],
                    1
                )[0]

                result[f"{variable}_slope"] = slope

            else:
                result[f"{variable}_slope"] = np.nan

        results.append(result)

    return pd.DataFrame(results)
