
import numpy as np
import pandas as pd


def calculate_variability_features(
    df,
    variables=None,
    participant_col="participant_id",
    date_col="date"
):
    """
    Calculate participant-level variability features.

    CV is calculated as sample SD / absolute mean * 100.
    Successive-difference RMSSD uses only adjacent
    calendar-day observations with nonmissing values.
    """

    if variables is None:
        variables = [
            "sleep_duration_min",
            "sleep_efficiency_pct",
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

    if data.duplicated([participant_col, date_col]).any():
        raise ValueError(
            "Duplicate participant-date observations detected."
        )

    results = []

    for participant, group in data.groupby(participant_col):

        group = group.sort_values(date_col)
        result = {participant_col: participant}

        for variable in variables:

            values = group[variable].astype(float)

            mean = values.mean()
            sd = values.std(ddof=1)

            if pd.notna(mean) and abs(mean) > 1e-10:
                cv = (sd / abs(mean)) * 100
            else:
                cv = np.nan

            # Only calculate differences between adjacent days
            day_gap = group[date_col].diff().dt.total_seconds() / 86400
            differences = values.diff()

            valid_pairs = (
                day_gap.eq(1)
                & values.notna()
                & values.shift(1).notna()
            )

            valid_differences = differences[valid_pairs]

            if len(valid_differences) > 0:
                rmssd = np.sqrt(
                    np.mean(valid_differences ** 2)
                )
            else:
                rmssd = np.nan

            result[f"{variable}_cv_pct"] = cv
            result[f"{variable}_successive_rmssd"] = rmssd
            result[f"{variable}_adjacent_pairs"] = len(
                valid_differences
            )

        results.append(result)

    return pd.DataFrame(results)
