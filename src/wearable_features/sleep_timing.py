
import numpy as np
import pandas as pd


def circular_time_features(values):
    """
    Calculate circular mean and circular SD of clock times.

    Parameters
    ----------
    values : array-like
        Clock times expressed as decimal hours (0 to <24).

    Returns
    -------
    tuple
        (mean_hour, circular_sd_hours, valid_count)
    """
    values = pd.to_numeric(
        pd.Series(values), errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan, np.nan, 0

    if ((values < 0) | (values >= 24)).any():
        raise ValueError(
            "Clock times must be between 0 and 24 hours."
        )

    angles = values.to_numpy() * 2 * np.pi / 24

    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    R = np.hypot(mean_sin, mean_cos)

    # Undefined mean when observations cancel out
    if R < 1e-10:
        return np.nan, np.nan, len(values)

    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_hour = (mean_angle % (2 * np.pi)) * 24 / (2 * np.pi)

    # Circular standard deviation in hours
    R = np.clip(R, 1e-15, 1.0)
    circular_sd = np.sqrt(-2 * np.log(R)) * 24 / (2 * np.pi)

    return mean_hour, circular_sd, len(values)


def calculate_sleep_timing_features(
    df,
    participant_col="participant_id"
):
    """
    Calculate participant-level bedtime and wake-time
    circular means and circular standard deviations.
    """
    required = [
        participant_col,
        "bedtime_hour",
        "wake_hour"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for participant, group in df.groupby(participant_col):

        result = {participant_col: participant}

        for variable in ["bedtime_hour", "wake_hour"]:

            mean, sd, count = circular_time_features(
                group[variable]
            )

            result[f"{variable}_circular_mean"] = mean
            result[f"{variable}_circular_sd"] = sd
            result[f"{variable}_count"] = count

        results.append(result)

    return pd.DataFrame(results)
