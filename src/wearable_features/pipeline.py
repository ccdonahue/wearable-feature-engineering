
import pandas as pd

from .summary_features import calculate_summary_features
from .temporal_features import calculate_temporal_features
from .variability_features import calculate_variability_features
from .sleep_timing import calculate_sleep_timing_features


def build_feature_dataset(df):
    """
    Generate participant-level features from
    longitudinal wearable observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Longitudinal wearable data containing participant
        identifiers, dates, sleep, HRV, and activity variables.

    Returns
    -------
    pandas.DataFrame
        One row per participant with derived features.
    """

    summary = calculate_summary_features(df)
    temporal = calculate_temporal_features(df)
    variability = calculate_variability_features(df)
    sleep_timing = calculate_sleep_timing_features(df)

    features = (
        summary
        .merge(
            temporal,
            on="participant_id",
            how="outer",
            validate="one_to_one"
        )
        .merge(
            variability,
            on="participant_id",
            how="outer",
            validate="one_to_one"
        )
        .merge(
            sleep_timing,
            on="participant_id",
            how="outer",
            validate="one_to_one"
        )
    )

    return features
