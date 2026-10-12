
import numpy as np
import pandas as pd
import pytest

from src.wearable_features.summary_features import calculate_summary_features
from src.wearable_features.temporal_features import calculate_temporal_features
from src.wearable_features.variability_features import calculate_variability_features
from src.wearable_features.sleep_timing import circular_time_features
from src.wearable_features.pipeline import build_feature_dataset


def test_summary_features():
    data = pd.DataFrame({
        "participant_id": ["P001"] * 4,
        "sleep_duration_min": [400, 410, np.nan, 430]
    })

    result = calculate_summary_features(
        data, variables=["sleep_duration_min"]
    )

    assert np.isclose(
        result["sleep_duration_min_mean"].iloc[0],
        413.3333333333333
    )
    assert result["sleep_duration_min_count"].iloc[0] == 3


def test_temporal_features():
    data = pd.DataFrame({
        "participant_id": ["P001"] * 3,
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-04"
        ]),
        "sleep_duration_min": [400, 410, 430]
    })

    result = calculate_temporal_features(
        data, variables=["sleep_duration_min"]
    )

    assert np.isclose(
        result["sleep_duration_min_slope"].iloc[0], 10
    )


def test_variability_features():
    data = pd.DataFrame({
        "participant_id": ["P001"] * 4,
        "date": pd.date_range("2026-01-01", periods=4),
        "sleep_duration_min": [400, 410, 420, 430]
    })

    result = calculate_variability_features(
        data, variables=["sleep_duration_min"]
    )

    assert np.isclose(
        result["sleep_duration_min_successive_rmssd"].iloc[0],
        10
    )
    assert result["sleep_duration_min_adjacent_pairs"].iloc[0] == 3


def test_missing_night_not_treated_as_adjacent():
    data = pd.DataFrame({
        "participant_id": ["P001"] * 3,
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-04"
        ]),
        "sleep_duration_min": [400, 410, 450]
    })

    result = calculate_variability_features(
        data, variables=["sleep_duration_min"]
    )

    assert result["sleep_duration_min_adjacent_pairs"].iloc[0] == 1
    assert np.isclose(
        result["sleep_duration_min_successive_rmssd"].iloc[0],
        10
    )


def test_circular_sleep_timing():
    mean, sd, count = circular_time_features(
        [23, 23.5, 0, 0.5]
    )

    assert np.isclose(mean, 23.75, atol=0.01)
    assert count == 4
    assert sd > 0


def test_circular_timing_missing_values():
    mean, sd, count = circular_time_features(
        [23, np.nan, 0]
    )

    assert count == 2
    assert np.isclose(mean, 23.5, atol=0.01)


def test_pipeline_output():
    data = pd.read_csv("data/synthetic_wearable_data.csv")

    result = build_feature_dataset(data)

    assert result.shape == (20, 46)
    assert result["participant_id"].is_unique
    assert result["participant_id"].nunique() == 20
