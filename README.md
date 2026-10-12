# Wearable Feature Engineering

A modular Python workflow for deriving interpretable participant-level features from longitudinal wearable data.

## Overview

Wearable devices generate repeated measurements of sleep, physiology, and physical activity. Transforming these observations into meaningful features requires careful consideration of temporal structure, missing data, and measurement characteristics.

This repository demonstrates reusable Python functions for deriving features from longitudinal wearable observations using entirely synthetic data.

## Features

| Domain | Derived features |
|---|---|
| Summary statistics | Mean, median, standard deviation, valid observation count |
| Temporal trends | Linear slopes across calendar days |
| Within-person variability | Coefficient of variation, successive-difference RMSSD, adjacent observation pairs |
| Sleep timing | Circular mean, circular standard deviation, valid observation count |

Successive-difference RMSSD describes fluctuations between adjacent observations and is distinct from the physiological HRV measure RMSSD.

## Repository Structure

```text
wearable-feature-engineering/
├── data/
│   ├── synthetic_wearable_data.csv
│   └── synthetic_participant_features.csv
├── notebooks/
│   └── feature_engineering_demo.ipynb
├── src/
│   └── wearable_features/
│       ├── __init__.py
│       ├── summary_features.py
│       ├── temporal_features.py
│       ├── variability_features.py
│       ├── sleep_timing.py
│       └── pipeline.py
├── tests/
│   └── test_features.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Open `notebooks/feature_engineering_demo.ipynb` in Jupyter and run all cells.

The notebook demonstrates how to load synthetic observations, generate participant-level features, examine selected outputs, and visualize longitudinal sleep measurements.

The full feature-engineering pipeline can also be called programmatically:

```python
import pandas as pd
from src.wearable_features.pipeline import build_feature_dataset

data = pd.read_csv(
    "data/synthetic_wearable_data.csv",
    parse_dates=["date"]
)

features = build_feature_dataset(data)
```

Run this example from the repository root.

## Testing

Run the automated tests from the repository root:

```bash
python -m pytest -v
```

The test suite checks summary statistics, temporal trends, within-person variability, missing observations, circular sleep timing, and pipeline output.

## Data and Limitations

All example observations are synthetic and do not represent actual research participants.

The pipeline demonstrates general-purpose feature engineering rather than a validated clinical or prognostic model. Feature suitability, completeness thresholds, device-specific measurement characteristics, and assumptions about missing data should be evaluated before application to research datasets.

## Research Context

This project demonstrates computational approaches relevant to intensive longitudinal wearable data, including sleep, autonomic physiology, and physical activity. It is intended to illustrate reproducible feature engineering without disclosing unpublished study-specific analytical methods.