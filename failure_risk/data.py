from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "monthly_km",
    "engine_hours_per_month",
    "load_factor",
    "temperature_exposure",
    "prior_repairs",
    "service_delay_days",
    "route_severity",
    "preventive_maintenance_score",
    "vehicle_class",
]

NUMERIC_FEATURES = [
    "monthly_km",
    "engine_hours_per_month",
    "load_factor",
    "temperature_exposure",
    "prior_repairs",
    "service_delay_days",
    "route_severity",
    "preventive_maintenance_score",
]

CATEGORICAL_FEATURES = ["vehicle_class"]


def generate_synthetic_fleet(
    n_assets: int = 8_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate a reproducible right-censored commercial-vehicle dataset.

    The data follows a Weibull proportional-hazards process. This is intentional:
    a statistically well-specified Cox model should recover the direction of the
    hidden risk factors, which makes the project useful for model validation and
    interview discussion.
    """
    rng = np.random.default_rng(seed)

    vehicle_class = rng.choice(
        ["Light", "Medium", "Heavy"],
        size=n_assets,
        p=[0.25, 0.45, 0.30],
    )
    monthly_km = np.clip(rng.normal(4_500, 1_500, n_assets), 800, 9_000)
    engine_hours = np.clip(monthly_km / 45 + rng.normal(0, 15, n_assets), 20, 220)
    load_factor = np.clip(rng.beta(5, 2, n_assets), 0.15, 0.99)
    temperature_exposure = np.clip(rng.beta(2.2, 3, n_assets), 0.0, 0.99)
    prior_repairs = rng.poisson(0.8, n_assets)
    service_delay_days = np.clip(rng.gamma(2, 3, n_assets), 0, 35)
    route_severity = np.clip(rng.beta(3, 2.5, n_assets), 0.0, 0.99)
    preventive_maintenance_score = np.clip(rng.beta(5, 2, n_assets), 0.10, 1.0)
    entry_month = rng.integers(0, 24, n_assets)

    class_effect = pd.Series(vehicle_class).map(
        {"Light": -0.15, "Medium": 0.0, "Heavy": 0.25}
    ).to_numpy()

    # Hidden data-generating risk structure. Positive values increase hazard.
    linear_predictor = (
        0.32 * (monthly_km - 4_500) / 1_500
        + 0.20 * (engine_hours - 100) / 40
        + 0.65 * (load_factor - 0.70) / 0.15
        + 0.45 * (temperature_exposure - 0.40) / 0.20
        + 0.18 * prior_repairs
        + 0.025 * service_delay_days
        + 0.55 * (route_severity - 0.55) / 0.20
        - 0.55 * (preventive_maintenance_score - 0.70) / 0.15
        + class_effect
    )
    linear_predictor = np.clip(linear_predictor, -3.0, 3.0)

    weibull_shape = 2.0
    baseline_scale_months = 75.0
    u = rng.uniform(size=n_assets)
    failure_time = baseline_scale_months * (
        (-np.log(u) / np.exp(linear_predictor)) ** (1.0 / weibull_shape)
    )

    censor_time = rng.uniform(18.0, 72.0, n_assets)
    duration_months = np.minimum(failure_time, censor_time)
    event = (failure_time <= censor_time).astype(int)

    start_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(entry_month * 30, unit="D")

    df = pd.DataFrame(
        {
            "asset_id": [f"VH-{i:05d}" for i in range(1, n_assets + 1)],
            "cohort_start_date": start_date,
            "entry_month": entry_month,
            "vehicle_class": vehicle_class,
            "monthly_km": np.round(monthly_km, 1),
            "engine_hours_per_month": np.round(engine_hours, 1),
            "load_factor": np.round(load_factor, 4),
            "temperature_exposure": np.round(temperature_exposure, 4),
            "prior_repairs": prior_repairs,
            "service_delay_days": np.round(service_delay_days, 1),
            "route_severity": np.round(route_severity, 4),
            "preventive_maintenance_score": np.round(preventive_maintenance_score, 4),
            "duration_months": np.round(duration_months, 4),
            "event": event,
        }
    )

    metadata = {
        "dataset_type": "synthetic",
        "seed": seed,
        "n_assets": n_assets,
        "event_rate": float(event.mean()),
        "censoring_rate": float(1 - event.mean()),
        "weibull_shape": weibull_shape,
        "baseline_scale_months": baseline_scale_months,
        "ground_truth": {
            "higher_hazard": [
                "monthly_km",
                "engine_hours_per_month",
                "load_factor",
                "temperature_exposure",
                "prior_repairs",
                "service_delay_days",
                "route_severity",
                "Heavy vehicle class",
            ],
            "protective": [
                "preventive_maintenance_score",
                "Light vehicle class",
            ],
        },
        "disclaimer": (
            "Synthetic portfolio data only. It contains no proprietary OEM, "
            "customer, vehicle, warranty, or manufacturing information."
        ),
    }
    return df, metadata


def validate_dataset(df: pd.DataFrame) -> None:
    required = {
        "asset_id",
        "entry_month",
        "duration_months",
        "event",
        *FEATURE_COLUMNS,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(df["event"].unique()).issubset({0, 1}):
        raise ValueError("event must contain only 0 and 1.")
    if (df["duration_months"] <= 0).any():
        raise ValueError("duration_months must be positive.")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values.")
    if not set(df["vehicle_class"].unique()).issubset({"Light", "Medium", "Heavy"}):
        raise ValueError("Unexpected vehicle_class value.")


def save_dataset(df: pd.DataFrame, metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
