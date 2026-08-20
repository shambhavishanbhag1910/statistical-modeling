from pathlib import Path

import pandas as pd

SAMPLE = [
    {
        "current_age_months": 30,
        "monthly_km": 6200,
        "engine_hours_per_month": 145,
        "load_factor": 0.86,
        "temperature_exposure": 0.72,
        "prior_repairs": 2,
        "service_delay_days": 14,
        "route_severity": 0.82,
        "preventive_maintenance_score": 0.55,
        "vehicle_class": "Heavy",
    },
    {
        "current_age_months": 24,
        "monthly_km": 3200,
        "engine_hours_per_month": 75,
        "load_factor": 0.52,
        "temperature_exposure": 0.28,
        "prior_repairs": 0,
        "service_delay_days": 1,
        "route_severity": 0.35,
        "preventive_maintenance_score": 0.90,
        "vehicle_class": "Medium",
    },
]


def main() -> None:
    path = Path("data/scoring_sample.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SAMPLE).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
