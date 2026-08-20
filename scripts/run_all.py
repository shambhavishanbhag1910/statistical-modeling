from __future__ import annotations

import json

from failure_risk.config import DEFAULT_DATA_PATH
from failure_risk.data import generate_synthetic_fleet, save_dataset
from failure_risk.train import train_and_evaluate


def main() -> None:
    if not DEFAULT_DATA_PATH.exists():
        print("Synthetic dataset not found. Generating it now...")
        df, metadata = generate_synthetic_fleet(n_assets=8_000, seed=42)
        save_dataset(df, metadata, DEFAULT_DATA_PATH)

    print("Training Cox PH and Weibull AFT models, evaluating, and saving artifacts...")
    metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))
    print("\nDone. Start the API with:")
    print("  uvicorn failure_risk.api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
