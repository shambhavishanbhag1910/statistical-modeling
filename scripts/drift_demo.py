from pathlib import Path

import numpy as np
import pandas as pd

from failure_risk.config import DEFAULT_DATA_PATH
from failure_risk.monitoring import drift_report, save_drift_report


def main() -> None:
    reference = pd.read_csv(DEFAULT_DATA_PATH)
    current = reference.sample(1_500, random_state=7).copy()
    # Introduce synthetic distribution shift to demonstrate monitoring.
    current["monthly_km"] *= 1.20
    current["load_factor"] = np.clip(current["load_factor"] + 0.08, 0, 1)
    report = drift_report(reference, current)
    path = Path("reports/drift_demo.json")
    save_drift_report(report, path)
    print(path)
    print(report)


if __name__ == "__main__":
    main()
