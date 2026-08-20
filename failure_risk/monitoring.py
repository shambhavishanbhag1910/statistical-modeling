from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from failure_risk.data import NUMERIC_FEATURES


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    rows = []
    for feature in NUMERIC_FEATURES:
        statistic, p_value = ks_2samp(reference[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "alert": bool(p_value < 0.01 and statistic > 0.10),
            }
        )

    ref_dist = reference["vehicle_class"].value_counts(normalize=True)
    cur_dist = current["vehicle_class"].value_counts(normalize=True)
    categories = sorted(set(ref_dist.index) | set(cur_dist.index))
    total_variation = 0.5 * sum(abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in categories)

    return {
        "numeric_feature_drift": rows,
        "vehicle_class_total_variation": float(total_variation),
        "vehicle_class_alert": bool(total_variation > 0.10),
        "any_alert": bool(any(row["alert"] for row in rows) or total_variation > 0.10),
    }


def save_drift_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
