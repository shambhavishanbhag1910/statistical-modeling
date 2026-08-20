from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from failure_risk.config import DEFAULT_MODEL_PATH


@dataclass
class RiskPrediction:
    risk_30d: float
    risk_60d: float
    risk_90d: float
    risk_level: str


class FailureRiskService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        with Path(model_path).open("rb") as fh:
            self.bundle = pickle.load(fh)

    def _survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        model = self.bundle["model"]
        if self.bundle["model_type"] == "cox_ph":
            return np.asarray(
                model.predict(exog=X, endog=times, pred_type="surv").predicted_values,
                dtype=float,
            )
        return model.predict_survival(X, times)

    def predict_one(self, row: dict) -> RiskPrediction:
        current_age = float(row.pop("current_age_months"))
        frame = pd.DataFrame([row], columns=self.bundle["feature_columns"])
        X = np.asarray(self.bundle["preprocessor"].transform(frame), dtype=float)

        s_now = float(self._survival(X, np.array([current_age]))[0])
        risks = []
        for months in (1.0, 2.0, 3.0):
            s_future = float(self._survival(X, np.array([current_age + months]))[0])
            conditional_survival = min(max(s_future / max(s_now, 1e-8), 0.0), 1.0)
            risks.append(1.0 - conditional_survival)

        thresholds = self.bundle["risk_thresholds"]
        if risks[2] >= thresholds["medium_to_high"]:
            level = "HIGH"
        elif risks[2] >= thresholds["low_to_medium"]:
            level = "MEDIUM"
        else:
            level = "LOW"

        return RiskPrediction(
            risk_30d=risks[0],
            risk_60d=risks[1],
            risk_90d=risks[2],
            risk_level=level,
        )

    def model_info(self) -> dict:
        return {
            "model_type": self.bundle["model_type"],
            "model_version": self.bundle["version"],
            "trained_at_utc": self.bundle["trained_at_utc"],
            "features": self.bundle["feature_columns"],
            "risk_thresholds": self.bundle["risk_thresholds"],
        }
