from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.duration.survfunc import SurvfuncRight


def concordance_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index where larger risk means earlier failure."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    concordant = 0.0
    comparable = 0.0
    n = len(duration)
    for i in range(n - 1):
        ti, ei, ri = duration[i], event[i], risk[i]
        t2 = duration[i + 1 :]
        e2 = event[i + 1 :]
        r2 = risk[i + 1 :]

        mask_i = (ti < t2) & (ei == 1)
        if np.any(mask_i):
            comparable += float(mask_i.sum())
            diff = ri - r2[mask_i]
            concordant += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())

        mask_j = (t2 < ti) & (e2 == 1)
        if np.any(mask_j):
            comparable += float(mask_j.sum())
            diff = r2[mask_j] - ri
            concordant += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())

    return float(concordant / comparable) if comparable else math.nan


@dataclass
class CensoringKM:
    times: np.ndarray
    survival: np.ndarray

    @classmethod
    def fit(cls, duration: np.ndarray, event: np.ndarray) -> "CensoringKM":
        # For censoring distribution, censoring is the event of interest.
        km = SurvfuncRight(duration, 1 - event)
        return cls(np.asarray(km.surv_times), np.asarray(km.surv_prob))

    def probability(self, t: float, left_limit: bool = False) -> float:
        side = "left" if left_limit else "right"
        idx = np.searchsorted(self.times, t, side=side) - 1
        if idx < 0:
            return 1.0
        return float(max(self.survival[idx], 1e-6))


def ipcw_brier_score(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_survival: np.ndarray,
    horizon: float,
    censoring_km: CensoringKM,
) -> float:
    """IPCW Brier score for right-censored survival outcomes."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    s_hat = np.asarray(predicted_survival, dtype=float)

    total = 0.0
    n = len(duration)
    g_horizon = censoring_km.probability(horizon)

    for ti, ei, si in zip(duration, event, s_hat):
        if ti <= horizon and ei == 1:
            weight = 1.0 / censoring_km.probability(float(ti), left_limit=True)
            total += weight * (0.0 - si) ** 2
        elif ti > horizon:
            weight = 1.0 / g_horizon
            total += weight * (1.0 - si) ** 2
        # Censored before horizon contributes zero under IPCW.
    return float(total / n)


def calibration_table(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_risk: np.ndarray,
    horizon: float,
    bins: int = 5,
) -> pd.DataFrame:
    """Compare predicted risk with Kaplan-Meier observed risk by risk quantile."""
    frame = pd.DataFrame(
        {"duration": duration, "event": event, "predicted_risk": predicted_risk}
    )
    frame["risk_bin"] = pd.qcut(frame["predicted_risk"], q=bins, duplicates="drop")
    rows: list[dict] = []
    for label, group in frame.groupby("risk_bin", observed=True):
        km = SurvfuncRight(group["duration"].to_numpy(), group["event"].to_numpy())
        idx = np.searchsorted(km.surv_times, horizon, side="right") - 1
        survival = 1.0 if idx < 0 else float(km.surv_prob[idx])
        rows.append(
            {
                "risk_bin": str(label),
                "n": int(len(group)),
                "predicted_risk_mean": float(group["predicted_risk"].mean()),
                "observed_risk_km": float(1.0 - survival),
            }
        )
    return pd.DataFrame(rows)


def schoenfeld_time_correlation(
    schoenfeld_residuals: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Diagnostic correlation of Schoenfeld residuals with log event time.

    This is a practical diagnostic, not a substitute for a full Grambsch-Therneau
    proportional-hazards test.
    """
    event_mask = np.asarray(event, dtype=int) == 1
    event_times = np.log(np.asarray(duration, dtype=float)[event_mask])
    residuals = np.asarray(schoenfeld_residuals)[event_mask]
    rows = []
    for idx, name in enumerate(feature_names):
        values = residuals[:, idx]
        valid = np.isfinite(values) & np.isfinite(event_times)
        if valid.sum() < 5 or np.std(values[valid]) == 0:
            corr, pvalue = math.nan, math.nan
        else:
            corr, pvalue = pearsonr(values[valid], event_times[valid])
        rows.append({"feature": name, "correlation": corr, "p_value": pvalue})
    return pd.DataFrame(rows)
