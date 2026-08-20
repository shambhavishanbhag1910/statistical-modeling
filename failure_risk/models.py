from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class WeibullAFTModel:
    """Small, dependency-light Weibull AFT implementation with censoring support."""

    beta_: np.ndarray | None = None
    shape_: float | None = None
    converged_: bool = False
    n_iter_: int = 0

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray) -> "WeibullAFTModel":
        X = np.asarray(X, dtype=float)
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=float)
        X_design = np.column_stack([np.ones(len(X)), X])

        def negative_log_likelihood(theta: np.ndarray) -> float:
            beta = theta[:-1]
            log_shape = theta[-1]
            shape = np.exp(log_shape)
            log_scale = X_design @ beta
            log_t = np.log(np.clip(duration, 1e-8, None))
            z = np.exp(np.clip(shape * (log_t - log_scale), -50, 50))
            log_survival = -z
            log_density = log_shape + (shape - 1.0) * log_t - shape * log_scale - z
            ll = event * log_density + (1.0 - event) * log_survival
            l2 = 1e-5 * np.sum(beta[1:] ** 2)
            return float(-np.sum(ll) + l2)

        initial = np.zeros(X_design.shape[1] + 1)
        initial[0] = np.log(np.median(duration))
        initial[-1] = np.log(1.5)
        result = minimize(
            negative_log_likelihood,
            initial,
            method="L-BFGS-B",
            options={"maxiter": 800},
        )
        self.beta_ = result.x[:-1]
        self.shape_ = float(np.exp(result.x[-1]))
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        return self

    def _log_scale(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.shape_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X]) @ self.beta_

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if np.any(times < 0):
            raise ValueError("Prediction times must be non-negative.")
        log_scale = self._log_scale(X)
        scale = np.exp(log_scale)
        return np.exp(-((times / scale) ** self.shape_))

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        # Larger value means greater failure risk / shorter survival.
        return -self._log_scale(X)
