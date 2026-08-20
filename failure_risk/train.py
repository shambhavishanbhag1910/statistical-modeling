from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from failure_risk.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    PLOT_DIR,
    TABLE_DIR,
)
from failure_risk.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    validate_dataset,
)
from failure_risk.evaluation import (
    CensoringKM,
    calibration_table,
    concordance_index,
    ipcw_brier_score,
    schoenfeld_time_correlation,
)
from failure_risk.models import WeibullAFTModel

EVAL_HORIZONS = [12.0, 24.0, 36.0]


def _cox_survival(result, X: np.ndarray, horizon: float) -> np.ndarray:
    return np.asarray(
        result.predict(
            exog=X,
            endog=np.full(len(X), horizon),
            pred_type="surv",
        ).predicted_values,
        dtype=float,
    )


def _plot_km(df: pd.DataFrame) -> float:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = float(df["load_factor"].median())
    group = (df["load_factor"] >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, value in [("Lower load", 0), ("Higher load", 1)]:
        subset = df[group == value]
        km = SurvfuncRight(subset["duration_months"], subset["event"])
        ax.step(km.surv_times, km.surv_prob, where="post", label=label)
    ax.set_xlabel("Component age (months)")
    ax.set_ylabel("Estimated survival probability")
    ax.set_title("Kaplan-Meier survival by operating load")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kaplan_meier_load.png", dpi=160)
    plt.close(fig)
    _, pvalue = survdiff(
        df["duration_months"].to_numpy(),
        df["event"].to_numpy(),
        group.to_numpy(),
    )
    return float(pvalue)


def _plot_calibration(table: pd.DataFrame, horizon: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        table["predicted_risk_mean"],
        table["observed_risk_km"],
        marker="o",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted event risk")
    ax.set_ylabel("Kaplan-Meier observed event risk")
    ax.set_title(f"Calibration at {int(horizon)} months")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"calibration_{int(horizon)}m.png", dpi=160)
    plt.close(fig)


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    df = pd.read_csv(data_path, parse_dates=["cohort_start_date"])
    validate_dataset(df)

    # Temporal cohort split: later cohorts are held out to mimic future deployment.
    split_month = int(df["entry_month"].quantile(0.72))
    train_df = df[df["entry_month"] <= split_month].copy()
    test_df = df[df["entry_month"] > split_month].copy()
    if len(test_df) < 500:
        split_month = int(df["entry_month"].quantile(0.65))
        train_df = df[df["entry_month"] <= split_month].copy()
        test_df = df[df["entry_month"] > split_month].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    X_train = np.asarray(preprocessor.fit_transform(train_df[FEATURE_COLUMNS]), dtype=float)
    X_test = np.asarray(preprocessor.transform(test_df[FEATURE_COLUMNS]), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())

    y_time_train = train_df["duration_months"].to_numpy(dtype=float)
    y_event_train = train_df["event"].to_numpy(dtype=int)
    y_time_test = test_df["duration_months"].to_numpy(dtype=float)
    y_event_test = test_df["event"].to_numpy(dtype=int)

    cox = PHReg(
        y_time_train,
        X_train,
        status=y_event_train,
        ties="breslow",
    ).fit(disp=False)
    cox_risk = np.asarray(cox.predict(exog=X_test, pred_type="lhr").predicted_values)
    cox_c_index = concordance_index(y_time_test, y_event_test, cox_risk)

    aft = WeibullAFTModel().fit(X_train, y_time_train, y_event_train)
    aft_risk = aft.risk_score(X_test)
    aft_c_index = concordance_index(y_time_test, y_event_test, aft_risk)

    censoring_km = CensoringKM.fit(y_time_train, y_event_train)
    cox_brier = {}
    aft_brier = {}
    calibration_paths = []
    for horizon in EVAL_HORIZONS:
        cox_surv = _cox_survival(cox, X_test, horizon)
        aft_surv = aft.predict_survival(X_test, np.full(len(X_test), horizon))
        cox_brier[str(int(horizon))] = ipcw_brier_score(
            y_time_test, y_event_test, cox_surv, horizon, censoring_km
        )
        aft_brier[str(int(horizon))] = ipcw_brier_score(
            y_time_test, y_event_test, aft_surv, horizon, censoring_km
        )

        cox_cal = calibration_table(
            y_time_test,
            y_event_test,
            1.0 - cox_surv,
            horizon,
        )
        path = TABLE_DIR / f"cox_calibration_{int(horizon)}m.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        cox_cal.to_csv(path, index=False)
        _plot_calibration(cox_cal, horizon)
        calibration_paths.append(str(path))

    cox_ibs = float(np.mean(list(cox_brier.values())))
    aft_ibs = float(np.mean(list(aft_brier.values())))

    # Prefer Cox PH when discrimination/calibration are effectively tied because
    # its hazard ratios are easier to explain to reliability and maintenance teams.
    # Choose AFT only when it demonstrates a material performance advantage.
    if cox_c_index >= aft_c_index - 0.01 and cox_ibs <= aft_ibs + 0.005:
        selected_type = "cox_ph"
        selected_model = cox
    else:
        selected_type = "weibull_aft"
        selected_model = aft

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ci_low = cox.params - 1.96 * cox.bse
    ci_high = cox.params + 1.96 * cox.bse
    hr = pd.DataFrame(
        {
            "feature": feature_names,
            "log_hazard_ratio": cox.params,
            "hazard_ratio": np.exp(cox.params),
            "ci_95_low": np.exp(ci_low),
            "ci_95_high": np.exp(ci_high),
            "p_value": cox.pvalues,
        }
    ).sort_values("hazard_ratio", ascending=False)
    hr.to_csv(TABLE_DIR / "cox_hazard_ratios.csv", index=False)

    ph_diag = schoenfeld_time_correlation(
        cox.schoenfeld_residuals,
        y_time_train,
        y_event_train,
        feature_names,
    )
    ph_diag.to_csv(TABLE_DIR / "cox_ph_diagnostics.csv", index=False)

    logrank_p = _plot_km(df)

    # Risk categories are data-driven from held-out 90-day conditional risk at
    # median current age. This avoids arbitrary hard-coded thresholds.
    median_age = float(np.median(y_time_test))
    if selected_type == "cox_ph":
        s_now = _cox_survival(selected_model, X_test, median_age)
        s_3m = _cox_survival(selected_model, X_test, median_age + 3.0)
    else:
        s_now = selected_model.predict_survival(X_test, np.full(len(X_test), median_age))
        s_3m = selected_model.predict_survival(X_test, np.full(len(X_test), median_age + 3.0))
    conditional_3m = 1.0 - np.clip(s_3m / np.clip(s_now, 1e-8, None), 0, 1)
    low_to_medium = float(np.quantile(conditional_3m, 0.50))
    medium_to_high = float(np.quantile(conditional_3m, 0.80))

    baseline_stats = {
        name: {
            "mean": float(train_df[name].mean()),
            "std": float(train_df[name].std(ddof=0)),
        }
        for name in NUMERIC_FEATURES
    }
    baseline_stats["vehicle_class_distribution"] = {
        str(k): float(v)
        for k, v in train_df["vehicle_class"].value_counts(normalize=True).items()
    }

    bundle = {
        "model_type": selected_type,
        "model": selected_model,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
        "feature_names": feature_names,
        "risk_thresholds": {
            "low_to_medium": low_to_medium,
            "medium_to_high": medium_to_high,
        },
        "training_baseline": baseline_stats,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(bundle, fh)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "event_rate": float(df["event"].mean()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "temporal_split_entry_month": split_month,
        },
        "cox_ph": {
            "c_index": cox_c_index,
            "brier_scores": cox_brier,
            "integrated_brier_proxy": cox_ibs,
        },
        "weibull_aft": {
            "c_index": aft_c_index,
            "brier_scores": aft_brier,
            "integrated_brier_proxy": aft_ibs,
            "shape": aft.shape_,
            "optimizer_converged": aft.converged_,
        },
        "selected_model": selected_type,
        "kaplan_meier_logrank_p_value_high_vs_low_load": logrank_p,
        "risk_thresholds": bundle["risk_thresholds"],
    }
    DEFAULT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata = {
        "model_version": "1.0.0",
        "selected_model": selected_type,
        "selection_reason": (
            "Cox PH and Weibull AFT had effectively tied holdout discrimination and calibration; "
            "Cox PH was selected for stakeholder-friendly hazard-ratio interpretation."
            if selected_type == "cox_ph"
            else "Weibull AFT demonstrated a material holdout performance advantage."
        ),
        "features": feature_names,
        "evaluation_horizons_months": EVAL_HORIZONS,
        "training_data": "synthetic commercial-vehicle component histories",
        "important_limitations": [
            "Synthetic data does not establish real-world OEM model performance.",
            "The Cox PH residual correlation table is a diagnostic heuristic, not a full formal PH test.",
            "The three-point Brier average is reported as an integrated-Brier proxy, not a continuous-time IBS integral.",
        ],
    }
    DEFAULT_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.data, args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
