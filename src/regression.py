"""
src/regression.py

State of Health (SoH %) Continuous Regression Module.
Trains, evaluates, and compares multiple regression models (Ridge, Polynomial Ridge, Random Forest)
using GroupKFold cross-validation. Outputs robust out-of-fold performance benchmarks,
actual vs. predicted plots, and serializes the winning model artifact.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

# CRITICAL FIX: Set non-interactive backend BEFORE importing pyplot to prevent Tkinter thread crashes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any, Optional

from sklearn.model_selection import KFold, GroupKFold, cross_validate, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Feature Definition aligned with preprocessing schema
DEFAULT_FEATURE_COLS = [
    "cycle_index",
    "ambient_temp_c",
    "rest_time_hours",
    "cumulative_time_hours",
    "long_rest_flag",
    "electrolyte_resistance_re",
    "charge_transfer_resistance_rct",
    "total_internal_resistance",
    "resistance_growth_ratio",
    "capacity_decay_rate_5c",
    "capacity_ema_5",
    "capacity_rolling_std_5"
]
TARGET_COL = "soh_percentage"


def get_model_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs candidate model pipelines with L2 regularization to prevent
    multicollinearity in linear/polynomial features.
    """
    pipelines = {
        "Ridge Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ]),
        "Polynomial Ridge (Degree 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=50.0, random_state=42))
        ]),
        "Random Forest Regressor": Pipeline([
            ("regressor", RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=1  # FIX: Set to 1 to avoid thread over-subscription when cross_validate uses n_jobs=-1
            ))
        ])
    }
    return pipelines


def evaluate_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    n_splits: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """
    Performs leakage-free GroupKFold Cross-Validation across candidate models
    and captures Out-Of-Fold (OOF) predictions for visual diagnosis.
    """
    pipelines = get_model_pipelines()
    results: List[Dict[str, Any]] = []
    oof_predictions: Dict[str, np.ndarray] = {}

    if groups is not None:
        unique_groups = groups.nunique()
        if unique_groups < n_splits:
            logger.warning(
                f"[WARNING] Requested n_splits={n_splits}, but only {unique_groups} unique battery groups available. "
                f"Adjusting n_splits down to {unique_groups}."
            )
            n_splits = max(2, unique_groups)

        logger.info(f"Using GroupKFold cross-validation (n_splits={n_splits}) grouped by battery ID.")
        cv = GroupKFold(n_splits=n_splits)
        cv_splits = list(cv.split(X, y, groups=groups))
    else:
        logger.warning("[SAFETY WARNING] Group column missing! Falling back to standard KFold. Data leakage across cycles may occur.")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y))

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    logger.info("Evaluating regression models across cross-validation folds...")
    
    for name, pipeline in pipelines.items():
        cv_res = cross_validate(
            pipeline, X, y, cv=cv_splits, scoring=scoring, n_jobs=-1
        )
        
        # Capture out-of-fold predictions for fair visual validation
        raw_oof_pred = cross_val_predict(pipeline, X, y, cv=cv_splits, n_jobs=-1)
        oof_predictions[name] = np.clip(raw_oof_pred, 0.0, 110.0)

        mae = -np.mean(cv_res["test_mae"])
        rmse = -np.mean(cv_res["test_rmse"])
        r2 = np.mean(cv_res["test_r2"])

        results.append({
            "Model": name,
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R² Score": round(float(r2), 4)
        })

    comparison_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False).reset_index(drop=True)
    
    logger.info("\n" + "="*55 + "\n       SOH REGRESSION MODEL COMPARISON BENCHMARK       \n" + "="*55)
    logger.info("\n" + comparison_df.to_string(index=False))

    return comparison_df, pipelines, oof_predictions


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred_oof: np.ndarray,
    model_name: str,
    output_path: str = "reports/figures/soh_actual_vs_predicted.png"
) -> None:
    """
    Generates and saves Out-Of-Fold (OOF) Actual vs. Predicted SOH scatter plot safely.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FIX: Use explicit figure and axes objects for safe resource handling
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred_oof, alpha=0.5, color="navy", edgecolors="k", s=30, label="OOF Predictions")
    
    min_val = min(y_true.min(), y_pred_oof.min())
    max_val = max(y_true.max(), y_pred_oof.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal (1:1 Perfect Fit)")
    
    ax.set_title(f"Out-Of-Fold Actual vs. Predicted SOH (%) - {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual State of Health (%)", fontsize=11)
    ax.set_ylabel("Predicted State of Health (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    fig.savefig(output_path, dpi=300)
    plt.close(fig)  # Explicitly close figure to release memory
    
    logger.info(f"OOF Actual vs. Predicted plot saved to '{output_path}'")


def train_and_save_soh_regressor(
    df: pd.DataFrame,
    feature_cols: List[str] = DEFAULT_FEATURE_COLS,
    target_col: str = TARGET_COL,
    group_col: Optional[str] = "battery_id",
    model_output_path: str = "models/best_soh_regressor.joblib"
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Full pipeline wrapper: validates inputs, evaluates candidates via leakage-free CV,
    plots out-of-fold generalization performance, fits the winning pipeline, and serializes artifact.
    """
    valid_feature_cols = [col for col in feature_cols if col in df.columns]
    if not valid_feature_cols:
        raise KeyError(f"None of the specified feature columns exist in DataFrame: {feature_cols}")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing from DataFrame.")

    X = df[valid_feature_cols].copy()
    y = df[target_col].copy()
    groups = df[group_col] if (group_col and group_col in df.columns) else None

    # Step 1: Leakage-Free Cross-Validation & Benchmark Comparison
    comparison_df, pipelines, oof_preds = evaluate_models_cv(X, y, groups=groups, n_splits=5)
    
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Winning model selected based on Out-Of-Fold R² Score: '{best_model_name}'")

    # Step 2: Plot Out-Of-Fold Predictions to verify true generalization
    plot_actual_vs_predicted(
        y_true=y.values,
        y_pred_oof=oof_preds[best_model_name],
        model_name=best_model_name
    )

    # Step 3: Fit Winning Pipeline on Full Dataset for Deployment
    winning_pipeline = pipelines[best_model_name]
    winning_pipeline.fit(X, y)

    # Step 4: Serialize Model Artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(winning_pipeline, model_output_path)
    logger.info(f"Saved winning SoH model pipeline to '{model_output_path}'")

    return winning_pipeline, comparison_df


if __name__ == "__main__":
    print("[TEST RUN] Executing Corrected SoH Regression Pipeline with Synthetic Data...")
    np.random.seed(42)
    n_samples = 800

    cycles = np.random.randint(1, 300, n_samples)
    r_re = np.random.uniform(0.04, 0.12, n_samples)
    r_rct = np.random.uniform(0.15, 0.40, n_samples)
    r_total = r_re + r_rct
    
    synthetic_soh = 100.0 - (0.08 * cycles) - (120.0 * r_total**2) + np.random.normal(0, 1.5, n_samples)
    synthetic_soh = np.clip(synthetic_soh, 50.0, 100.0)

    synthetic_df = pd.DataFrame({
        "battery_id": np.random.choice(["B0005", "B0006", "B0007", "B0018"], n_samples),
        "cycle_index": cycles,
        "ambient_temp_c": np.random.normal(24, 2, n_samples),
        "rest_time_hours": np.random.exponential(1.5, n_samples),
        "cumulative_time_hours": cycles * 2.5,
        "long_rest_flag": np.random.choice([0, 1], n_samples),
        "electrolyte_resistance_re": r_re,
        "charge_transfer_resistance_rct": r_rct,
        "total_internal_resistance": r_total,
        "resistance_growth_ratio": np.random.uniform(0.001, 0.05, n_samples),
        "capacity_decay_rate_5c": np.random.uniform(-0.005, -0.0001, n_samples),
        "capacity_ema_5": synthetic_soh / 50.0,
        "capacity_rolling_std_5": np.random.uniform(0.001, 0.02, n_samples),
        "soh_percentage": synthetic_soh
    })

    model, benchmark_df = train_and_save_soh_regressor(
        df=synthetic_df,
        group_col="battery_id"
    )