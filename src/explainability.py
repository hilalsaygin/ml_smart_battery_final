"""
src/explainability.py

Explainable AI (SHAP Analysis) Engine for EV Battery Diagnostics.

Unpacks model decisions to provide:
1. Global Feature Importance (Beeswarm & Bar plots)
2. Local Feature Attribution (Waterfall plot for high-risk thermal failure instances)
3. Console-level top feature attribution summary
"""

import os
import joblib
import logging
from typing import Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Directory Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH_JOB = os.path.join(BASE_DIR, "models", "thermal_failure_classifier.joblib")
MODEL_PATH_PKL = os.path.join(BASE_DIR, "models", "thermal_failure_classifier.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def ensure_reports_dir() -> str:
    """Ensures the reports output directory exists."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return REPORTS_DIR


def load_classifier_artifact() -> Tuple[Any, List[str], float]:
    """Loads the serialized classification artifact (.joblib or .pkl)."""
    target_path = MODEL_PATH_JOB if os.path.exists(MODEL_PATH_JOB) else MODEL_PATH_PKL
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Classification model not found at {MODEL_PATH_JOB} or {MODEL_PATH_PKL}")

    logger.info(f"Loading classifier artifact from: {target_path}")
    artifact = joblib.load(target_path)

    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline") or artifact.get("model")
        feature_cols = artifact.get("feature_cols", [])
        optimal_threshold = artifact.get("optimal_threshold", 0.5)
    else:
        pipeline = artifact
        feature_cols = getattr(pipeline, "feature_names_in_", [])
        optimal_threshold = 0.5

    return pipeline, list(feature_cols), optimal_threshold


def generate_synthetic_test_data(feature_cols: List[str], num_samples: int = 150) -> pd.DataFrame:
    """Generates evaluation test data matching expected feature schema."""
    np.random.seed(42)
    data = {}
    
    for col in feature_cols:
        if "temp" in col:
            data[col] = np.random.uniform(15.0, 55.0, num_samples)
        elif "resistance" in col or "re" in col or "rct" in col:
            data[col] = np.random.uniform(0.05, 0.45, num_samples)
        elif "cycle" in col:
            data[col] = np.random.uniform(50, 800, num_samples)
        elif "c_rate" in col:
            data[col] = np.random.choice([1.0, 2.0, 3.0, 5.0], num_samples)
        elif "stress" in col or "ratio" in col:
            data[col] = np.random.uniform(0.5, 3.5, num_samples)
        elif "persona" in col or "cluster" in col:
            data[col] = np.random.choice([0.0, 1.0], num_samples)
        else:
            data[col] = np.random.uniform(0.0, 1.0, num_samples)

    return pd.DataFrame(data)[feature_cols]


def extract_estimator_and_preprocessor(pipeline: Any) -> Tuple[Any, Any]:
    """Extracts final estimator step and optional preprocessor step from pipeline."""
    if hasattr(pipeline, "named_steps"):
        preprocessor = None
        if "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]
        elif "scaler" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["scaler"]
            
        estimator = pipeline.steps[-1][1]
        return estimator, preprocessor
    return pipeline, None


def run_shap_explainability(X_test: Optional[pd.DataFrame] = None) -> None:
    """
    Executes SHAP analysis pipeline using appropriate explainer based on model family.
    """
    reports_path = ensure_reports_dir()
    pipeline, feature_cols, optimal_threshold = load_classifier_artifact()

    if X_test is None:
        logger.info("No X_test provided. Generating standardized test sample dataset.")
        X_test = generate_synthetic_test_data(feature_cols)

    X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)
    estimator, preprocessor = extract_estimator_and_preprocessor(pipeline)

    # Transform features if a standalone preprocessor step exists
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_test_aligned)
        X_eval = pd.DataFrame(X_transformed, columns=feature_cols) if isinstance(X_transformed, np.ndarray) else X_transformed
    else:
        X_eval = X_test_aligned.copy()

    # Dynamic Explainer Selection
    model_type_str = str(type(estimator)).lower()
    is_tree_model = any(t in model_type_str for t in ["forest", "tree", "gbm", "xgb", "boost"])

    if is_tree_model:
        logger.info(f"Instantiating TreeExplainer for model: {type(estimator).__name__}")
        explainer = shap.TreeExplainer(estimator)
        shap_explanation = explainer(X_eval)
    elif "logistic" in model_type_str or "linear" in model_type_str:
        logger.info(f"Instantiating LinearExplainer for linear model: {type(estimator).__name__}")
        explainer = shap.LinearExplainer(estimator, X_eval)
        shap_explanation = explainer(X_eval)
    else:
        logger.info(f"Instantiating General Model Explainer for: {type(estimator).__name__}")
        explainer = shap.Explainer(estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator, X_eval)
        shap_explanation = explainer(X_eval)

    # Standardize Explanation object formatting across binary/multiclass output shapes
    if hasattr(shap_explanation, "values") and len(shap_explanation.values.shape) == 3:
        shap_exp_pos = shap_explanation[:, :, 1]
    else:
        shap_exp_pos = shap_explanation

    # Calculate probabilities for target instance selection
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test_aligned)[:, 1]
    elif hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X_eval)[:, 1]
    else:
        probs = estimator.predict(X_eval)

    # 1. SHAP Summary Beeswarm Plot
    summary_fig_path = os.path.join(reports_path, "shap_summary.png")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_exp_pos, show=False, max_display=12)
    plt.title("SHAP Beeswarm Summary Plot - Thermal Failure Risk", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(summary_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {summary_fig_path}")

    # 2. SHAP Bar Plot
    importance_fig_path = os.path.join(reports_path, "shap_importance.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_exp_pos, show=False, max_display=12)
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(importance_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {importance_fig_path}")

    # 3. Local Waterfall Plot for High-Risk Instance
    high_risk_idx = int(np.argmax(probs))
    waterfall_fig_path = os.path.join(reports_path, "shap_waterfall_single.png")
    
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_exp_pos[high_risk_idx], show=False, max_display=10)
    plt.title(
        f"Local SHAP Waterfall (Battery #{high_risk_idx} | Failure Prob: {probs[high_risk_idx]:.1%})", 
        fontsize=11, 
        pad=12
    )
    plt.tight_layout()
    plt.savefig(waterfall_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {waterfall_fig_path}")

    # 4. Console Top-5 Feature Attribution Summary
    shap_vals = shap_exp_pos.values if hasattr(shap_exp_pos, "values") else shap_exp_pos
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    feature_ranking = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_|SHAP|_Attribution": mean_abs_shap
    }).sort_values(by="Mean_|SHAP|_Attribution", ascending=False).reset_index(drop=True)

    print("\n" + "="*60)
    print("      TOP-5 GLOBAL FEATURE ATTRIBUTION SUMMARY (SHAP)")
    print("="*60)
    for idx, row in feature_ranking.head(5).iterrows():
        print(f"  {idx + 1}. {row['Feature']:<32} | Score: {row['Mean_|SHAP|_Attribution']:.5f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_shap_explainability()