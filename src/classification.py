"""
src/classification.py

Thermal & End-of-Life (EOL) Failure Risk Binary Classification Module.

Functional Responsibilities:
1. One-hot encodes nominal cluster features to prevent distance distortion.
2. Uses imblearn Pipelines to execute SMOTE rebalancing inside CV folds without data leakage or double-weighting.
3. Performs Stratified K-Fold Cross-Validation across candidate classifiers.
4. Tunes decision threshold on Out-Of-Fold (OOF) validation probabilities to hit target Recall >= 0.90.
5. Evaluates performance on an untouched held-out test set and saves ROC-AUC curves.
6. Serializes model artifact package using joblib.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

# CRITICAL FIX: Non-interactive backend set before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any, Optional

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLS = [
    "cycle_index",
    "ambient_temp_c",
    "electrolyte_resistance_re",
    "charge_transfer_resistance_rct",
    "total_internal_resistance",
    "capacity_decay_rate_5c",
    "resistance_growth_ratio",
    "driver_persona_cluster"
]
TARGET_COL = "thermal_failure_flag"


def preprocess_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Preprocesses feature matrix by one-hot encoding nominal categorical features
    such as 'driver_persona_cluster'.
    """
    X = df[feature_cols].copy()
    if "driver_persona_cluster" in X.columns:
        X = pd.get_dummies(X, columns=["driver_persona_cluster"], drop_first=True, dtype=float)
    return X


def build_candidate_pipelines() -> Dict[str, ImbPipeline]:
    """
    Constructs candidate classification pipelines binding scaling, SMOTE, and model fitting.
    Class weights are left neutral to avoid double-counting imbalance correction with SMOTE.
    """
    pipelines = {
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "K-Nearest Neighbors (K=5)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=1))
        ]),
        "Support Vector Classifier (RBF)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Random Forest Classifier": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                random_state=42,
                n_jobs=1  # Avoid process thrashing inside CV
            ))
        ])
    }
    return pipelines


def tune_threshold_oof(
    y_train: np.ndarray,
    oof_probs: np.ndarray,
    target_recall: float = 0.90
) -> Tuple[float, Dict[str, float]]:
    """
    Searches for optimal decision threshold on Out-Of-Fold (OOF) training probabilities
    to achieve target Recall >= 0.90 while maximizing F1-Score.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    best_f1 = -1.0
    metrics_at_optimal: Dict[str, float] = {}

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        prec = precision_score(y_train, preds, zero_division=0)
        f1 = f1_score(y_train, preds, zero_division=0)

        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            metrics_at_optimal = {
                "threshold": round(best_threshold, 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4)
            }

    if not metrics_at_optimal:
        logger.warning(f"Target recall {target_recall} not reached during OOF tuning. Defaulting to low threshold.")
        best_threshold = 0.15
        preds = (oof_probs >= best_threshold).astype(int)
        metrics_at_optimal = {
            "threshold": best_threshold,
            "precision": round(float(precision_score(y_train, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_train, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_train, preds, zero_division=0)), 4)
        }

    logger.info(
        f"OOF Decision Threshold Tuning Result:\n"
        f"  - Optimal Threshold: {metrics_at_optimal['threshold']}\n"
        f"  - OOF Recall:        {metrics_at_optimal['recall']:.4f} (Target >= {target_recall})\n"
        f"  - OOF Precision:     {metrics_at_optimal['precision']:.4f}\n"
        f"  - OOF F1-Score:      {metrics_at_optimal['f1_score']:.4f}"
    )

    return best_threshold, metrics_at_optimal


def evaluate_test_performance(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluates the final fitted model on the untouched test set using the tuned threshold.
    """
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)

    metrics = {
        "Precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    }
    return metrics, test_probs


def plot_combined_roc_curves(
    test_probs_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: str = "reports/figures/thermal_failure_roc_curves.png"
) -> None:
    """
    Generates overlayed ROC curves on the test set for candidate pipelines.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, y_prob), color in zip(test_probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Test Set ROC-AUC Comparison - Thermal Failure Risk Classifiers", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved test set ROC curve plot to '{output_path}'")


def train_and_save_thermal_failure_classifier(
    df: pd.DataFrame,
    feature_cols: List[str] = DEFAULT_FEATURE_COLS,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    target_recall: float = 0.90,
    model_output_path: str = "models/thermal_failure_classifier.joblib"
) -> Dict[str, Any]:
    """
    Executes leakage-free classification pipeline: preprocessing, CV benchmarking,
    OOF threshold tuning, holdout test evaluation, and artifact serialization.
    """
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns: {missing_cols}")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing.")

    # Step 1: Preprocess Features (Handle Categoricals)
    X = preprocess_features(df, feature_cols)
    y = df[target_col].values

    # Step 2: Hold Out Clean Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Step 3: Stratified K-Fold CV Benchmarking & OOF Probability Estimation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_candidate_pipelines()
    
    cv_benchmark_results = []
    oof_predictions: Dict[str, np.ndarray] = {}

    logger.info("Evaluating candidate classification pipelines via Stratified 5-Fold CV...")
    for name, pipeline in pipelines.items():
        oof_prob = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        oof_predictions[name] = oof_prob
        cv_auc = roc_auc_score(y_train, oof_prob)
        
        cv_benchmark_results.append({
            "Model": name,
            "CV ROC-AUC": round(float(cv_auc), 4)
        })

    benchmark_df = pd.DataFrame(cv_benchmark_results).sort_values(by="CV ROC-AUC", ascending=False).reset_index(drop=True)
    logger.info("\n" + "="*50 + "\n     CROSS-VALIDATION ROC-AUC BENCHMARK     \n" + "="*50)
    logger.info("\n" + benchmark_df.to_string(index=False))

    # Step 4: Select Winning Model Based on CV ROC-AUC
    winning_model_name = benchmark_df.iloc[0]["Model"]
    winning_pipeline = pipelines[winning_model_name]
    logger.info(f"Winning model pipeline: '{winning_model_name}'")

    # Step 5: Tune Decision Threshold on Training Set OOF Predictions
    optimal_threshold, oof_metrics = tune_threshold_oof(
        y_train=y_train,
        oof_probs=oof_predictions[winning_model_name],
        target_recall=target_recall
    )

    # Step 6: Fit Winning Pipeline on Entire Training Set and Evaluate on Untouched Test Set
    winning_pipeline.fit(X_train, y_train)

    test_probs_dict = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        test_probs_dict[name] = pipeline.predict_proba(X_test)[:, 1]

    plot_combined_roc_curves(test_probs_dict, y_test)

    final_test_metrics, _ = evaluate_test_performance(
        model=winning_pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold
    )

    logger.info(
        f"Final Held-Out Test Set Performance ({winning_model_name} @ Threshold = {optimal_threshold}):\n"
        f"  - Precision: {final_test_metrics['Precision']}\n"
        f"  - Recall:    {final_test_metrics['Recall']}\n"
        f"  - F1-Score:  {final_test_metrics['F1-Score']}\n"
        f"  - ROC-AUC:   {final_test_metrics['ROC-AUC']}\n"
        f"  - Confusion Matrix: TN={final_test_metrics['TN']}, FP={final_test_metrics['FP']}, "
        f"FN={final_test_metrics['FN']}, TP={final_test_metrics['TP']}"
    )

    # Step 7: Serialize Model Package via Joblib
    artifact_package = {
        "model_name": winning_model_name,
        "pipeline": winning_pipeline,
        "optimal_threshold": optimal_threshold,
        "feature_cols": list(X.columns),
        "test_metrics": final_test_metrics
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(artifact_package, model_output_path)
    logger.info(f"Saved classification artifact package to '{model_output_path}'")

    return artifact_package


if __name__ == "__main__":
    print("[TEST RUN] Executing Corrected Thermal Failure Classification Pipeline...")
    np.random.seed(42)
    n_samples = 1000

    r_re = np.random.uniform(0.04, 0.12, n_samples)
    r_rct = np.random.uniform(0.12, 0.35, n_samples)
    r_total = r_re + r_rct
    temps = np.random.normal(25, 8, n_samples)

    risk_score = 1 / (1 + np.exp(-(-8.0 + 18.0 * r_total + 0.1 * temps)))
    failure_flags = (risk_score > 0.72).astype(int)

    synthetic_df = pd.DataFrame({
        "cycle_index": np.random.randint(10, 300, n_samples),
        "ambient_temp_c": temps,
        "electrolyte_resistance_re": r_re,
        "charge_transfer_resistance_rct": r_rct,
        "total_internal_resistance": r_total,
        "capacity_decay_rate_5c": np.random.uniform(-0.005, -0.0001, n_samples),
        "resistance_growth_ratio": np.random.uniform(0.001, 0.05, n_samples),
        "driver_persona_cluster": np.random.choice([0, 1, 2], n_samples),
        "thermal_failure_flag": failure_flags
    })

    artifact = train_and_save_thermal_failure_classifier(df=synthetic_df)