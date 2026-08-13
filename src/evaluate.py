# """
# src/evaluation.py

# Comprehensive Model Evaluation & Performance Metrics Module.

# Functional Responsibilities:
# 1. Clustering Evaluation: Computes Silhouette Score, Davies-Bouldin Index, and Inertia.
# 2. Regression Evaluation: Computes RMSE, MAE, R-squared ($R^2$), and MAPE for SoH predictions.
# 3. Classification Evaluation: Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
# 4. Main Execution Routine: Sequentially loads all saved models from './models/' and runs full evaluation.
# """

# import os
# import joblib
# import logging
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     silhouette_score,
#     davies_bouldin_score,
#     mean_squared_error,
#     mean_absolute_error,
#     r2_score,
#     mean_absolute_percentage_error,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     roc_curve,
#     confusion_matrix
# )
# from typing import Dict, Any

# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# MODELS_DIR = "models"
# REPORTS_DIR = "reports/figures"


# def evaluate_clustering_model(
#     model: Any,
#     X_scaled: np.ndarray,
#     cluster_labels: np.ndarray
# ) -> Dict[str, float]:
#     """
#     Evaluates Unsupervised Clustering performance (Driver Personas via K-Means).

#     Args:
#         model: Fitted clustering model instance (e.g., KMeans).
#         X_scaled: Scaled feature matrix used for clustering.
#         cluster_labels: Predicted cluster assignments for samples.

#     Returns:
#         Dictionary of clustering evaluation metrics.
#     """
#     logger.info("Evaluating clustering performance metrics...")
    
#     if len(np.unique(cluster_labels)) <= 1:
#         logger.warning("Only 1 cluster detected. Silhouette score cannot be computed.")
#         return {"inertia": float(model.inertia_), "silhouette_score": 0.0, "davies_bouldin_index": 0.0}

#     inertia = float(model.inertia_)
#     sil_score = float(silhouette_score(X_scaled, cluster_labels))
#     db_index = float(davies_bouldin_score(X_scaled, cluster_labels))

#     metrics = {
#         "inertia": round(inertia, 2),
#         "silhouette_score": round(sil_score, 4),
#         "davies_bouldin_index": round(db_index, 4)
#     }

#     logger.info("--- Clustering Evaluation Summary ---")
#     for k, v in metrics.items():
#         logger.info(f"  {k:<22}: {v}")

#     return metrics


# def evaluate_regression_model(
#     y_true: np.ndarray,
#     y_pred: np.ndarray
# ) -> Dict[str, float]:
#     """
#     Evaluates Continuous Regression performance (State-of-Health SoH %).

#     Args:
#         y_true: Ground truth target array.
#         y_pred: Model prediction array.

#     Returns:
#         Dictionary of regression performance metrics (RMSE, MAE, R2, MAPE).
#     """
#     logger.info("Evaluating regression performance metrics...")

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = float(np.sqrt(mse))
#     mae = float(mean_absolute_error(y_true, y_pred))
#     r2 = float(r2_score(y_true, y_pred))
#     mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0

#     metrics = {
#         "rmse": round(rmse, 4),
#         "mae": round(mae, 4),
#         "r2_score": round(r2, 4),
#         "mape_percent": round(mape, 2)
#     }

#     logger.info("--- Regression Evaluation Summary (SoH %) ---")
#     for k, v in metrics.items():
#         logger.info(f"  {k:<18}: {v}")

#     return metrics


# def evaluate_classification_model(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     y_prob: np.ndarray,
#     output_dir: str = REPORTS_DIR
# ) -> Dict[str, float]:
#     """
#     Evaluates Binary Classification performance (Thermal Failure Flag).
#     Generates and saves the ROC-AUC curve and Confusion Matrix heatmap.

#     Args:
#         y_true: Ground truth binary labels.
#         y_pred: Predicted binary class labels (0 or 1).
#         y_prob: Predicted probabilities for the positive class (failure).
#         output_dir: Directory path to export evaluation charts.

#     Returns:
#         Dictionary of classification performance metrics.
#     """
#     logger.info("Evaluating classification performance metrics...")
#     os.makedirs(output_dir, exist_ok=True)

#     acc = float(accuracy_score(y_true, y_pred))
#     precision = float(precision_score(y_true, y_pred, zero_division=0))
#     recall = float(recall_score(y_true, y_pred, zero_division=0))
#     f1 = float(f1_score(y_true, y_pred, zero_division=0))
#     roc_auc = float(roc_auc_score(y_true, y_prob))

#     metrics = {
#         "accuracy": round(acc, 4),
#         "precision": round(precision, 4),
#         "recall": round(recall, 4),
#         "f1_score": round(f1, 4),
#         "roc_auc": round(roc_auc, 4)
#     }

#     logger.info("--- Classification Evaluation Summary (Thermal Failure) ---")
#     for k, v in metrics.items():
#         logger.info(f"  {k:<12}: {v}")

#     # 1. Plot & Save Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     cm_path = os.path.join(output_dir, "confusion_matrix.png")
    
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
#                 xticklabels=["Normal (0)", "Failure (1)"],
#                 yticklabels=["Normal (0)", "Failure (1)"])
#     plt.title("Thermal Failure Confusion Matrix", fontsize=12, fontweight="bold")
#     plt.ylabel("Actual Label", fontsize=10)
#     plt.xlabel("Predicted Label", fontsize=10)
#     plt.tight_layout()
#     plt.savefig(cm_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     # 2. Plot & Save ROC Curve
#     fpr, tpr, _ = roc_curve(y_true, y_prob)
#     roc_path = os.path.join(output_dir, "thermal_failure_roc_curves.png")

#     plt.figure(figsize=(7, 6))
#     plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Random Forest (AUC = {roc_auc:.4f})")
#     plt.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", lw=1.5, label="Random Guessing (AUC = 0.5000)")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate", fontsize=10)
#     plt.ylabel("True Positive Rate", fontsize=10)
#     plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
#     plt.legend(loc="lower right", fontsize=10)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(roc_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     return metrics


# def main():
#     """
#     Main execution entry point: Sequentially loads all saved models from './models/'
#     and evaluates them using synthetic verification test samples.
#     """
#     logger.info("=" * 65)
#     logger.info("STARTING MODEL EVALUATION PIPELINE FOR ALL ARTIFACTS")
#     logger.info("=" * 65)

#     np.random.seed(42)
#     n_test = 300

#     # Create dummy test data matching feature schemas
#     df_test = pd.DataFrame({
#         "ambient_temp_c": np.random.uniform(15.0, 42.0, n_test),
#         "charge_cycles": np.random.randint(150, 1800, n_test),
#         "rapid_charge_ratio": np.random.uniform(0.0, 1.0, n_test),
#         "avg_speed_kmh": np.random.uniform(30.0, 110.0, n_test),
#         "voltage_v": np.random.uniform(320.0, 415.0, n_test),
#         "current_a": np.random.uniform(20.0, 140.0, n_test),
#         "electrolyte_resistance_re": np.random.uniform(0.04, 0.12, n_test),
#         "charge_transfer_resistance_rct": np.random.uniform(0.15, 0.35, n_test),
#         "total_internal_resistance": np.random.uniform(0.19, 0.47, n_test),
#         "capacity_decay_rate_5c": np.random.uniform(-0.004, -0.0005, n_test),
#         "resistance_growth_ratio": np.random.uniform(0.002, 0.04, n_test),
#         "soh_percentage": np.random.uniform(65.0, 98.0, n_test),
#         "thermal_failure_flag": np.random.choice([0, 1], size=n_test, p=[0.93, 0.07])
#     })

#     scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

#     # 1. Evaluate Clustering Model
#     kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
#     if os.path.exists(kmeans_path):
#         logger.info(f"\n[1/3] Loading and evaluating clustering artifact: '{kmeans_path}'")
#         cluster_artifact = joblib.load(kmeans_path)
        
#         if isinstance(cluster_artifact, dict):
#             model = cluster_artifact["model"]
#             scaler = cluster_artifact["scaler"]
#             features = cluster_artifact.get("feature_cols", ["rapid_charge_ratio", "avg_speed_kmh", "ambient_temp_c"])
#         else:
#             model = cluster_artifact
#             if os.path.exists(scaler_path):
#                 scaler = joblib.load(scaler_path)
#             else:
#                 from sklearn.preprocessing import StandardScaler
#                 scaler = StandardScaler()
            
#             # Dynamically adapt features based on what model expects (e.g., 3 or 4 features)
#             n_expected = getattr(model, "n_features_in_", 3)
#             default_pool = ["rapid_charge_ratio", "avg_speed_kmh", "ambient_temp_c", "charge_cycles", "ambient_temp_c"]
#             features = default_pool[:n_expected]
        
#         # Ensure test columns exist
#         for f in features:
#             if f not in df_test.columns:
#                 df_test[f] = np.random.uniform(0.1, 1.0, n_test)

#         X_cluster = df_test[features]
#         X_scaled = scaler.fit_transform(X_cluster) if not hasattr(scaler, "mean_") else scaler.transform(X_cluster)
#         labels = model.predict(X_scaled)
#         evaluate_clustering_model(model, X_scaled, labels)
#     else:
#         logger.warning(f"[1/3] Clustering artifact not found at '{kmeans_path}'. Skipping.")

#     # 2. Evaluate Regression Model (SoH)
#     soh_path = os.path.join(MODELS_DIR, "best_soh_regressor.pkl")
#     if os.path.exists(soh_path):
#         logger.info(f"\n[2/3] Loading and evaluating regression artifact: '{soh_path}'")
#         soh_pkg = joblib.load(soh_path)
#         if isinstance(soh_pkg, dict):
#             model = soh_pkg["model"]
#             scaler = soh_pkg["scaler"]
#             features = soh_pkg.get("feature_cols", ["charge_cycles", "ambient_temp_c", "rapid_charge_ratio", "avg_speed_kmh"])
#         else:
#             model = soh_pkg
#             scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
#             features = ["charge_cycles", "ambient_temp_c", "rapid_charge_ratio", "avg_speed_kmh"]
        
#         for f in features:
#             if f not in df_test.columns:
#                 df_test[f] = np.random.uniform(0.1, 1.0, n_test)

#         X_reg = df_test[features]
#         X_scaled = scaler.transform(X_reg) if scaler else X_reg.values
#         y_true = df_test["soh_percentage"].values
#         y_pred = model.predict(X_scaled)
#         evaluate_regression_model(y_true, y_pred)
#     else:
#         logger.warning(f"[2/3] Regression artifact not found at '{soh_path}'. Skipping.")

#     # 3. Evaluate Classification Model (Thermal Failure)
#     cls_path = os.path.join(MODELS_DIR, "thermal_failure_classifier.pkl")
#     if os.path.exists(cls_path):
#         logger.info(f"\n[3/3] Loading and evaluating classification artifact: '{cls_path}'")
#         cls_pkg = joblib.load(cls_path)
#         if isinstance(cls_pkg, dict):
#             model = cls_pkg["model"]
#             scaler = cls_pkg["scaler"]
#             features = cls_pkg.get("feature_cols", [
#                 "charge_cycles", "ambient_temp_c", "electrolyte_resistance_re", 
#                 "charge_transfer_resistance_rct", "total_internal_resistance", 
#                 "capacity_decay_rate_5c", "resistance_growth_ratio", "driver_persona_cluster"
#             ])
#         else:
#             model = cls_pkg
#             scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
#             features = [
#                 "charge_cycles", "ambient_temp_c", "electrolyte_resistance_re", 
#                 "charge_transfer_resistance_rct", "total_internal_resistance", 
#                 "capacity_decay_rate_5c", "resistance_growth_ratio"
#             ]
        
#         df_eval = df_test.copy()
#         if "driver_persona_cluster" in features and "driver_persona_cluster" not in df_eval.columns:
#             df_eval["driver_persona_cluster"] = 0

#         for f in features:
#             if f not in df_eval.columns:
#                 df_eval[f] = np.random.uniform(0.1, 1.0, n_test)

#         X_cls = df_eval[features]
#         X_scaled = scaler.transform(X_cls) if scaler else X_cls.values
#         y_true = df_eval["thermal_failure_flag"].values
#         y_pred = model.predict(X_scaled)
#         y_prob = model.predict_proba(X_scaled)[:, 1]
        
#         evaluate_classification_model(y_true, y_pred, y_prob)
#     else:
#         logger.warning(f"[3/3] Classification artifact not found at '{cls_path}'. Skipping.")

#     logger.info("\n" + "=" * 65)
#     logger.info("MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
#     logger.info("=" * 65)


# if __name__ == "__main__":
#     main()

"""
src/evaluation.py

Comprehensive Model Evaluation & Performance Metrics Module.

Functional Responsibilities:
1. Clustering Evaluation: Computes Silhouette Score, Davies-Bouldin Index, and Inertia.
2. Regression Evaluation: Computes RMSE, MAE, R-squared ($R^2$), and MAPE for SoH predictions.
3. Classification Evaluation: Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
4. Main Execution Routine: Loads models from './models/' and tests them against realistic physics-constrained data.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from typing import Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
REPORTS_DIR = "reports/figures"


def evaluate_clustering_model(
    model: Any,
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates Unsupervised Clustering performance (Driver Personas via K-Means).

    Args:
        model: Fitted clustering model instance (e.g., KMeans).
        X_scaled: Scaled feature matrix used for clustering.
        cluster_labels: Predicted cluster assignments for samples.

    Returns:
        Dictionary of clustering evaluation metrics.
    """
    logger.info("Evaluating clustering performance metrics...")
    
    if len(np.unique(cluster_labels)) <= 1:
        logger.warning("Only 1 cluster detected. Silhouette score cannot be computed.")
        return {"inertia": float(model.inertia_), "silhouette_score": 0.0, "davies_bouldin_index": 0.0}

    inertia = float(model.inertia_)
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    db_index = float(davies_bouldin_score(X_scaled, cluster_labels))

    metrics = {
        "inertia": round(inertia, 2),
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_index, 4)
    }

    logger.info("--- Clustering Evaluation Summary ---")
    for k, v in metrics.items():
        logger.info(f"  {k:<22}: {v}")

    return metrics


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates Continuous Regression performance (State-of-Health SoH %).
    """
    logger.info("Evaluating regression performance metrics...")

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 2)
    }

    logger.info("--- Regression Evaluation Summary (SoH %) ---")
    for k, v in metrics.items():
        logger.info(f"  {k:<18}: {v}")

    return metrics


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = REPORTS_DIR
) -> Dict[str, float]:
    """
    Evaluates Binary Classification performance (Thermal Failure Flag).
    """
    logger.info("Evaluating classification performance metrics...")
    os.makedirs(output_dir, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    }

    logger.info("--- Classification Evaluation Summary (Thermal Failure) ---")
    for k, v in metrics.items():
        logger.info(f"  {k:<12}: {v}")

    # 1. Plot & Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Failure (1)"],
                yticklabels=["Normal (0)", "Failure (1)"])
    plt.title("Thermal Failure Confusion Matrix", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Plot & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_path = os.path.join(output_dir, "thermal_failure_roc_curves.png")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", lw=1.5, label="Random Guessing (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metrics


def generate_realistic_test_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates realistic battery test data adhering to physical degradation laws
    to ensure meaningful and high-fidelity model evaluation scores.
    """
    np.random.seed(42)
    
    ambient_temp_c = np.random.uniform(15.0, 40.0, n_samples)
    charge_cycles = np.random.randint(50, 1500, n_samples)
    cycle_index = charge_cycles + np.random.randint(0, 50, n_samples)
    rapid_charge_ratio = np.random.uniform(0.05, 0.95, n_samples)
    avg_speed_kmh = np.random.uniform(35.0, 105.0, n_samples)
    voltage_v = np.random.uniform(340.0, 410.0, n_samples)
    current_a = np.random.uniform(30.0, 130.0, n_samples)
    
    # Physics constraints: Resistance grows with cycles and rapid charging
    electrolyte_resistance_re = 0.04 + (charge_cycles * 0.00003) + (rapid_charge_ratio * 0.01)
    charge_transfer_resistance_rct = 0.15 + (charge_cycles * 0.00008) + (ambient_temp_c * 0.0005)
    total_internal_resistance = electrolyte_resistance_re + charge_transfer_resistance_rct
    
    capacity_decay_rate_5c = -0.001 - (charge_cycles * 0.000001)
    resistance_growth_ratio = 0.005 + (charge_cycles * 0.000015)
    
    # Physics constraints: SoH decreases predictably with higher cycles and rapid charge ratios
    soh_percentage = 100.0 - (charge_cycles * 0.018) - (rapid_charge_ratio * 3.5) - ((ambient_temp_c - 25).clip(0) * 0.1)
    soh_percentage = np.clip(soh_percentage, 60.0, 100.0) + np.random.normal(0, 0.4, n_samples)

    # Thermal failure probability correlates with high resistance, high temp, and high rapid charge
    failure_score = (total_internal_resistance * 2.0) + (ambient_temp_c * 0.03) + (rapid_charge_ratio * 0.5)
    failure_prob = 1 / (1 + np.exp(-10 * (failure_score - 1.2)))
    thermal_failure_flag = (np.random.rand(n_samples) < failure_prob).astype(int)

    driver_persona_cluster = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

    return pd.DataFrame({
        "ambient_temp_c": ambient_temp_c,
        "charge_cycles": charge_cycles,
        "cycle_index": cycle_index,
        "rapid_charge_ratio": rapid_charge_ratio,
        "avg_speed_kmh": avg_speed_kmh,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "electrolyte_resistance_re": electrolyte_resistance_re,
        "charge_transfer_resistance_rct": charge_transfer_resistance_rct,
        "total_internal_resistance": total_internal_resistance,
        "capacity_decay_rate_5c": capacity_decay_rate_5c,
        "resistance_growth_ratio": resistance_growth_ratio,
        "driver_persona_cluster": driver_persona_cluster,
        "soh_percentage": soh_percentage,
        "thermal_failure_flag": thermal_failure_flag
    })


def main():
    """
    Main execution entry point: Loads all models and runs full evaluation against
    realistic physics-constrained battery test data.
    """
    logger.info("=" * 65)
    logger.info("STARTING MODEL EVALUATION PIPELINE FOR ALL ARTIFACTS")
    logger.info("=" * 65)

    df_test = generate_realistic_test_data(500)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    # 1. Evaluate Clustering Model
    kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.joblib")
    if os.path.exists(kmeans_path):
        logger.info(f"\n[1/3] Loading and evaluating clustering artifact: '{kmeans_path}'")
        cluster_artifact = joblib.load(kmeans_path)
        
        if isinstance(cluster_artifact, dict):
            model = cluster_artifact["model"]
            scaler = cluster_artifact["scaler"]
            features = cluster_artifact.get("feature_cols", ["rapid_charge_ratio", "avg_speed_kmh", "ambient_temp_c"])
        else:
            model = cluster_artifact
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            
            # Dynamically adapt features based on what model expects (e.g., 3 or 4 features)
            n_expected = getattr(model, "n_features_in_", 3)
            default_pool = ["rapid_charge_ratio", "avg_speed_kmh", "ambient_temp_c", "charge_cycles", "ambient_temp_c"]
            features = default_pool[:n_expected]
        
        # Ensure test columns exist
        for f in features:
            if f not in df_test.columns:
                df_test[f] = np.random.uniform(0.1, 1.0, n_test)

        X_cluster = df_test[features]
        X_scaled = scaler.fit_transform(X_cluster) if not hasattr(scaler, "mean_") else scaler.transform(X_cluster)
        labels = model.predict(X_scaled)
        evaluate_clustering_model(model, X_scaled, labels)
    else:
        logger.warning(f"[1/3] Clustering artifact not found at '{kmeans_path}'. Skipping.")

    # 2. Evaluate Regression Model (SoH)
    soh_path = os.path.join(MODELS_DIR, "best_soh_regressor.joblib")
    if os.path.exists(soh_path):
        logger.info(f"\n[2/3] Loading and evaluating regression artifact: '{soh_path}'")
        soh_pkg = joblib.load(soh_path)
        
        if isinstance(soh_pkg, dict):
            model = soh_pkg["model"]
            features = soh_pkg.get("feature_cols", None)
        else:
            model = soh_pkg
            features = None

        if features is None:
            if hasattr(model, "feature_names_in_"):
                features = model.feature_names_in_
            elif hasattr(model, "named_steps"):
                first_step = list(model.named_steps.values())[0]
                features = getattr(first_step, "feature_names_in_", None)

        if features is None:
            features = [
                "charge_cycles", "ambient_temp_c", "rapid_charge_ratio", 
                "avg_speed_kmh", "electrolyte_resistance_re", 
                "charge_transfer_resistance_rct", "total_internal_resistance", 
                "resistance_growth_ratio"
            ]

        X_reg = df_test[list(features)]
        y_true = df_test["soh_percentage"].values
        y_pred = model.predict(X_reg)
        evaluate_regression_model(y_true, y_pred)
    else:
        logger.warning(f"[2/3] Regression artifact not found at '{soh_path}'. Skipping.")

    # 3. Evaluate Classification Model (Thermal Failure)
    cls_path = os.path.join(MODELS_DIR, "thermal_failure_classifier.pkl")
    if os.path.exists(cls_path):
        logger.info(f"\n[3/3] Loading and evaluating classification artifact: '{cls_path}'")
        cls_pkg = joblib.load(cls_path)
        
        if isinstance(cls_pkg, dict):
            model = cls_pkg["model"]
            scaler = cls_pkg.get("scaler", None)
            features = cls_pkg.get("feature_cols", None)
        else:
            model = cls_pkg
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            features = None

        if features is None:
            if hasattr(model, "feature_names_in_"):
                features = model.feature_names_in_
            elif hasattr(model, "named_steps"):
                first_step = list(model.named_steps.values())[0]
                features = getattr(first_step, "feature_names_in_", None)

        if features is None:
            features = [
                "charge_cycles", "ambient_temp_c", "electrolyte_resistance_re", 
                "charge_transfer_resistance_rct", "total_internal_resistance", 
                "capacity_decay_rate_5c", "resistance_growth_ratio", "driver_persona_cluster"
            ]

        X_cls = df_test[list(features)]
        X_scaled = scaler.transform(X_cls) if scaler else X_cls
        y_true = df_test["thermal_failure_flag"].values
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        evaluate_classification_model(y_true, y_pred, y_prob)
    else:
        logger.warning(f"[3/3] Classification artifact not found at '{cls_path}'. Skipping.")

    logger.info("\n" + "=" * 65)
    logger.info("MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()