"""
src/clustering.py

Unsupervised Driver & Charging Risk Profiling Module.
Performs K-Means behavioral clustering on driver telemetry to discover operational
risk profiles (e.g., Eco Driver vs. High Stress / Thermal Aggressive Driver).

Outputs:
- Enhanced DataFrame with 'driver_persona_cluster' column.
- Optimal K diagnostic plot (Elbow Curve & Silhouette Scores).
- Serialized model artifact: 'models/kmeans_model.joblib'.
- Serialized feature scaler artifact: 'models/scaler_clustering.joblib'.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default Feature Schema for Driver Profiling
CLUSTERING_FEATURES: List[str] = [
    "rapid_charge_ratio",
    "avg_speed_kmh",
    "charge_cycles",
    "thermal_stress_index"
]


def sanitize_and_scale_features(
    df: pd.DataFrame,
    features: List[str] = CLUSTERING_FEATURES,
    scaler_path: str = "models/scaler_clustering.joblib"
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Extracts, sanitizes (handles NaN/inf), and scales clustering features using StandardScaler.
    Saves the scaler artifact using joblib for downstream inference.

    Args:
        df: Input DataFrame containing telemetry features.
        features: List of feature column names to scale.
        scaler_path: File path to export fitted scaler.

    Returns:
        Tuple containing scaled feature array (np.ndarray) and fitted StandardScaler.
    """
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required clustering features in DataFrame: {missing_cols}")

    logger.info(f"Sanitizing and scaling {len(features)} clustering features: {features}")
    
    # Extract copy and handle infinities/NaNs to prevent scaling crashes
    X_raw = df[features].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values with column medians for robust center estimation
    if X_raw.isna().sum().sum() > 0:
        logger.warning("NaN or Infinite values detected in clustering features. Imputing with column medians.")
        X_raw = X_raw.fillna(X_raw.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Save fitted scaler using joblib
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved clustering feature scaler to '{scaler_path}'")

    return X_scaled, scaler


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 8),
    plot_output_path: str = "reports/figures/kmeans_optimal_k.png"
) -> Tuple[int, Dict[str, List[float]]]:
    """
    Calculates Inertia (Elbow Method) and Silhouette Scores for a range of K.
    Generates diagnostic visual plots and automatically selects the optimal K 
    based on the maximum Silhouette score.

    Args:
        X_scaled: Pre-scaled numerical feature array.
        k_range: Iterable range of K cluster values to evaluate (default: 2 to 7).
        plot_output_path: Path to export the evaluation plots.

    Returns:
        Tuple containing best_k (int) and dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating optimal K across range {list(k_range)}...")
    
    inertias: List[float] = []
    silhouette_scores: List[float] = []
    k_values: List[int] = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init="k-means++")
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        # Subsample for silhouette score if dataset is very large
        sample_size = min(len(X_scaled), 5000)
        sil_score = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=42)
        
        inertias.append(float(inertia))
        silhouette_scores.append(float(sil_score))
        
        logger.info(f"K={k} | Inertia: {inertia:10.2f} | Silhouette Score: {sil_score:.4f}")

    # Determine best K based on max silhouette score
    best_k_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_k_idx]
    logger.info(f"Optimal K selected: {best_k} (Silhouette Score: {silhouette_scores[best_k_idx]:.4f})")

    # Plotting Evaluation Metrics
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Elbow Method)', color=color)
    ax1.plot(k_values, inertias, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, silhouette_scores, marker='s', color=color, linewidth=2, linestyle='--', label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'K-Means Optimization (Selected Best K = {best_k})')
    fig.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    plt.close()

    logger.info(f"Optimal K diagnostic plot saved to '{plot_output_path}'")

    metrics = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores
    }
    
    return best_k, metrics


def train_kmeans_clustering(
    X_scaled: np.ndarray,
    df: pd.DataFrame,
    features: List[str] = CLUSTERING_FEATURES,
    n_clusters: int = 3,
    model_output_path: str = "models/kmeans_model.joblib"
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Fits K-Means clustering model, assigns cluster labels to the DataFrame,
    prints dynamic cluster profile statistics, and serializes the trained model.

    Args:
        X_scaled: Pre-scaled numerical feature array.
        df: Input original DataFrame to augment with cluster labels.
        features: List of feature names used for profiling.
        n_clusters: Number of clusters to create.
        model_output_path: File path to save the fitted KMeans model.

    Returns:
        Tuple containing enhanced DataFrame with 'driver_persona_cluster' column
        and fitted KMeans model instance.
    """
    logger.info(f"Fitting KMeans model with n_clusters={n_clusters} (n_init=10, random_state=42)...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init="k-means++")
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Attach cluster assignments to DataFrame
    enhanced_df = df.copy()
    enhanced_df["driver_persona_cluster"] = cluster_labels

    # Compute and Print Cluster Behavioral Profiles Dynamically
    logger.info("=" * 60)
    logger.info("       DRIVER BEHAVIORAL CLUSTER PROFILES        ")
    logger.info("=" * 60)
    
    profile_df = enhanced_df.groupby("driver_persona_cluster")[features].mean()
    cluster_counts = enhanced_df["driver_persona_cluster"].value_counts().sort_index()
    profile_df["sample_count"] = cluster_counts
    
    logger.info("\n" + profile_df.to_string())

    # Dynamically log mean feature values without hardcoding feature names
    for cluster_id in range(n_clusters):
        means = profile_df.loc[cluster_id]
        feature_summary = " | ".join([f"{feat}: {means[feat]:.2f}" for feat in features])
        logger.info(f"Cluster {cluster_id} Profile -> {feature_summary} | Count: {int(means['sample_count'])}")

    # Save Trained KMeans Artifact using joblib
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(kmeans, model_output_path)
    logger.info(f"Saved trained KMeans model artifact to '{model_output_path}'")

    return enhanced_df, kmeans


def run_driver_clustering_pipeline(
    df: pd.DataFrame,
    features: List[str] = CLUSTERING_FEATURES,
    n_clusters: Optional[int] = None,
    auto_find_k: bool = True
) -> pd.DataFrame:
    """
    Orchestration wrapper to execute feature sanitization, scaling, optional optimal K discovery,
    model training, dynamic profiling, and DataFrame enhancement.

    Args:
        df: Raw or preprocessed driver telemetry DataFrame.
        features: Features to use for clustering.
        n_clusters: Targeted cluster count. If None and auto_find_k is True, uses best_k.
        auto_find_k: Whether to compute optimal K dynamically via Silhouette score.

    Returns:
        Enhanced DataFrame containing 'driver_persona_cluster' column.
    """
    logger.info("Starting Driver & Charging Risk Profiling Pipeline...")
    
    # Step 1: Sanitize and Scale Features
    X_scaled, _ = sanitize_and_scale_features(df, features=features)

    # Step 2: Optimal K Analysis
    if auto_find_k or n_clusters is None:
        best_k, _ = find_optimal_k(X_scaled)
        if n_clusters is None:
            n_clusters = best_k
            logger.info(f"Automatically adopting optimal n_clusters={n_clusters}")

    # Fallback default if n_clusters was neither supplied nor computed
    if n_clusters is None:
        n_clusters = 3

    # Step 3: Train Model & Annotate Data
    enhanced_df, _ = train_kmeans_clustering(
        X_scaled=X_scaled,
        df=df,
        features=features,
        n_clusters=n_clusters
    )

    logger.info("Clustering pipeline completed successfully.")
    return enhanced_df


if __name__ == "__main__":
    # Standalone Demonstration Execution
    print("[TEST RUN] Executing Clustering Module with Synthetic Telemetry...")
    np.random.seed(42)
    sample_size = 500

    # Generate Synthetic Driver Telemetry with noise and edge cases
    synthetic_data = pd.DataFrame({
        "rapid_charge_ratio": np.random.uniform(0.0, 0.9, sample_size),
        "avg_speed_kmh": np.random.normal(55, 15, sample_size),
        "charge_cycles": np.random.randint(20, 400, sample_size),
        "thermal_stress_index": np.random.uniform(0.1, 0.95, sample_size)
    })

    # Execute Pipeline
    processed_df = run_driver_clustering_pipeline(
        df=synthetic_data,
        auto_find_k=True
    )
    
    print("\nSample Output DataFrame:")
    print(processed_df[["rapid_charge_ratio", "thermal_stress_index", "driver_persona_cluster"]].head())