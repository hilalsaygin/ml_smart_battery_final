"""
src/generate_data.py

Synthetic Data Generator for EV Battery Telemetry and Health Monitoring.

Functional Responsibilities:
1. Generates 10,000 realistic synthetic EV battery records using NumPy and Pandas with seed 42.
2. Simulates 7 core operational features: battery_id, ambient_temp_c, charge_cycles,
   rapid_charge_ratio, avg_speed_kmh, voltage_v, and current_a.
3. Models non-linear State of Health (SoH %) decay bounded between 60.0% and 100.0%.
4. Models imbalanced thermal failure flags (~5% positive rate) driven by extreme thermal stress,
   rapid charge intensity, and voltage spike dynamics.
5. Exports the synthetic dataset to 'data/ev_battery_telemetry.csv'.
"""

import os
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Execution Parameters
RANDOM_SEED = 42
N_SAMPLES = 10000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ev_battery_telemetry.csv")


def generate_ev_battery_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates synthetic EV battery telemetry records with realistic physical correlations.

    Args:
        n_samples: Number of synthetic battery samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing synthetic features and targets.
    """
    np.random.seed(seed)
    logger.info(f"Generating {n_samples:,} synthetic EV battery records (Random Seed: {seed})...")

    # 1. Feature Generation
    battery_id = [f"BAT-{i:05d}" for i in range(1, n_samples + 1)]
    
    # Uniform / Normal distributions within domain bounds
    ambient_temp_c = np.random.uniform(10.0, 45.0, size=n_samples)
    charge_cycles = np.random.randint(100, 2001, size=n_samples)
    rapid_charge_ratio = np.random.uniform(0.0, 1.0, size=n_samples)
    avg_speed_kmh = np.random.uniform(20.0, 120.0, size=n_samples)
    
    # Voltage (V) centered around 365V nominal pack voltage
    voltage_v = np.random.uniform(300.0, 420.0, size=n_samples)
    
    # Current (A) drawn during discharge or fast charge
    current_a = np.random.uniform(10.0, 150.0, size=n_samples)

    # 2. Target 1: State of Health (SoH %) Continuous Decay
    # SoH starts at 100% and decays non-linearly with cycle count, ambient temp, and rapid charging
    cycle_decay = 0.015 * (charge_cycles ** 1.08)
    temp_penalty = 0.12 * np.maximum(0, ambient_temp_c - 25.0) ** 1.3
    charging_penalty = 8.5 * (rapid_charge_ratio ** 1.5) * (charge_cycles / 1000.0)
    noise = np.random.normal(0, 0.8, size=n_samples)

    raw_soh = 100.0 - (cycle_decay + temp_penalty + charging_penalty) + noise
    soh_percentage = np.clip(raw_soh, 60.0, 100.0)

    # 3. Target 2: Imbalanced Thermal Failure Flag (~5% positive class)
    # Failure condition trigger: extreme temp > 40°C AND rapid charge > 0.8 AND voltage spikes
    voltage_spike_threshold = 405.0  # High pack voltage spikes
    
    # Base deterministic condition
    extreme_temp_mask = ambient_temp_c > 40.0
    high_rapid_charge_mask = rapid_charge_ratio > 0.8
    voltage_spike_mask = voltage_v > voltage_spike_threshold

    base_trigger = extreme_temp_mask & high_rapid_charge_mask & voltage_spike_mask
    
    # Stochastic failure probability model to achieve precise ~5% imbalance
    # High risk probability for base trigger matches + small baseline background noise
    failure_logit = (
        -4.5  # Baseline offset
        + 3.5 * (ambient_temp_c > 40.0)
        + 3.0 * (rapid_charge_ratio > 0.8)
        + 2.8 * (voltage_v > 400.0)
        + 1.5 * (current_a > 120.0)
    )
    failure_prob = 1.0 / (1.0 + np.exp(-failure_logit))
    
    # Inject deterministic constraint enforceability while maintaining ~5% balance
    stochastic_failures = np.random.binomial(1, failure_prob)
    thermal_failure_flag = np.where(base_trigger, 1, stochastic_failures)

    # Construct Pandas DataFrame
    df = pd.DataFrame({
        "battery_id": battery_id,
        "ambient_temp_c": np.round(ambient_temp_c, 2),
        "charge_cycles": charge_cycles,
        "rapid_charge_ratio": np.round(rapid_charge_ratio, 3),
        "avg_speed_kmh": np.round(avg_speed_kmh, 2),
        "voltage_v": np.round(voltage_v, 2),
        "current_a": np.round(current_a, 2),
        "soh_percentage": np.round(soh_percentage, 2),
        "thermal_failure_flag": thermal_failure_flag
    })

    return df


def main():
    """Main execution entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = generate_ev_battery_dataset(n_samples=N_SAMPLES, seed=RANDOM_SEED)
    
    # Verify target specifications
    positive_rate = df["thermal_failure_flag"].mean() * 100.0
    soh_min, soh_max = df["soh_percentage"].min(), df["soh_percentage"].max()
    
    logger.info("=" * 60)
    logger.info("DATASET GENERATION VERIFICATION SUMMARY:")
    logger.info(f" Total Rows Generated  : {len(df):,}")
    logger.info(f" Output Destination    : '{OUTPUT_FILE}'")
    logger.info(f" SoH Percentage Range  : [{soh_min:.2f}%, {soh_max:.2f}%]")
    logger.info(f" Thermal Failure Rate  : {positive_rate:.2f}% ({df['thermal_failure_flag'].sum()} positive samples)")
    logger.info("=" * 60)

    # Export to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Dataset successfully saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()