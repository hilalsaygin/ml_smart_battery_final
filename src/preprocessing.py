"""
src/preprocessing.py

Advanced Data Cleaning, Alignment & Feature Engineering Pipeline.
Optimized for RUL (Remaining Useful Life) Prediction on the NASA Battery Dataset.
"""

import os
import re
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit


class NASAPreprocessor:
    """Production preprocessing pipeline for NASA Battery dataset metadata."""

    NOMINAL_CAPACITY_DEFAULT = 2.0  # Ah nominal rating for NASA 18650 cells
    EOL_SOH_THRESHOLD = 70.0        # End-of-Life threshold (% SoH)

    def __init__(self, raw_filepath: str, output_dir: str = "data/processed"):
        self.raw_filepath = raw_filepath
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _parse_matlab_time(t_str: str) -> pd.Timestamp:
        """Parses MATLAB string arrays like '[2.0100e+03 7.00e+00 ...]' into Datetime."""
        if pd.isna(t_str):
            return pd.NaT
        
        # Clean brackets and split by whitespace
        t_str = str(t_str).replace('[', '').replace(']', '').strip()
        parts = re.split(r'\s+', t_str)
        
        if len(parts) >= 6:
            try:
                y, m, d, h, mn, s = [float(p) for p in parts[:6]]
                return pd.Timestamp(
                    year=int(y), month=int(m), day=int(d), 
                    hour=int(h), minute=int(mn), second=int(s),
                    microsecond=int((s % 1) * 1e6)
                )
            except Exception:
                return pd.NaT
        return pd.NaT

    def load_and_clean_raw_metadata(self) -> pd.DataFrame:
        """
        Loads raw metadata, parses time, and cleans alignment issues without look-ahead bias.
        """
        if not os.path.exists(self.raw_filepath):
            raise FileNotFoundError(f"Raw metadata missing at '{self.raw_filepath}'.")

        print(f"[INFO] Loading raw metadata from: {self.raw_filepath}")
        df = pd.read_csv(self.raw_filepath)

        # Standardize column names
        column_map = {
            'type': 'type',
            'start_time': 'start_time',
            'ambient_temperature': 'ambient_temp_c',
            'battery_id': 'battery_id',
            'uid': 'uid',
            'test_id': 'test_id',
            'Capacity': 'capacity_ahr',
            'Re': 'electrolyte_resistance_re',
            'Rct': 'charge_transfer_resistance_rct'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Parse timestamps and ensure numeric types
        if 'start_time' in df.columns:
            df['start_time'] = df['start_time'].apply(self._parse_matlab_time)

        numeric_cols = ['ambient_temp_c', 'capacity_ahr', 'electrolyte_resistance_re', 'charge_transfer_resistance_rct']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort chronologically by battery and time/sequence
        sort_col = 'start_time' if 'start_time' in df.columns else 'uid'
        df = df.sort_values(by=['battery_id', sort_col]).reset_index(drop=True)

        # CLEANING: Forward-fill impedance (Strictly NO backfill to prevent look-ahead bias)
        if 'electrolyte_resistance_re' in df.columns:
            df['electrolyte_resistance_re'] = df.groupby('battery_id')['electrolyte_resistance_re'].ffill()
        if 'charge_transfer_resistance_rct' in df.columns:
            df['charge_transfer_resistance_rct'] = df.groupby('battery_id')['charge_transfer_resistance_rct'].ffill()

        # Filter strictly to 'discharge' cycles
        df_discharge = df[df['type'] == 'discharge'].dropna(subset=['capacity_ahr']).copy()

        # Generate sequential cycle index per battery
        df_discharge['cycle_index'] = df_discharge.groupby('battery_id').cumcount() + 1

        print(f"[SUCCESS] Cleaned data: Retained {len(df_discharge)} discharge cycles.")
        return df_discharge

    def engineer_features_and_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates advanced prognostic features and strict RUL targets.
        """
        df = df.copy()

        # ---------------------------------------------------------
        # TARGET ENGINEERING (SoH & RUL)
        # ---------------------------------------------------------
        # Use the *first* recorded capacity to avoid look-ahead bias from .max()
        initial_capacities = df.groupby('battery_id')['capacity_ahr'].transform('first')
        nominal_ref = np.where(initial_capacities > 0, initial_capacities, self.NOMINAL_CAPACITY_DEFAULT)
        
        df['soh_percentage'] = (df['capacity_ahr'] / nominal_ref) * 100.0
        df['soh_percentage'] = np.clip(df['soh_percentage'], 0.0, 110.0)

        # Vectorized Remaining Useful Life (RUL)
        eol_candidates = np.where(df['soh_percentage'] <= self.EOL_SOH_THRESHOLD, df['cycle_index'], np.nan)
        df['_temp_eol_cycle'] = eol_candidates
        eol_min_cycles = df.groupby('battery_id')['_temp_eol_cycle'].transform('min')
        max_cycles = df.groupby('battery_id')['cycle_index'].transform('max')
        
        eol_cycles = eol_min_cycles.fillna(max_cycles)
        df['rul_cycles'] = np.maximum(0, eol_cycles - df['cycle_index']).astype('int64')
        df.drop(columns=['_temp_eol_cycle'], inplace=True)

        # ---------------------------------------------------------
        # ADVANCED FEATURE ENGINEERING
        # ---------------------------------------------------------
        
        # 1. Temporal & Regeneration Features
        if 'start_time' in df.columns:
            # Time since the last discharge cycle (calendar aging)
            time_diff = df.groupby('battery_id')['start_time'].diff()
            df['rest_time_hours'] = time_diff.dt.total_seconds() / 3600.0
            df['rest_time_hours'] = df['rest_time_hours'].fillna(0.0)
            
            # Cumulative operational time
            df['cumulative_time_hours'] = df.groupby('battery_id')['rest_time_hours'].cumsum()
            
            # Regeneration flag (long rest usually causes temporary capacity recovery)
            # Flag if battery rested for more than 3 hours
            df['long_rest_flag'] = (df['rest_time_hours'] > 3.0).astype(int)
        else:
            df['rest_time_hours'] = 0.0
            df['cumulative_time_hours'] = 0.0
            df['long_rest_flag'] = 0

        # 2. Impedance Total & Growth (with safety for unmeasured first cycles)
        if 'electrolyte_resistance_re' in df.columns and 'charge_transfer_resistance_rct' in df.columns:
            df['total_internal_resistance'] = (
                df['electrolyte_resistance_re'].fillna(0.0) + df['charge_transfer_resistance_rct'].fillna(0.0)
            )
            df['resistance_growth_ratio'] = (
                df.groupby('battery_id')['total_internal_resistance'].pct_change().fillna(0.0)
            )
        else:
            df['total_internal_resistance'] = 0.0
            df['resistance_growth_ratio'] = 0.0

        # 3. Prevent Leakage: Create shifted targets for feature calculations
        # The model can only see the capacity from cycle N-1 to predict RUL at cycle N.
        
        # FIXED LINE HERE:
        shifted_capacity = df.groupby('battery_id')['capacity_ahr'].shift(1).bfill()

        # Capacity Decay Rate (Historical)
        df['capacity_decay_rate_5c'] = (
            df.groupby('battery_id')[shifted_capacity.name].diff(5) / 5.0
        ).fillna(0.0)
        
        # EMA (Exponential Moving Average) to smooth capacity regeneration noise
        df['capacity_ema_5'] = df.groupby('battery_id')[shifted_capacity.name].transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        )
        
        # Rolling volatility (indicates instability near end of life)
        df['capacity_rolling_std_5'] = df.groupby('battery_id')[shifted_capacity.name].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        ).fillna(0.0)

        if 'ambient_temp_c' not in df.columns:
            df['ambient_temp_c'] = 24.0

        print(f"[SUCCESS] Feature Engineering Complete. RUL Mean: {df['rul_cycles'].mean():.1f} cycles")
        return df

    def split_by_battery(
        self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset by Battery ID to eliminate data leakage across sets.
        """
        unique_batteries = df['battery_id'].unique()

        if len(unique_batteries) >= 3:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_val_idx, test_idx = next(gss_test.split(df, groups=df['battery_id']))

            df_train_val = df.iloc[train_val_idx].copy()
            df_test = df.iloc[test_idx].copy()

            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=42)
            train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['battery_id']))

            df_train = df_train_val.iloc[train_idx].copy()
            df_val = df_train_val.iloc[val_idx].copy()
        else:
            # Fallback to temporal splitting per battery if battery count is very low
            train_list, val_list, test_list = [], [], []
            for _, group in df.groupby('battery_id'):
                n = len(group)
                n_test, n_val = int(n * test_size), int(n * val_size)
                train_list.append(group.iloc[: (n - n_test - n_val)])
                val_list.append(group.iloc[(n - n_test - n_val) : (n - n_test)])
                test_list.append(group.iloc[(n - n_test) :])

            df_train = pd.concat(train_list, axis=0)
            df_val = pd.concat(val_list, axis=0)
            df_test = pd.concat(test_list, axis=0)

        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    def scale_and_save(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fits StandardScaler on training data only and exports processed split CSVs.
        """
        valid_cols = [c for c in feature_cols if c in df_train.columns]

        df_train_scaled, df_val_scaled, df_test_scaled = df_train.copy(), df_val.copy(), df_test.copy()

        # Handle any residual NaNs safely before scaling
        for df in [df_train_scaled, df_val_scaled, df_test_scaled]:
            df[valid_cols] = df[valid_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Fit on TRAIN ONLY
        self.scaler.fit(df_train_scaled[valid_cols])

        df_train_scaled[valid_cols] = self.scaler.transform(df_train_scaled[valid_cols])
        df_val_scaled[valid_cols] = self.scaler.transform(df_val_scaled[valid_cols])
        df_test_scaled[valid_cols] = self.scaler.transform(df_test_scaled[valid_cols])

        # Save Scaler artifact
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.joblib"))

        # Save processed CSVs
        df_train_scaled.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        df_val_scaled.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)
        df_test_scaled.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        print(f"[ARTIFACTS] Processed CSVs and scaler saved to '{self.output_dir}/'")
        return {"train": df_train_scaled, "val": df_val_scaled, "test": df_test_scaled}

    def run(self) -> Dict[str, pd.DataFrame]:
        # Include the new advanced prognostic features
        feature_cols = [
            'cycle_index',
            'ambient_temp_c',
            'rest_time_hours',
            'cumulative_time_hours',
            'long_rest_flag',
            'electrolyte_resistance_re',
            'charge_transfer_resistance_rct',
            'total_internal_resistance',
            'resistance_growth_ratio',
            'capacity_decay_rate_5c',
            'capacity_ema_5',
            'capacity_rolling_std_5'
        ]
        
        df_clean = self.load_and_clean_raw_metadata()
        df_engineered = self.engineer_features_and_targets(df_clean)
        df_train, df_val, df_test = self.split_by_battery(df_engineered)
        return self.scale_and_save(df_train, df_val, df_test, feature_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA Metadata Preprocessing Pipeline")
    parser.add_argument("--input_path", type=str, default="data/raw/nasa_battery_cleaned.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")

    args = parser.parse_args()
    pipeline = NASAPreprocessor(raw_filepath=args.input_path, output_dir=args.output_dir)
    pipeline.run()