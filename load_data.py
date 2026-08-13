import os
import pandas as pd
import numpy as np
# NASA Battery Dataset, Kaggle EV Battery Degradation, or Seattle Open EV Data
import os
import shutil
import glob
import argparse
import kagglehub


def download_nasa_dataset(
    dataset_name: str = "patrickfleith/nasa-battery-dataset",
    output_dir: str = "data/raw",
    target_filename: str = "nasa_battery_cleaned.csv"
) -> str:
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Fetching dataset '{dataset_name}' from Kaggle via kagglehub...")
    # Downloads dataset to local Kagglehub cache directory and returns cache path
    cache_path = kagglehub.dataset_download(dataset_name)
    print(f"[SUCCESS] Cached at: {cache_path}")

    # Search recursively for CSV files in the downloaded cache folder
    csv_files = glob.glob(os.path.join(cache_path, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"[WARNING] No direct CSV files found in cache root. Inspecting all cached files...")
        # Fallback: copy entire contents of cache to target raw directory
        for item in os.listdir(cache_path):
            src_item = os.path.join(cache_path, item)
            dest_item = os.path.join(output_dir, item)
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dest_item)
        final_destination = output_dir
        print(f"[SUCCESS] Copied directory contents to '{output_dir}/'")
    else:
        # Take the main CSV file (or first available CSV)
        primary_csv = csv_files[0]
        final_destination = os.path.join(output_dir, target_filename)

        shutil.copy2(primary_csv, final_destination)
        print(f"[SUCCESS] Staged raw CSV file to '{final_destination}'")

    return final_destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Dataset Direct Downloader")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="patrickfleith/nasa-battery-dataset",
        help="Kaggle dataset identifier"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/raw",
        help="Destination directory for raw data"
    )

    args = parser.parse_args()

    try:
        staged_path = download_nasa_dataset(
            dataset_name=args.dataset, 
            output_dir=args.output_dir
        )
        print(f"\n[READY] Dataset successfully prepared at: {staged_path}")
        print("Run `python src/preprocessing.py` next.")
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch dataset from Kaggle: {e}")