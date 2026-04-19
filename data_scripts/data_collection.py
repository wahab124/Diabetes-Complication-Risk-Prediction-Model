"""
data_collection.py
==================
Phase 2 – Data Collection
Downloads the UCI Diabetes 130-US Hospitals dataset and performs an initial overview.

Dataset : UCI ML Repository – Diabetes 130-US Hospitals (1999-2008)
URL     : https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Usage:
    python data_scripts/data_collection.py

Output:
    data/raw/diabetic_data.csv
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

# ── Paths (relative to project root) ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
ZIP_PATH     = os.path.join(RAW_DIR, "diabetes_raw.zip")
OUTPUT_PATH  = os.path.join(RAW_DIR, "diabetic_data.csv")

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


def download_dataset():
    """Download and extract the dataset from UCI if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Dataset already exists at: {OUTPUT_PATH}")
        return

    print(f"[INFO] Downloading dataset from:\n       {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print(f"[INFO] Download complete.")

    print("[INFO] Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
    os.remove(ZIP_PATH)
    print(f"[INFO] Extracted to: {RAW_DIR}")


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and replace '?' sentinel values with NaN."""
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    return df


def overview(df: pd.DataFrame):
    """Print a structured summary of the raw dataset."""
    print("\n── Shape ────────────────────────────────")
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n── Columns ──────────────────────────────")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")

    print("\n── Data Types ───────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values (%) ───────────────────")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print(missing_pct.to_string() if not missing_pct.empty else "  None")

    print("\n── First 5 Rows ─────────────────────────")
    print(df.head())


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved: {path}")


if __name__ == "__main__":
    download_dataset()
    df = load_dataset(OUTPUT_PATH)
    overview(df)
    # Save a copy with NaN already substituted so downstream scripts don't re-do it
    save_data(df, OUTPUT_PATH)
    print("\n[DONE] Data collection complete.")
