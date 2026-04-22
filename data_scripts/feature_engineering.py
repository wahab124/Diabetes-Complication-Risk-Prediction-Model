"""
feature_engineering.py  (REVISED)
==================================
Phase 2 – Feature Engineering

Changes from original:
  1. Keeps ALL patient encounters (not just the first) to build
     genuine longitudinal features (slope, delta, std across visits).
  2. Derives three binary complication labels from ICD-9 codes.
  3. Produces a SINGLE ROW per patient with aggregated visit features.
  4. Drops raw diagnosis/encounter columns after extraction.

ICD-9 Ranges:
    Kidney Disease   : 580–589
    Neuropathy       : 354–357
    Cardiovascular   : 390–459

Input  : data/raw/diabetic_data.csv          ← raw file (all encounters)
Output : data/processed/diabetic_final.csv   ← one row per patient
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "raw",       "diabetic_data.csv")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_final.csv")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "scaler_final.pkl")

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "insulin",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def age_midpoint(bracket: str) -> float:
    cleaned = bracket.replace("[","").replace("(","").replace("]","").replace(")","")
    lo, hi  = cleaned.split("-")
    return (int(lo) + int(hi)) / 2


def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float) - np.arange(n, dtype=float).mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x, values - values.mean()) / denom)


# ── Step 1: load & basic clean ─────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)

    # Drop columns with >50% missing or no clinical value
    drop_cols = ["weight", "payer_code", "medical_specialty", "encounter_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Remove invalid gender
    df = df[df["gender"].isin(["Male","Female"])].copy()

    # Fill categoricals
    df["race"]         = df["race"].fillna("Unknown")
    df["max_glu_serum"]= df["max_glu_serum"].fillna("None")
    df["A1Cresult"]    = df["A1Cresult"].fillna("None")
    for col in ["diag_1","diag_2","diag_3"]:
        df[col] = df[col].fillna("0")

    # Numeric conversions
    df["age"]    = df["age"].apply(age_midpoint)
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["change"] = df["change"].map({"Ch": 1, "No": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    for col in MEDICATION_COLS:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)

    print(f"[INFO] Loaded & cleaned: {df.shape[0]:,} rows, "
          f"{df['patient_nbr'].nunique():,} unique patients")
    return df


# ── Step 2: complication labels from ICD-9 ─────────────────────────────────────

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    def classify(row):
        diags = [safe_float(row["diag_1"]),
                 safe_float(row["diag_2"]),
                 safe_float(row["diag_3"])]
        cardio  = int(any(d and 390 <= d <= 459 for d in diags))
        kidney  = int(any(d and 580 <= d <= 589 for d in diags))
        neuro   = int(any(d and 354 <= d <= 357 for d in diags))
        return pd.Series([cardio, kidney, neuro])

    df[["cardiovascular_complication",
        "kidney_complication",
        "neuropathy_complication"]] = df.apply(classify, axis=1)

    labels = ["cardiovascular_complication","kidney_complication","neuropathy_complication"]
    print("[INFO] Label prevalence (encounter-level):")
    for lbl in labels:
        pct = df[lbl].mean()*100
        print(f"       {lbl:<35}: {pct:.1f}%")

    # Patient-level: a patient IS positive if ANY of their encounters is positive
    pat_labels = df.groupby("patient_nbr")[labels].max().reset_index()
    print("\n[INFO] Label prevalence (patient-level, any-visit positive):")
    for lbl in labels:
        pct = pat_labels[lbl].mean()*100
        print(f"       {lbl:<35}: {pct:.1f}%")

    return df, pat_labels


# ── Step 3: longitudinal feature aggregation ────────────────────────────────────

VISIT_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses",
    "number_outpatient", "number_emergency", "number_inpatient",
    "age",
]

STATIC_COLS = ["gender", "diabetesMed"]


def build_longitudinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all encounters for each patient into a single row.
    For each visit-varying numeric column we compute:
        _mean, _std, _min, _max, _last, _delta (last-first), _slope
    """
    rows = []
    for pid, grp in df.groupby("patient_nbr"):
        grp = grp.reset_index(drop=True)          # visit order = encounter order
        row = {"patient_nbr": pid, "num_visits": len(grp)}

        # Longitudinal aggregations
        for col in VISIT_NUMERIC:
            if col not in grp.columns:
                continue
            vals = grp[col].fillna(grp[col].median()).values.astype(float)
            row[f"{col}_mean"]  = vals.mean()
            row[f"{col}_std"]   = vals.std(ddof=0) if len(vals) > 1 else 0.0
            row[f"{col}_min"]   = vals.min()
            row[f"{col}_max"]   = vals.max()
            row[f"{col}_last"]  = vals[-1]
            row[f"{col}_delta"] = float(vals[-1] - vals[0])
            row[f"{col}_slope"] = _slope(vals)

        # Medication usage across visits (mean ordinal)
        for col in MEDICATION_COLS:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()

        # Static columns from last visit
        for col in STATIC_COLS:
            if col in grp.columns:
                row[col] = grp[col].iloc[-1]

        # A1C / glucose: most recent non-None value
        for col in ["max_glu_serum","A1Cresult"]:
            if col in grp.columns:
                last_val = grp[col].replace("None", np.nan).dropna()
                row[col] = last_val.iloc[-1] if len(last_val) else "None"

        rows.append(row)

    feat_df = pd.DataFrame(rows)
    print(f"\n[INFO] Longitudinal feature matrix: {feat_df.shape[0]:,} patients × "
          f"{feat_df.shape[1]} features")
    return feat_df


# ── Step 4: encode categoricals & scale ────────────────────────────────────────

def finalize(feat_df: pd.DataFrame, pat_labels: pd.DataFrame):
    # Merge labels
    df = feat_df.merge(pat_labels, on="patient_nbr", how="inner")

    # One-hot encode remaining categoricals
    cat_cols = [c for c in ["max_glu_serum","A1Cresult"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    df = df.replace({True: 1, False: 0})

    # Scale numeric columns
    label_cols = ["cardiovascular_complication","kidney_complication",
                  "neuropathy_complication","patient_nbr"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in label_cols]

    # Fill any NaN remaining (shouldn't be much)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Drop patient_nbr (not a feature)
    df.drop(columns=["patient_nbr"], inplace=True)

    print(f"[INFO] Final feature matrix: {df.shape} "
          f"(labels are last 3 columns)")
    return df, scaler, num_cols


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  FEATURE ENGINEERING  (Longitudinal)")
    print("="*55)

    raw_df            = load_and_clean(INPUT_PATH)
    raw_df, pat_labels = assign_labels(raw_df)
    feat_df           = build_longitudinal_features(raw_df)
    final_df, scaler, num_cols = finalize(feat_df, pat_labels)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Saved → {OUTPUT_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "numeric_cols": num_cols}, f)
    print(f"[INFO] Saved scaler → {SCALER_PATH}")
    print("\n[DONE] Feature engineering complete.")


if __name__ == "__main__":
    main()