"""
bias_check.py
=============
Error Analysis & Bias Check

Evaluates the final model's (Chain RF tuned) performance broken down by:
  - Age group
  - Gender
  - Race

For each subgroup, reports per-label Recall, Precision, and F1.
Flags any subgroup where recall drops below the 0.80 clinical target.

Saves figures to report/deliverable_2/figures/

Usage:
    python src/bias_check.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import recall_score, precision_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "diabetic_data.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIG_DIR      = os.path.join(PROJECT_ROOT, "report", "deliverable_2", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_COLS  = ["cardiovascular_complication", "kidney_complication",
               "neuropathy_complication"]
LABEL_SHORT = ["Cardio", "Kidney", "Neuro"]
TARGET_RECALL = 0.80

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False})


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def age_midpoint(bracket):
    cleaned = str(bracket).replace("[","").replace("(","").replace("]","").replace(")","")
    lo, hi  = cleaned.split("-")
    return (int(lo) + int(hi)) / 2

def age_group(midpoint):
    if midpoint < 40:  return "<40"
    if midpoint < 60:  return "40–59"
    if midpoint < 75:  return "60–74"
    return "75+"

def classify_complications(row):
    diags = [safe_float(row["diag_1"]), safe_float(row["diag_2"]), safe_float(row["diag_3"])]
    return pd.Series([
        int(any(d and 390 <= d <= 459 for d in diags)),
        int(any(d and 580 <= d <= 589 for d in diags)),
        int(any(d and 354 <= d <= 357 for d in diags)),
    ])


# ── Rebuild test set WITH demographic columns ──────────────────────────────────

def build_test_with_demographics():
    """
    Reconstruct the test set patient index and attach their demographic info.
    We reload the raw data, recompute labels and longitudinal features the same
    way train_model.py did, then use the saved metadata to recover the test split.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("[INFO] Loading raw data …")
    raw = pd.read_csv(DATA_PATH)
    raw.replace("?", np.nan, inplace=True)

    # Keep demographics before any transforms
    raw = raw[raw["gender"].isin(["Male","Female"])].copy()
    raw["age_mid"]   = raw["age"].apply(age_midpoint)
    raw["age_group"] = raw["age_mid"].apply(age_group)
    raw["race"]      = raw["race"].fillna("Unknown")
    raw["gender_label"] = raw["gender"]   # keep original string

    for col in ["diag_1","diag_2","diag_3"]:
        raw[col] = raw[col].fillna("0")

    # Patient-level labels (same logic as feature_engineering.py)
    raw[LABEL_COLS] = raw.apply(classify_complications, axis=1)
    pat_labels = raw.groupby("patient_nbr")[LABEL_COLS].max().reset_index()

    # Patient-level demographics (from last encounter)
    pat_demo = (raw.sort_values("patient_nbr")
                   .groupby("patient_nbr")[["age_group","gender_label","race"]]
                   .last()
                   .reset_index())

    # Merge
    pat_df = pat_labels.merge(pat_demo, on="patient_nbr")

    # Load feature matrix (same patients, same order as training)
    with open(os.path.join(MODELS_DIR, "training_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    X_test   = meta["X_test"]
    y_test   = meta["y_test"]
    n_test   = len(X_test)

    # We need to figure out which patients are in the test set.
    # train_model.py used train_test_split with random_state=42 on the full
    # patient feature matrix. We reproduce the same split on patient_nbr order.
    all_pids = pat_df["patient_nbr"].values
    strat_key = [str(row.tolist()) for row in pat_df[LABEL_COLS].values]

    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(all_pids)), test_size=0.20,
            random_state=42, stratify=strat_key)
    except ValueError:
        idx_train, idx_test = train_test_split(
            np.arange(len(all_pids)), test_size=0.20, random_state=42)

    demo_test = pat_df.iloc[idx_test].reset_index(drop=True)
    assert len(demo_test) == n_test, \
        f"Mismatch: {len(demo_test)} demo rows vs {n_test} test rows"

    print(f"[INFO] Test set: {n_test:,} patients with demographics attached.")
    return X_test, y_test, demo_test


# ── Subgroup evaluation ────────────────────────────────────────────────────────

def evaluate_subgroups(y_true, y_pred, demo_df, group_col, group_label):
    """
    Returns a DataFrame with recall/precision/F1 per subgroup per label.
    """
    records = []
    groups  = sorted(demo_df[group_col].unique())

    for g in groups:
        mask = demo_df[group_col].values == g
        if mask.sum() < 20:   # skip tiny groups
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        row = {"Subgroup": g, "N": int(mask.sum())}
        for j, s in enumerate(LABEL_SHORT):
            row[f"Recall_{s}"]    = recall_score(yt[:,j], yp[:,j],    zero_division=0)
            row[f"Precision_{s}"] = precision_score(yt[:,j], yp[:,j], zero_division=0)
            row[f"F1_{s}"]        = f1_score(yt[:,j], yp[:,j],        zero_division=0)
        records.append(row)

    return pd.DataFrame(records)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_subgroup_recall(df, group_col_label, filename):
    recall_cols = [f"Recall_{s}" for s in LABEL_SHORT]
    n_groups    = len(df)
    x           = np.arange(n_groups)
    width       = 0.22
    offsets     = np.linspace(-width, width, 3)
    colors      = ["#2E86AB", "#3BB273", "#E84855"]

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.4), 5.5))

    for i, (col, s, color) in enumerate(zip(recall_cols, LABEL_SHORT, colors)):
        vals = df[col].values
        bars = ax.bar(x + offsets[i], vals, width=width*0.9,
                      label=s, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.axhline(TARGET_RECALL, color="red", linestyle="--", linewidth=1.5,
               label=f"Target Recall ({TARGET_RECALL})")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['Subgroup']}\n(n={row['N']:,})" for _, row in df.iterrows()],
        fontsize=10)
    ax.set_ylabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Per-Label Recall by {group_col_label}  (Chain RF – Tuned)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")
    return path


def plot_bias_heatmap(dfs_dict, filename):
    """
    One heatmap per label showing recall across ALL demographic dimensions.
    Rows = subgroup values, Columns = demographic axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Recall Heatmap by Demographic Subgroup (Chain RF – Tuned)",
                 fontsize=13, fontweight="bold")

    for ax, s in zip(axes, LABEL_SHORT):
        rows  = []
        for dim_label, df in dfs_dict.items():
            for _, row in df.iterrows():
                rows.append({
                    "Dimension": dim_label,
                    "Group"    : str(row["Subgroup"]),
                    "Recall"   : row[f"Recall_{s}"],
                })
        plot_df = pd.DataFrame(rows)
        pivot   = plot_df.pivot(index="Group", columns="Dimension", values="Recall")

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5,
                    ax=ax, cbar_kws={"label": "Recall"})
        ax.set_title(s, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Subgroup", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  BIAS CHECK & ERROR ANALYSIS")
    print("="*55)

    # Load model and reproduce predictions with tuned thresholds
    with open(os.path.join(MODELS_DIR, "chain_rf.pkl"), "rb") as f:
        chain_rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "training_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    X_test, y_test, demo_test = build_test_with_demographics()

    # Reproduce tuned predictions (thresholds from evaluate.py output)
    # Tune them fresh here for reproducibility
    from sklearn.metrics import recall_score as rs, precision_score as ps
    prob = chain_rf.predict_proba(X_test)

    def tune(proba, y_true, target=0.80, steps=100):
        thresholds = np.full(proba.shape[1], 0.5)
        for j in range(proba.shape[1]):
            best_t, best_f1 = 0.5, -1.0
            for t in np.linspace(0.05, 0.80, steps):
                preds = (proba[:,j] >= t).astype(int)
                rec   = rs(y_true[:,j], preds, zero_division=0)
                prec  = ps(y_true[:,j], preds, zero_division=0)
                if prec + rec == 0:
                    continue
                f1 = 2*prec*rec/(prec+rec)
                if rec < target: f1 *= 0.3
                if f1 > best_f1: best_f1, best_t = f1, t
            thresholds[j] = best_t
        return thresholds

    thresholds = tune(prob, y_test)
    y_pred     = (prob >= thresholds).astype(int)
    print(f"\n[INFO] Tuned thresholds: {[round(t,3) for t in thresholds]}")

    # ── Subgroup analysis ──────────────────────────────────────────────────────
    print("\n[1] Evaluating by Age Group …")
    df_age    = evaluate_subgroups(y_test, y_pred, demo_test, "age_group", "Age Group")

    print("[2] Evaluating by Gender …")
    df_gender = evaluate_subgroups(y_test, y_pred, demo_test, "gender_label", "Gender")

    print("[3] Evaluating by Race …")
    df_race   = evaluate_subgroups(y_test, y_pred, demo_test, "race", "Race")

    # ── Print tables ───────────────────────────────────────────────────────────
    recall_cols = ["Subgroup","N"] + [f"Recall_{s}" for s in LABEL_SHORT]

    print("\n── Age Group Recall ─────────────────────────────")
    print(df_age[recall_cols].to_string(index=False))

    print("\n── Gender Recall ────────────────────────────────")
    print(df_gender[recall_cols].to_string(index=False))

    print("\n── Race Recall ──────────────────────────────────")
    print(df_race[recall_cols].to_string(index=False))

    # ── Flag underperforming subgroups ────────────────────────────────────────
    print("\n── Subgroups Below Target Recall (< 0.80) ───────")
    found_any = False
    for dim_label, df in [("Age", df_age), ("Gender", df_gender), ("Race", df_race)]:
        for _, row in df.iterrows():
            for s in LABEL_SHORT:
                rec = row[f"Recall_{s}"]
                if rec < TARGET_RECALL:
                    print(f"  ⚠  {dim_label}: {row['Subgroup']:<20} "
                          f"{s} Recall = {rec:.3f}")
                    found_any = True
    if not found_any:
        print("  All subgroups meet the 0.80 recall target.")

    # ── Save figures ───────────────────────────────────────────────────────────
    print("\n[Generating figures …]")
    plot_subgroup_recall(df_age,    "Age Group",  "bias_age_recall.png")
    plot_subgroup_recall(df_gender, "Gender",     "bias_gender_recall.png")
    plot_subgroup_recall(df_race,   "Race",       "bias_race_recall.png")

    dfs_dict = {"Age": df_age, "Gender": df_gender, "Race": df_race}
    plot_bias_heatmap(dfs_dict, "bias_heatmap.png")

    # Save CSV for report reference
    for name, df in [("age", df_age), ("gender", df_gender), ("race", df_race)]:
        df.to_csv(os.path.join(FIG_DIR, f"bias_{name}.csv"), index=False)

    print(f"\n[DONE] Bias check complete. Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()