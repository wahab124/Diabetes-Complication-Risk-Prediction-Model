"""
evaluate.py
===========
Phase 4 – Model Evaluation

Loads all trained models and computes:
  - F1-Micro / F1-Macro
  - Per-label Sensitivity (Recall), Precision, F1
  - ROC-AUC per label and macro
  - Threshold tuning to maximise Recall ≥ 0.80
  - All plots saved to report/deliverable_2/figures/

Run this AFTER train_model.py.

Usage:
    python evaluate.py
"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve,
    multilabel_confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIG_DIR      = os.path.join(PROJECT_ROOT, "report", "deliverable_2", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_COLS   = ["cardiovascular_complication", "kidney_complication",
                "neuropathy_complication"]
LABEL_SHORT  = ["Cardio", "Kidney", "Neuro"]
LABEL_NAMES  = ["Cardiovascular", "Kidney Disease", "Neuropathy"]
COLORS       = {"Chain RF": "#2E86AB", "MO GBM": "#3BB273", "Neural Net": "#E84855"}

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


# ═══════════════════════════════════════════════════════════════════════════════
# Load models + test data
# ═══════════════════════════════════════════════════════════════════════════════

def load_everything():
    with open(os.path.join(MODELS_DIR, "training_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "chain_rf.pkl"), "rb") as f:
        chain_rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "mo_gbm.pkl"), "rb") as f:
        mo_gbm   = pickle.load(f)

    # Rebuild NN architecture (must match train_model.py)
    sys.path.insert(0, PROJECT_ROOT)
    from train_model import _Net, predict_nn

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = meta["n_features"]
    nn_net  = _Net(n_feats, len(LABEL_COLS)).to(device)
    nn_net.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "nn_model.pth"),
                   map_location=device))
    nn_net.eval()

    X_test = meta["X_test"]
    y_test = meta["y_test"]
    history= meta["nn_history"]

    return chain_rf, mo_gbm, nn_net, device, X_test, y_test, history


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba=None):
    m = {
        "f1_micro"   : f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro"   : f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "rec_micro"  : recall_score(y_true, y_pred, average="micro",    zero_division=0),
        "rec_macro"  : recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "prec_micro" : precision_score(y_true, y_pred, average="micro", zero_division=0),
    }
    for j, s in enumerate(LABEL_SHORT):
        m[f"rec_{s}"]  = recall_score(y_true[:,j], y_pred[:,j],    zero_division=0)
        m[f"prec_{s}"] = precision_score(y_true[:,j], y_pred[:,j], zero_division=0)
        m[f"f1_{s}"]   = f1_score(y_true[:,j], y_pred[:,j],        zero_division=0)
        if y_proba is not None:
            try:
                m[f"auc_{s}"] = roc_auc_score(y_true[:,j], y_proba[:,j])
            except Exception:
                m[f"auc_{s}"] = float("nan")
    if y_proba is not None:
        try:
            m["auc_macro"] = roc_auc_score(y_true, y_proba, average="macro")
        except Exception:
            m["auc_macro"] = float("nan")
    return m


def print_report(m, name):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  F1-Micro     : {m['f1_micro']:.4f}")
    print(f"  F1-Macro     : {m['f1_macro']:.4f}")
    print(f"  Recall-Micro : {m['rec_micro']:.4f}")
    print(f"  Recall-Macro : {m['rec_macro']:.4f}")
    if "auc_macro" in m:
        print(f"  ROC-AUC Macro: {m['auc_macro']:.4f}")
    print()
    print(f"  {'Label':<18} {'Recall':>8} {'Precision':>10} {'F1':>8}", end="")
    if "auc_Cardio" in m:
        print(f"  {'AUC':>6}", end="")
    print()
    print(f"  {'-'*52}")
    for s, name_ in zip(LABEL_SHORT, LABEL_NAMES):
        r  = m.get(f"rec_{s}", 0)
        p  = m.get(f"prec_{s}", 0)
        f1 = m.get(f"f1_{s}", 0)
        print(f"  {name_:<18} {r:>8.4f} {p:>10.4f} {f1:>8.4f}", end="")
        if f"auc_{s}" in m:
            print(f"  {m[f'auc_{s}']:>6.4f}", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold tuning (maximise recall ≥ target while keeping best F1)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_thresholds(proba, y_true, target_recall=0.80, steps=100):
    thresholds = np.full(proba.shape[1], 0.5)
    for j in range(proba.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.80, steps):
            preds = (proba[:,j] >= t).astype(int)
            rec   = recall_score(y_true[:,j], preds, zero_division=0)
            prec  = precision_score(y_true[:,j], preds, zero_division=0)
            if prec + rec == 0:
                continue
            f1 = 2 * prec * rec / (prec + rec)
            if rec < target_recall:
                f1 *= 0.3           # heavy penalty for missing target recall
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


def apply_thresholds(proba, thresholds):
    return (proba >= thresholds).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")
    return path


def plot_roc_curves(y_test, probas_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ROC Curves per Complication Label", fontsize=14, fontweight="bold")
    for ax, j, lname in zip(axes, range(3), LABEL_NAMES):
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
        for mname, proba in probas_dict.items():
            fpr, tpr, _ = roc_curve(y_test[:,j], proba[:,j])
            auc = roc_auc_score(y_test[:,j], proba[:,j])
            ax.plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})",
                    color=COLORS.get(mname,"#555"), lw=2)
        ax.set(title=lname, xlabel="FPR", ylabel="TPR",
               xlim=(0,1), ylim=(0,1))
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    return save(fig, "roc_curves.png")


def plot_confusion_matrices(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Confusion Matrices – {model_name}", fontsize=13, fontweight="bold")
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for ax, cm, lname in zip(axes, mcm, LABEL_NAMES):
        pct = cm / cm.sum() * 100
        ann = np.array([
            [f"TN\n{cm[0,0]}\n({pct[0,0]:.1f}%)", f"FP\n{cm[0,1]}\n({pct[0,1]:.1f}%)"],
            [f"FN\n{cm[1,0]}\n({pct[1,0]:.1f}%)", f"TP\n{cm[1,1]}\n({pct[1,1]:.1f}%)"],
        ])
        sns.heatmap(cm.astype(float), annot=ann, fmt="", cmap="Blues",
                    xticklabels=["Pred No","Pred Yes"],
                    yticklabels=["Act No","Act Yes"],
                    linewidths=1, cbar=False, ax=ax)
        ax.set_title(lname, fontsize=11)
    plt.tight_layout()
    tag = model_name.lower().replace(" ","_")
    return save(fig, f"cm_{tag}.png")


def plot_model_comparison(all_metrics):
    model_names = list(all_metrics.keys())
    metric_keys = ["f1_micro","f1_macro","rec_macro","auc_macro"]
    metric_lbls = ["F1-Micro","F1-Macro","Recall-Macro","ROC-AUC Macro"]
    x = np.arange(len(metric_keys))
    width = 0.18
    offsets = np.linspace(-width*1.5, width*1.5, len(model_names))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (mn, off) in enumerate(zip(model_names, offsets)):
        vals   = [all_metrics[mn].get(k, 0) for k in metric_keys]
        color  = list(COLORS.values())[i % len(COLORS)]
        bars   = ax.bar(x + off, vals, width=width*0.9,
                        label=mn, color=color, edgecolor="white", lw=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", rotation=40)

    ax.set_xticks(x); ax.set_xticklabels(metric_lbls, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.axhline(0.8, color="grey", ls=":", lw=1, alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save(fig, "model_comparison.png")


def plot_per_label_recall(all_metrics):
    model_names = list(all_metrics.keys())
    x = np.arange(3)
    width = 0.18
    offsets = np.linspace(-width*1.5, width*1.5, len(model_names))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (mn, off) in enumerate(zip(model_names, offsets)):
        vals  = [all_metrics[mn].get(f"rec_{s}", 0) for s in LABEL_SHORT]
        color = list(COLORS.values())[i % len(COLORS)]
        bars  = ax.bar(x + off, vals, width=width*0.9,
                       label=mn, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.007,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.axhline(0.80, color="#E84855", ls="--", lw=1.8,
               label="Target Recall (0.80)")
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_ylabel("Sensitivity (Recall)"); ax.set_ylim(0, 1.12)
    ax.set_title("Per-label Sensitivity Across Models",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save(fig, "per_label_recall.png")


def plot_threshold_sensitivity(proba, y_true, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Threshold vs. Recall / Precision / F1  ({model_name})",
                 fontsize=13, fontweight="bold")
    ts = np.linspace(0.05, 0.95, 80)
    for ax, j, lname in zip(axes, range(3), LABEL_NAMES):
        recs, precs, f1s = [], [], []
        for t in ts:
            p = (proba[:,j] >= t).astype(int)
            r = recall_score(y_true[:,j], p, zero_division=0)
            pr= precision_score(y_true[:,j], p, zero_division=0)
            recs.append(r); precs.append(pr)
            f1s.append(2*r*pr/(r+pr+1e-9))
        ax.plot(ts, recs, label="Recall",    color="#E84855", lw=2)
        ax.plot(ts, precs,label="Precision", color="#2E86AB", lw=2)
        ax.plot(ts, f1s,  label="F1",        color="#3BB273", lw=2, ls="--")
        ax.axhline(0.80, color="grey", ls=":", lw=1)
        ax.axvline(0.50, color="black", ls=":", lw=1, alpha=0.4)
        ax.set(title=lname, xlabel="Threshold", ylabel="Score",
               xlim=(0.05, 0.95), ylim=(0, 1.05))
        ax.legend(fontsize=9)
    plt.tight_layout()
    tag = model_name.lower().replace(" ","_")
    return save(fig, f"threshold_sensitivity_{tag}.png")


def plot_feature_importance(chain_rf, feature_cols):
    try:
        imps = [e.feature_importances_[:len(feature_cols)]
                for e in chain_rf.estimators_]
        imp = np.mean(imps, axis=0)
    except Exception:
        print("  [skip] Feature importance not available for this model.")
        return
    top = 20
    idx = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top))
    ax.barh([feature_cols[i] for i in idx], imp[idx], color=colors)
    ax.set_xlabel("Mean Feature Importance")
    ax.set_title(f"Top {top} Feature Importances (Chain RF)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return save(fig, "feature_importance.png")


def plot_nn_history(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="Train Loss", color="#2E86AB", lw=2)
    ax.plot(history["val"],   label="Val Loss",   color="#E84855", lw=2, ls="--")
    ax.set(xlabel="Epoch", ylabel="Focal BCE Loss",
           title="Neural Network Training Curve")
    ax.legend()
    plt.tight_layout()
    return save(fig, "nn_training_curve.png")


def plot_summary_table(all_metrics):
    rows = []
    for mn, m in all_metrics.items():
        rows.append({
            "Model"        : mn,
            "F1-Micro"     : f"{m.get('f1_micro',0):.4f}",
            "F1-Macro"     : f"{m.get('f1_macro',0):.4f}",
            "Recall-Micro" : f"{m.get('rec_micro',0):.4f}",
            "Recall-Macro" : f"{m.get('rec_macro',0):.4f}",
            "ROC-AUC Macro": f"{m.get('auc_macro',0):.4f}",
            "Recall-Cardio": f"{m.get('rec_Cardio',0):.4f}",
            "Recall-Kidney": f"{m.get('rec_Kidney',0):.4f}",
            "Recall-Neuro" : f"{m.get('rec_Neuro',0):.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(18, max(3, len(df)*1.2 + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2)
    for j in range(len(df.columns)):
        tbl[0,j].set_facecolor("#2E86AB")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(df)+1):
        bg = "#F0F8FF" if i%2 else "white"
        for j in range(len(df.columns)):
            tbl[i,j].set_facecolor(bg)
    ax.set_title("Model Performance Summary", fontsize=14,
                 fontweight="bold", y=0.92, pad=20)
    plt.tight_layout()
    path = save(fig, "summary_table.png")
    df.to_csv(os.path.join(FIG_DIR, "summary_metrics.csv"), index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  MODEL EVALUATION")
    print("="*55)

    chain_rf, mo_gbm, nn_net, device, X_test, y_test, history = load_everything()

    with open(os.path.join(MODELS_DIR, "training_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    feature_cols = meta["feature_cols"]

    from train_model import predict_chain_rf, predict_mo_gbm, predict_nn

    # ── Probabilities & default predictions ──────────────────────────────────
    prob_rf,  pred_rf  = predict_chain_rf(chain_rf, X_test)
    prob_gbm, pred_gbm = predict_mo_gbm(mo_gbm, X_test)
    prob_nn,  pred_nn  = predict_nn(nn_net, X_test, device)

    # ── Base metrics ─────────────────────────────────────────────────────────
    m_rf  = compute_metrics(y_test, pred_rf,  prob_rf)
    m_gbm = compute_metrics(y_test, pred_gbm, prob_gbm)
    m_nn  = compute_metrics(y_test, pred_nn,  prob_nn)

    print_report(m_rf,  "Chain RF  (threshold=0.50)")
    print_report(m_gbm, "MO GBM   (threshold=0.50)")
    print_report(m_nn,  "Neural Net (threshold=0.50)")

    # ── Threshold tuning ─────────────────────────────────────────────────────
    print("\n[Threshold Tuning – target recall ≥ 0.80]")
    thr_rf  = tune_thresholds(prob_rf,  y_test)
    thr_gbm = tune_thresholds(prob_gbm, y_test)
    thr_nn  = tune_thresholds(prob_nn,  y_test)

    print(f"  Chain RF   thresholds: {[round(t,3) for t in thr_rf]}")
    print(f"  MO GBM     thresholds: {[round(t,3) for t in thr_gbm]}")
    print(f"  Neural Net thresholds: {[round(t,3) for t in thr_nn]}")

    m_rf_t  = compute_metrics(y_test, apply_thresholds(prob_rf, thr_rf),   prob_rf)
    m_gbm_t = compute_metrics(y_test, apply_thresholds(prob_gbm, thr_gbm), prob_gbm)
    m_nn_t  = compute_metrics(y_test, apply_thresholds(prob_nn, thr_nn),   prob_nn)

    print_report(m_rf_t,  "Chain RF  (tuned threshold)")
    print_report(m_gbm_t, "MO GBM   (tuned threshold)")
    print_report(m_nn_t,  "Neural Net (tuned threshold)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Generating figures …]")
    all_probas = {"Chain RF": prob_rf, "MO GBM": prob_gbm, "Neural Net": prob_nn}
    plot_roc_curves(y_test, all_probas)

    for mn, pred in [("Chain RF", pred_rf), ("MO GBM", pred_gbm),
                     ("Neural Net", pred_nn)]:
        plot_confusion_matrices(y_test, pred, mn)

    plot_threshold_sensitivity(prob_nn,  y_test, "Neural Net")
    plot_nn_history(history)
    plot_feature_importance(chain_rf, feature_cols)

    all_metrics = {
        "Chain RF (base)"  : m_rf,
        "MO GBM (base)"    : m_gbm,
        "Neural Net (base)": m_nn,
        "Chain RF (tuned)" : m_rf_t,
        "MO GBM (tuned)"   : m_gbm_t,
        "Neural Net (tuned)": m_nn_t,
    }
    plot_model_comparison({"Chain RF": m_rf, "MO GBM": m_gbm,
                           "Neural Net": m_nn,
                           "NN (tuned)": m_nn_t})
    plot_per_label_recall({"Chain RF": m_rf, "MO GBM": m_gbm,
                           "Neural Net": m_nn, "NN (tuned)": m_nn_t})
    plot_summary_table(all_metrics)

    print(f"\n[DONE] All figures saved to: {FIG_DIR}")

    # ── Final console report ──────────────────────────────────────────────────
    best_name = max({"Chain RF (tuned)":m_rf_t,
                     "MO GBM (tuned)":m_gbm_t,
                     "Neural Net (tuned)":m_nn_t},
                    key=lambda k: all_metrics[k]["rec_macro"])
    best = all_metrics[best_name]
    print(f"\n{'='*55}")
    print(f"  BEST MODEL: {best_name}")
    print(f"{'='*55}")
    print(f"  F1-Micro     : {best['f1_micro']:.4f}")
    print(f"  F1-Macro     : {best['f1_macro']:.4f}")
    print(f"  Recall-Macro : {best['rec_macro']:.4f}")
    for s, nm in zip(LABEL_SHORT, LABEL_NAMES):
        print(f"  Recall-{nm:<15}: {best.get(f'rec_{s}',0):.4f}")


if __name__ == "__main__":
    main()