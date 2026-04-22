"""
train_model.py
==============
Phase 3 – Multi-Label Model Training

Trains three multi-label classifiers on the longitudinal patient features:
  1. ClassifierChain (Random Forest)  – captures label correlations
  2. MultiOutputClassifier (Gradient Boosting)  – strong per-label baseline
  3. PyTorch Neural Network with Focal BCE loss  – optimised for high recall

All models use class weighting / focal loss to handle class imbalance
and maximise Sensitivity (Recall), as required for a clinical setting.

Input  : data/processed/diabetic_final.csv
Output : models/chain_rf.pkl
         models/mo_gbm.pkl
         models/nn_model.pth
         models/training_metadata.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_final.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_COLS = ["cardiovascular_complication", "kidney_complication",
              "neuropathy_complication"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].values.astype(float)
    y = df[LABEL_COLS].values.astype(int)

    # Stratify on label combination string
    strat_key = [str(row.tolist()) for row in y]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=strat_key)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

    # Further split train → train + val for NN early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0)

    print(f"[INFO] Dataset shape : {X.shape[0]:,} patients, {X.shape[1]} features")
    print(f"       Train: {X_train.shape[0]:,}   Val: {X_val.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"       Label prevalence (test):")
    for j, col in enumerate(LABEL_COLS):
        print(f"         {col:<35}: {y_test[:,j].mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1 – ClassifierChain (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════

def train_chain_rf(X_train, y_train):
    print("\n[MODEL 1] ClassifierChain – Random Forest …")
    base = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = ClassifierChain(base, order="random", random_state=42, cv=None)
    model.fit(X_train, y_train)

    # Predict probabilities: ClassifierChain returns shape (n, n_labels)
    print("   [✓] Trained.")
    return model


def predict_chain_rf(model, X):
    proba = model.predict_proba(X)      # (n, n_labels) already
    pred  = model.predict(X)
    return proba, pred


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2 – MultiOutputClassifier (Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════════════

def train_mo_gbm(X_train, y_train):
    print("\n[MODEL 2] MultiOutputClassifier – Gradient Boosting …")
    base = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.10,
        max_depth=4,
        subsample=0.85,
        random_state=42,
    )
    model = MultiOutputClassifier(base, n_jobs=1)
    model.fit(X_train, y_train)
    print("   [✓] Trained.")
    return model


def predict_mo_gbm(model, X):
    proba = np.column_stack(
        [est.predict_proba(X)[:, 1] for est in model.estimators_]
    )
    pred  = model.predict(X)
    return proba, pred


# ═══════════════════════════════════════════════════════════════════════════════
# Model 3 – PyTorch Multi-label Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

class _Net(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(64, n_out),
        )
        nn.init.constant_(self.net[-1].bias, 0.1)

    def forward(self, x):
        return self.net(x)


class FocalBCE(nn.Module):
    """Focal loss for class-imbalanced multi-label problems."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pw    = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pw, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def train_nn(X_tr, y_tr, X_val, y_val,
             epochs=70, batch_size=128, lr=3e-4, dropout=0.30,
             gamma=2.0, pos_weight_scale=2.5):
    print("\n[MODEL 3] PyTorch Multi-label Neural Network …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats, n_labels = X_tr.shape[1], y_tr.shape[1]

    net = _Net(n_feats, n_labels, dropout).to(device)

    # Per-label pos_weight from training data
    pos  = y_tr.sum(axis=0) + 1e-6
    neg  = len(y_tr) - pos
    pw   = torch.tensor((neg / pos) * pos_weight_scale, dtype=torch.float32).to(device)

    criterion = FocalBCE(gamma=gamma, pos_weight=pw)
    optim_    = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)

    X_t = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_t = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    best_val, best_state, history = float("inf"), None, {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        net.train()
        loss_sum = 0
        for Xb, yb in dl:
            optim_.zero_grad()
            loss = criterion(net(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim_.step()
            loss_sum += loss.item() * len(Xb)
        tloss = loss_sum / len(X_tr)
        history["train"].append(tloss)

        net.eval()
        with torch.no_grad():
            vloss = criterion(net(X_v), y_v).item()
        history["val"].append(vloss)

        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{epochs}  train={tloss:.4f}  val={vloss:.4f}")

        scheduler.step()

    net.load_state_dict(best_state)
    print("   [✓] Trained.")
    return net, history, device


def predict_nn(net, X, device, threshold=0.50):
    net.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X, dtype=torch.float32).to(device)
        proba = torch.sigmoid(net(X_t)).cpu().numpy()
    pred = (proba >= threshold).astype(int)
    return proba, pred


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  MULTI-LABEL MODEL TRAINING")
    print("="*55)

    (X_train, X_test, y_train, y_test,
     X_tr, X_val, y_tr, y_val,
     feature_cols) = load_data()

    # Train
    chain_rf  = train_chain_rf(X_train, y_train)
    mo_gbm    = train_mo_gbm(X_train, y_train)
    nn_model, nn_history, device = train_nn(X_tr, y_tr, X_val, y_val)

    # Save
    with open(os.path.join(MODELS_DIR, "chain_rf.pkl"), "wb") as f:
        pickle.dump(chain_rf, f)
    with open(os.path.join(MODELS_DIR, "mo_gbm.pkl"), "wb") as f:
        pickle.dump(mo_gbm, f)
    torch.save(nn_model.state_dict(),
               os.path.join(MODELS_DIR, "nn_model.pth"))
    with open(os.path.join(MODELS_DIR, "training_metadata.pkl"), "wb") as f:
        pickle.dump({
            "feature_cols": feature_cols,
            "label_cols"  : LABEL_COLS,
            "n_features"  : X_train.shape[1],
            "nn_history"  : nn_history,
            "X_test"      : X_test,
            "y_test"      : y_test,
        }, f)

    print(f"\n[INFO] All models saved to: {MODELS_DIR}")
    print("[DONE] Training complete.  Run evaluate.py next.")


if __name__ == "__main__":
    main()