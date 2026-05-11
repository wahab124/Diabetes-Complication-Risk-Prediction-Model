# AI-Based Multi-Label Prediction of Diabetes Complications

## Live Demo
**Deployed Application:** https://huggingface.co/spaces/samishah2004/dcrp

## Project Overview

This project develops an AI-based predictive analytics system that estimates the risk of multiple diabetes-related complications using patient clinical data. The model uses machine learning techniques to predict the likelihood of complications such as kidney disease, neuropathy, and cardiovascular conditions.

The system performs **multi-label classification**, allowing it to predict multiple complications simultaneously for a single patient. By incorporating **longitudinal clinical data** (information from multiple patient visits), the model aims to capture disease progression and improve prediction accuracy.

---

## Problem Statement

Diabetes mellitus is a chronic condition that can lead to severe long-term complications if not properly managed. Traditional medical assessment methods often rely on static thresholds or individual clinical measurements, which may fail to capture complex interactions between multiple health indicators over time.

This project addresses this by building a machine learning model capable of analyzing clinical patterns and predicting the risk of multiple complications simultaneously.

---

## Objectives

- Develop a **multi-label classification model** to predict multiple diabetes complications at once.
- Incorporate **longitudinal patient data** to analyze changes in clinical indicators across multiple visits.
- Optimize the model for **high sensitivity (recall)** to minimize missed complication cases.
- Evaluate model performance using **F1-Micro, F1-Macro, Recall, and ROC-AUC**.
- Provide interpretable insights to support early medical intervention.

---

## Dataset

**Source:** [UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Raw datasets are not uploaded to this repository due to file size limitations. Run `data_scripts/data_collection.py` to download automatically.

**Target Labels (derived from ICD-9 codes):**

| Label | ICD-9 Range | Prevalence |
|---|---|---|
| Cardiovascular complication | 390–459 | ~61.8% |
| Kidney disease | 580–589 | ~9.3% |
| Diabetic neuropathy | 354–357 | ~0.8% |

---

## Project Structure

```
Diabetes-Complication-Risk-Prediction-Model/
│
├── README.md
├── .gitignore
├── requirements.txt
├── app.py                        # Streamlit web application (deployed on HF Spaces)
│
├── data_scripts/
│   ├── data_collection.py        # Downloads dataset from UCI
│   ├── cleaning.py               # Handles missing values, outliers
│   └── feature_engineering.py   # Builds longitudinal features, derives labels
│
├── src/
│   ├── train_model.py            # Phase 3: trains all 3 models
│   ├── evaluate.py               # Phase 4: metrics, ROC curves, all plots
│   └── bias_check.py            # Phase 4: demographic subgroup bias analysis
│
├── models/                       # Saved model files (tracked via Git LFS)
│   ├── chain_rf.pkl
│   ├── mo_gbm.pkl
│   ├── nn_model.pth
│   └── training_metadata.pkl
│
├── data/processed/
│   └── scaler_final.pkl          # StandardScaler fitted on training set (Git LFS)
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_results.ipynb
│
└── report/
    ├── report.tex                # Full LaTeX report (Deliverables 1, 2 & 3)
    ├── report.pdf
    ├── deliverable_1/figures/
    └── deliverable_2/figures/    # All evaluation plots and bias check figures
```

---

## Model Weights

> **Trained model weights are hosted externally due to file size.**
>
> Download: [Hugging Face — Model Weights](https://huggingface.co/samishah2004/Diabetes-Complication-Risk-Prediction-Model)
>
> Place the downloaded files into the `models/` directory before running `evaluate.py` or `bias_check.py`.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/wahab124/Diabetes-Complication-Risk-Prediction-Model.git
cd Diabetes-Complication-Risk-Prediction-Model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python data_scripts/data_collection.py
```

### 4. Run the data pipeline

```bash
python data_scripts/cleaning.py
python data_scripts/data_preprocessing.py
python data_scripts/feature_engineering.py
```

### 5. Train all models (Phase 3)

```bash
python src/train_model.py
```

Outputs saved to `models/`:
- `chain_rf.pkl` — ClassifierChain (Random Forest)
- `mo_gbm.pkl` — MultiOutputClassifier (Gradient Boosting)
- `nn_model.pth` — PyTorch Neural Network weights
- `training_metadata.pkl` — test split + feature metadata

### 6. Evaluate models (Phase 4)

```bash
python src/evaluate.py
```

Generates all evaluation plots in `report/deliverable_2/figures/`: ROC curves, confusion matrices, threshold sensitivity analysis, model comparison chart, per-label recall, feature importance, training curve, and summary table.

### 7. Run bias check

```bash
python src/bias_check.py
```

Generates demographic subgroup recall plots (age, gender, race) and flags subgroups falling below the 0.80 clinical recall target.

### 8. Explore results in notebook

```bash
jupyter notebook notebooks/model_results.ipynb
```

### 9. Run the app locally

```bash
streamlit run app.py
```
---

## Model Architecture

### Three Multi-Label Models Trained

| Model | Strategy | Imbalance Handling |
|---|---|---|
| ClassifierChain (Random Forest) | Sequential label chaining | `class_weight="balanced"` |
| MultiOutputClassifier (Gradient Boosting) | Independent per-label | Per-estimator weighting |
| PyTorch Neural Network (MLP) | Joint multi-label output | Focal BCE loss + `pos_weight` |

### Neural Network Architecture

```
Input (n_features)
  -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.30)
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.30)
  -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.15)
  -> Linear(3)   [logits for 3 labels]
Loss: Focal BCE (gamma=2.0) with per-label pos_weight
Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
Scheduler: CosineAnnealingLR
```

---

## Hyperparameter Tuning

Parameters were tuned to maximise Recall-Macro >= 0.80 while maintaining reasonable F1-Macro.

### Random Forest (ClassifierChain)

| n_estimators | max_depth | min_samples_leaf | Recall-Macro | F1-Macro | Selected |
|---|---|---|---|---|---|
| 100 | 10 | 5 | 0.821 | 0.338 | |
| 150 | 12 | 4 | 0.849 | 0.347 | |
| **200** | **14** | **4** | **0.878** | **0.362** | YES |
| 200 | None | 2 | 0.871 | 0.341 | |

**Justification:** `n_estimators=200` gives strong ensemble stability. `max_depth=14` limits overfitting while capturing complex longitudinal feature interactions. `class_weight="balanced"` is essential given neuropathy's 0.8% prevalence.

### Gradient Boosting (MultiOutputClassifier)

| n_estimators | learning_rate | max_depth | subsample | Recall-Macro | Selected |
|---|---|---|---|---|---|
| 100 | 0.10 | 3 | 1.00 | 0.298 | |
| 100 | 0.05 | 4 | 0.85 | 0.311 | |
| **120** | **0.10** | **4** | **0.85** | **0.313** | YES |
| 150 | 0.10 | 5 | 0.85 | 0.309 | |

**Justification:** Retained as a baseline. Cannot recover neuropathy recall even after tuning due to the absence of focal loss. `subsample=0.85` reduces variance without sacrificing performance.

### Neural Network (PyTorch MLP)

| LR | Batch | Dropout | Epochs | Focal gamma | pos_weight_scale | Recall-Macro | Selected |
|---|---|---|---|---|---|---|---|
| 1e-3 | 128 | 0.20 | 50 | 2.0 | 2.0 | 0.831 | |
| 3e-4 | 64  | 0.30 | 70 | 2.0 | 2.5 | 0.852 | |
| **3e-4** | **128** | **0.30** | **70** | **2.0** | **2.5** | **0.853** | YES |
| 1e-4 | 128 | 0.30 | 100 | 3.0 | 3.0 | 0.848 | |

**Justification:** `lr=3e-4` with AdamW and cosine decay provides stable convergence. `dropout=0.30` with BatchNorm prevents overfitting. `gamma=2.0` in Focal loss gives the best recall/precision balance — higher gamma over-suppresses negatives and degrades kidney recall. `pos_weight_scale=2.5` amplifies signal for positive cases.

---

## Validation Strategy

A **three-way hold-out split** was used:

| Split | Proportion | Purpose |
|---|---|---|
| Train | 68% | Model fitting |
| Validation | 12% | Neural Network early stopping (best checkpoint) |
| Test | 20% | Final held-out evaluation (all models) |

Stratification on the label-combination string ensures proportional representation of all complication patterns across splits.

---

## Evaluation Metrics

| Metric | Why used |
|---|---|
| Recall (Sensitivity) | Primary target — minimise missed complications in clinical setting |
| F1-Micro | Overall system performance across all labels |
| F1-Macro | Equal weight to rare labels (critical for neuropathy at 0.8%) |
| ROC-AUC | Threshold-independent discrimination ability |

---

## Results Summary

| Model | F1-Micro | F1-Macro | Recall-Macro | AUC-Macro |
|---|---|---|---|---|
| Chain RF (base)    | 0.6056 | 0.3425 | 0.3938 | 0.7080 |
| MO GBM (base)      | 0.7259 | 0.2973 | 0.3131 | 0.7243 |
| Neural Net (base)  | 0.4637 | 0.3364 | 0.9158 | 0.7071 |
| Chain RF (tuned)   | 0.4949 | 0.3616 | **0.8778** | 0.7080 |
| MO GBM (tuned)     | 0.6495 | 0.3634 | 0.5913 | 0.7243 |
| Neural Net (tuned) | 0.5031 | 0.3614 | 0.8528 | 0.7071 |

**Selected model: ClassifierChain (Random Forest) with tuned thresholds** — the only model achieving Recall >= 0.80 simultaneously for all three labels.

---

## Technologies Used

- Python 3.10+
- PyTorch 2.3 — Neural network with Focal BCE loss
- Scikit-learn 1.5 — ClassifierChain, MultiOutputClassifier, metrics
- Pandas / NumPy — Data processing and longitudinal feature engineering
- Matplotlib / Seaborn — Evaluation visualisations

---

## Contributors

| Name | Contributions |
|---|---|
| Abdul Wahab | Data acquisition, cleaning & preprocessing, threshold tuning, confusion matrices, LaTeX report |
| Sami Shah | Longitudinal feature engineering, PyTorch NN architecture, Focal BCE loss, metrics, bias detection |
| Maheer Khurram | ClassifierChain (RF), hyperparameter tuning, ROC curves, feature importance |
| Azan Aziz | MultiOutputClassifier (GBM), class imbalance handling, threshold sensitivity, demographic subgroup analysis |
