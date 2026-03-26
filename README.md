# AI-Based Multi-Label Prediction of Diabetes Complications

## Project Overview

This project develops an AI-based predictive analytics system that estimates the risk of multiple diabetes-related complications using patient clinical data. The model uses machine learning techniques to predict the likelihood of complications such as kidney disease, neuropathy, and cardiovascular conditions.

The system performs **multi-label classification**, allowing it to predict multiple complications simultaneously for a single patient. By incorporating **longitudinal clinical data** (information from multiple patient visits), the model aims to capture disease progression and improve prediction accuracy.

Early identification of high-risk patients can support healthcare professionals in implementing preventive interventions and improving long-term disease management.

---

## Problem Statement

Diabetes mellitus is a chronic condition that can lead to severe long-term complications if not properly managed. Common complications include diabetic kidney disease, neuropathy, and cardiovascular disorders. These complications often develop gradually and may remain undetected until significant damage has occurred.

Traditional medical assessment methods often rely on static thresholds or individual clinical measurements, which may fail to capture complex interactions between multiple health indicators over time.

This project aims to address this problem by building a machine learning model capable of analyzing clinical patterns and predicting the risk of multiple complications simultaneously.

---

## Objectives

The main objectives of this project are:

- Develop a **multi-label classification model** to predict multiple diabetes complications at once.
- Incorporate **longitudinal patient data** to analyze changes in clinical indicators across multiple visits.
- Optimize the model for **high sensitivity (recall)** to minimize missed complication cases.
- Evaluate model performance using **F1-Micro and F1-Macro scores**.
- Provide interpretable insights to support early medical intervention.

---

## Dataset

The dataset used in this project contains patient clinical records related to diabetes management.

Features may include:

- Age
- Blood Pressure
- HbA1c (blood glucose indicator)
- Cholesterol levels
- Body Mass Index (BMI)
- Medication information
- Other clinical measurements

These features are used to predict the following complications:

- Kidney Disease
- Diabetic Neuropathy
- Cardiovascular Disease

**Dataset Source:**  
[UCI Machine Learning Repository: Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

**Note:**  
Raw datasets are not uploaded to this repository due to file size limitations.

---

## Project Structure
```
diabetes-complication-risk-prediction-model/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data_scripts/
│   ├── data_collection.py
│   ├── cleaning.py
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│
└── report/
    ├── deliverable_1/
        ├── report.tex
        ├── report.pdf
        └── figures/
            ├── age_distribution.png
            ├── average_visits.png
            ├── class_distribution.png
            ├── complication_correlation.png
            ├── feature_correlation.png
            └── insulin_usage.png
        
```






---

## Data Processing Pipeline

The data processing pipeline consists of the following steps:

### 1. Data Collection
- Dataset downloaded from an open data source, UCI.

### 2. Data Cleaning
- Handling missing values
- Removing duplicate records
- Detecting and managing outliers

### 3. Feature Engineering
- Encoding categorical variables
- Normalizing numerical features
- Generating longitudinal features from multiple patient visits

### 4. Exploratory Data Analysis (EDA)
- Visualizing feature distributions
- Analyzing correlations between variables
- Studying class distributions for complications

---

## Technologies Used

**Programming Language**

- Python

**Libraries**

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyTorch

---

## Model Evaluation Metrics

Since missing a complication can have serious consequences, model evaluation focuses on **sensitivity and balanced performance across labels**.

Metrics used include:

- **Sensitivity (Recall)**
- **F1 Score (Micro)**
- **F1 Score (Macro)**

These metrics ensure the model performs effectively across multiple complication predictions.

---

## How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/wahab124/Diabetes-Complication-Risk-Prediction-Model.git
```
2. Navigate to the project directory
```bash
cd Diabetes-Complication-Risk-Prediction-Model
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the data preprocessing scripts
```bash
python data_scripts/data_collection.py
python data_scripts/cleaning.py
python data_scripts/data_preprocessing.py
python data_scripts/feature_engineering.py
```
5. Open the exploratory analysis notebook
```bash
jupyter notebook notebooks/deliverable_1.ipynb
```

## Contributors
- Abdul Wahab
- Maheer Khurram
- Syed Sami Shah

