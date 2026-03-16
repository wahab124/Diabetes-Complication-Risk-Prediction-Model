import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_collection import load_dataset, save_data

# Convert age range to numeric by finding the midpoint of the range
def convert_age(age_range):
    age_range = age_range.strip("[]()")
    lower, upper = age_range.split("-")
    return (int(lower) + int(upper)) / 2

from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_dataset(input_path, output_path):

    df = load_dataset(input_path)

    # Convert age ranges to numeric
    df["age"] = df["age"].apply(convert_age)

    # Encode gender
    df["gender"] = df["gender"].map({
        "Male": 1,
        "Female": 0
    })

    # Encode binary variables
    df["change"] = df["change"].map({"Ch": 1, "No": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    # Encode medication columns
    med_map = {
        "No": 0,
        "Steady": 1,
        "Up": 2,
        "Down": 3
    }

    medication_cols = [
        "metformin","repaglinide","nateglinide","chlorpropamide",
        "glimepiride","acetohexamide","glipizide","glyburide",
        "tolbutamide","pioglitazone","rosiglitazone","acarbose",
        "miglitol","troglitazone","tolazamide","examide",
        "citoglipton","insulin","glyburide-metformin",
        "glipizide-metformin","glimepiride-pioglitazone",
        "metformin-rosiglitazone","metformin-pioglitazone"
    ]

    for col in medication_cols:
        df[col] = df[col].map(med_map).fillna(0)

    # Ensure lab-test columns explicitly include "None"
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")

    # One-hot encode categorical columns (including "None")
    categorical_cols = [
        "race",
        "max_glu_serum",
        "A1Cresult"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Convert booleans to integers
    df = df.replace({True: 1, False: 0})

    # Scale numeric columns
    numeric_cols = [
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.to_csv(output_path, index=False)

    print("Preprocessed dataset saved.")




if __name__ == "__main__":

    preprocess_dataset(
        "/home/sami/Diabetes Complication Risk Prediction Model/data/cleaned_diabetic_data.csv",
        "/home/sami/Diabetes Complication Risk Prediction Model/data/diabetes_processed.csv"
    )
    