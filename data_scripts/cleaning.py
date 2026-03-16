import pandas as pd
from data_collection import load_dataset,save_data


def replace_missing_values(df):
    df = df.replace("?", pd.NA)
    return df



def clean_dataset(df):

    # Replace '?' with NaN
    df = df.replace("?", pd.NA)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop ID columns
    df = df.drop(columns=["encounter_id", "patient_nbr"])

    # Drop columns with excessive missing values
    df = df.drop(columns=["weight", "payer_code", "medical_specialty"])
    # Weight Column has 96% missing values, 
    # Payer code has 40% missing values, 
    # Medical specialty has 50% missing values


    # Fill race
    df["race"] = df["race"].fillna("Unknown")

    # Fill diagnosis columns
    df["diag_1"] = df["diag_1"].fillna("0")
    df["diag_2"] = df["diag_2"].fillna("0")
    df["diag_3"] = df["diag_3"].fillna("0")

    # Fill lab test columns
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")

    return df




if __name__ == "__main__":

    df = load_dataset("/home/sami/Diabetes Complication Risk Prediction Model/data/diabetic_data.csv")
    
    
    df = replace_missing_values(df)
    null_percent = df.isnull().sum() * 100 / len(df)
    print(null_percent)
    
    cleaned_df = clean_dataset(df)
    print(cleaned_df.isnull().sum())
    save_data(cleaned_df, "/home/sami/Diabetes Complication Risk Prediction Model/data/cleaned_diabetic_data.csv")
    