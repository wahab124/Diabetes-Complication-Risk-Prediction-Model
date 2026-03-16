import pandas as pd
from data_collection import load_dataset,save_data

# def load_data(path):
#     return pd.read_csv(path)


def safe_float(value):
    try:
        return float(value)
    except:
        return None


def classify_complications(row):

    diagnoses = [
        row["diag_1"],
        row["diag_2"],
        row["diag_3"]
    ]

    cardiovascular = 0
    kidney = 0
    neuropathy = 0

    for diag in diagnoses:

        diag_val = safe_float(diag)

        if diag_val is None:
            continue

        # Cardiovascular
        if 390 <= diag_val <= 459:
            cardiovascular = 1

        # Kidney disease
        if 580 <= diag_val <= 629:
            kidney = 1

        # Neuropathy
        if int(diag_val) == 357 or str(diag).startswith("250.6"):
            neuropathy = 1

    return pd.Series([cardiovascular, kidney, neuropathy])


def create_labels(df):

    df[
        [
            "cardiovascular_complication",
            "kidney_complication",
            "neuropathy_complication",
        ]
    ] = df.apply(classify_complications, axis=1)

    return df

def drop_labels(df):

    features_to_drop = [
        "diag_1",
        "diag_2",
        "diag_3"
    ]

    df = df.drop(columns=features_to_drop)
    df = df.drop(columns=["readmitted"])
    return df

def run_feature_engineering(input_path, output_path):

    df = load_dataset(input_path)

    df = create_labels(df)

    df = drop_labels(df)

    save_data(df, output_path)

    print("Feature engineering complete.")


if __name__ == "__main__":

    run_feature_engineering(
        "/home/sami/Diabetes Complication Risk Prediction Model/data/diabetes_processed.csv",
        "/home/sami/Diabetes Complication Risk Prediction Model/data/diabetes_final.csv"
    )