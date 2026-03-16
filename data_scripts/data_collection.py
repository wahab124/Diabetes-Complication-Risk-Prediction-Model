import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def overview(df):
    print("\n\nShape:\n ", df.shape)
    print("\n\nColumns:\n ", df.columns.tolist())
    print("\n\nData types:\n ", df.dtypes)
    print("\n\nMissing values:\n ", df.isnull().sum())
    print("\n\nFirst 5 rows:\n ", df.head())

def save_data(df, path):
    df.to_csv(path, index=False)
    
if __name__ == "__main__":
    path = "/home/sami/Diabetes Complication Risk Prediction Model/data/diabetic_data.csv"
    df = load_dataset(path)
    overview(df)
    save_data(df, "/home/sami/Diabetes Complication Risk Prediction Model/data/raw_data.csv")
