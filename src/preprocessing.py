import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    return df

def split_features_target(df, target="DIABETES_STATUS"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y