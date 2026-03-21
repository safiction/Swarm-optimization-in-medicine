import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df):
    X = df.drop(columns=["DIABETES_STATUS"])
    y = df["DIABETES_STATUS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_splits(X_train, X_test, y_train, y_test, base_path):
    X_train.to_csv(f"{base_path}/X_train.csv", index=False)
    X_test.to_csv(f"{base_path}/X_test.csv", index=False)
    y_train.to_csv(f"{base_path}/y_train.csv", index=False)
    y_test.to_csv(f"{base_path}/y_test.csv", index=False)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_to_drop = ["BP_MEDS", "ALHL_STATUS", "POOR_HLTH_DAYS", "YEAR"]

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    return df.drop(columns=cols_to_drop)


def get_feature_groups():
    numeric_cols = [
        "WGHT (lbs)",
        "HGHT (ft)",
        "BMI",
        "PHYS_HLTH_DAYS",
        "MENT_HLTH_DAYS"
    ]

    categorical_cols = [
        "SEX",
        "AGE",
        "EDUCATION_LEVEL",
        "EMPLOYMENT_STATUS",
        "INCOME_LEVEL",
        "MARITAL_STATUS",
        "INSR_STATUS",
        "DCTR_STATUS",
        "COST_STATUS",
        "CHKP_STATUS",
        "GEN_HLTH",
        "SMOK_STATUS",
        "EXER_STATUS",
        "HIGH_BP",
        "HIGH_CHOL",
        "CHOL_MEDS",
        "HAD_STROKE",
        "HAD_HEARTDISEASE",
        "DIABETES_STATUS"
    ]

    return numeric_cols, categorical_cols


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols, categorical_cols = get_feature_groups()

    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    summary = pd.DataFrame({
        "feature": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percent": missing_pct.values
    })

    summary = summary[summary["missing_count"] > 0]
    summary = summary.sort_values("missing_percent", ascending=False)

    return summary


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)