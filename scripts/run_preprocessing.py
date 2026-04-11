import os
from src.preprocessing import (
    load_data,
    drop_unnecessary_columns,
    impute_missing_values,
    summarize_missing_values,
    save_data,
    split_data,
    save_splits
)


def main():
    # define project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_path = os.path.join(base_dir, "data", "raw", "2023_BRFSS_CLEANED.csv")
    processed_path = os.path.join(base_dir, "data", "processed", "diabetes_clean.csv")

    print("Loading raw dataset...")
    df = load_data(raw_path)
    print("Original shape:", df.shape)

    # quick overview of missing values before processing
    print("\nMissing values before cleaning:")
    print(summarize_missing_values(df))

    # remove columns with high missingness / low relevance
    df = drop_unnecessary_columns(df)
    print("\nShape after dropping columns:", df.shape)

    print("\nMissing values after dropping high-missing features:")
    print(summarize_missing_values(df))

    # fill remaining missing values (imputation strategy defined in preprocessing.py)
    df = impute_missing_values(df)
    print("\nRemaining missing values after imputation:")
    print(summarize_missing_values(df))

    # convert multiclass target into binary (0 = no diabetes, 1 = any condition)
    df["DIABETES_STATUS"] = df["DIABETES_STATUS"].apply(
        lambda x: 0 if x == 0 else 1
    )

    # save cleaned dataset
    save_data(df, processed_path)
    print("\nClean dataset saved to:")
    print(processed_path)

    print("\nSplitting data into train and test...")
    X_train, X_test, y_train, y_test = split_data(df)

    # save splits for reproducible experiments
    save_splits(
        X_train,
        X_test,
        y_train,
        y_test,
        os.path.join(base_dir, "data", "processed")
    )

    print("Train/test datasets saved.")

if __name__ == "__main__":
    main()