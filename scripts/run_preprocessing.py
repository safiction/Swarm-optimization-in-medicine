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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_path = os.path.join(base_dir, "data", "raw", "2023_BRFSS_CLEANED.csv")
    processed_path = os.path.join(base_dir, "data", "processed", "diabetes_clean.csv")

    print("Loading raw dataset...")
    df = load_data(raw_path)
    print("Original shape:", df.shape)

    print("\nMissing values before cleaning:")
    print(summarize_missing_values(df))

    df = drop_unnecessary_columns(df)
    print("\nShape after dropping columns:", df.shape)

    print("\nMissing values after dropping high-missing features:")
    print(summarize_missing_values(df))

    df = impute_missing_values(df)

    print("\nRemaining missing values after imputation:")
    print(summarize_missing_values(df))

    save_data(df, processed_path)
    print("\nClean dataset saved to:")
    print(processed_path)
    print("\nSplitting data into train and test...")
    X_train, X_test, y_train, y_test = split_data(df)

    save_splits(X_train, X_test, y_train, y_test, os.path.join(base_dir, "data", "processed"))

    print("Train/test datasets saved.")

if __name__ == "__main__":
    main()