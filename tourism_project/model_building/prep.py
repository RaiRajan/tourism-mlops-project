"""Clean the dataset and create stratified train/test splits."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"
DROP_COLS = ["CustomerID"]          # unique identifier: no predictive value
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # 1. Load the dataset directly from the repository data folder
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {DATA_PATH}")

    # 2. Remove unnecessary columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]   # stray index columns
    dropped = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Dropped identifier columns: {dropped}")

    # 3. Fix the known data-entry inconsistency in Gender ("Fe Male" -> "Female")
    if "Gender" in df.columns:
        n_fixed = (df["Gender"] == "Fe Male").sum()
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
        print(f"Corrected {n_fixed} 'Fe Male' entries to 'Female'")
        print("Gender categories after cleaning:", sorted(df['Gender'].unique()))

    # 4. Remove exact duplicate records
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows; {len(df)} rows remain")

    # 5. Separate features and target
    #    Any residual missing values are handled by the imputers inside the
    #    model pipeline, which prevents data leakage from the test split.
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 6. Stratified split so both sets keep the original class balance
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Save the splits locally for the next pipeline job
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nXtrain: {Xtrain.shape}   Xtest: {Xtest.shape}")
    print(f"ytrain positive rate: {ytrain.mean():.3f} | ytest positive rate: {ytest.mean():.3f}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
