"""Clean the registered dataset and produce stratified train/test splits."""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"
DROP_COLUMNS = ["CustomerID"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {DATA_PATH}")

    # Drop the stray index column produced by the original Excel export
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        print("Dropped export artefact columns:", unnamed)

    # Drop identifier columns with no predictive value
    present = [c for c in DROP_COLUMNS if c in df.columns]
    if present:
        df = df.drop(columns=present)
        print("Dropped identifier columns:", present)

    # Fix the known Gender typo
    n_typo = int((df["Gender"] == "Fe Male").sum())
    if n_typo:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        print(f"Corrected {n_typo} 'Fe Male' entries to 'Female'")
    print("Gender categories after cleaning:", sorted(df["Gender"].unique().tolist()))

    # Remove duplicates that surface once the unique identifier is removed
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows; {len(df)} rows remain")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nXtrain: {Xtrain.shape}   Xtest: {Xtest.shape}")
    print("ytrain positive rate:", round(ytrain.mean(), 3))
    print("ytest  positive rate:", round(ytest.mean(), 3))
    print("\nSaved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
