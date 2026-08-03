"""Validate and register the tourism dataset stored in the repository."""

import os
import sys
import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"

# Columns expected from the project data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    # 1. Confirm the dataset exists in the repository
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]   # drop stray index columns

    # 2. Validate the schema against the data dictionary
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset is missing expected columns: {missing}")

    if df[TARGET].isna().any():
        sys.exit("ERROR: target column ProdTaken contains missing values")

    # 3. Print a summary of the registered dataset
    print("=" * 55)
    print("DATASET REGISTERED SUCCESSFULLY")
    print("=" * 55)
    print(f"Source        : {DATA_PATH}")
    print(f"Rows          : {df.shape[0]}")
    print(f"Columns       : {df.shape[1]}")
    print(f"Duplicates    : {df.duplicated().sum()}")
    print("\nAll expected columns are present.")
    print("\nTarget distribution (ProdTaken):")
    print(df[TARGET].value_counts().to_string())
    print(f"Positive class share: {df[TARGET].mean():.3f}")
    nulls = df.isna().sum()
    print("\nMissing values (non-zero only):")
    print(nulls[nulls > 0].to_string() if nulls.sum() else "None")
    print("\nColumn data types:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
