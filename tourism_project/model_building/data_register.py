"""Register and validate the raw tourism dataset."""

import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected columns: {missing}")

    if df[TARGET].isna().any():
        sys.exit(f"ERROR: target '{TARGET}' contains missing values")

    print("Dataset registered successfully")
    print("Source     :", DATA_PATH)
    print("Rows       :", f"{len(df):,}")
    print("Columns    :", df.shape[1])
    print("Duplicates :", int(df.duplicated().sum()))

    counts = df[TARGET].value_counts().sort_index()
    print(f"\nTarget ({TARGET}) distribution:")
    for label, n in counts.items():
        print(f"  {label} -> {n:,}")
    print("Positive class share:", round(df[TARGET].mean(), 3))

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print("\nMissing values:")
    print("  none" if nulls.empty else nulls.to_string())

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
