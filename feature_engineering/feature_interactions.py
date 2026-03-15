"""
AutoML Feature Interaction Module

Reads scaled dataset from:
storage/output/{userid}/{filename}_scaling.csv

Reads profiling metadata from:
storage/meta_data/{userid}/{filename}_profiling.json

Creates safe interaction features between numeric columns.

Output:
Modifies the same scaling CSV file.
"""

import os
import json
import pandas as pd
import numpy as np


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
OUTPUT_DIR = os.path.join(STORAGE_DIR, "output")
META_DIR = os.path.join(STORAGE_DIR, "meta_data")


# =========================================================
# NUMERIC COLUMN DETECTION
# =========================================================

def get_numeric_columns_from_metadata(metadata):

    numeric_cols = []

    for col in metadata["column_wise_summary"]:

        if col.get("structural_type") == "numeric":
            numeric_cols.append(col["column_name"])

    return numeric_cols


# =========================================================
# FEATURE INTERACTION ENGINE
# =========================================================

def create_feature_interactions(df, numeric_columns):

    created_features = []

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):

            col1 = numeric_columns[i]
            col2 = numeric_columns[j]

            if col1 not in df.columns or col2 not in df.columns:
                continue

            # Multiplication
            new_col = f"{col1}_x_{col2}"

            try:
                df[new_col] = df[col1] * df[col2]
                created_features.append(new_col)
            except Exception:
                pass

            # Difference
            new_col = f"{col1}_minus_{col2}"

            try:
                df[new_col] = df[col1] - df[col2]
                created_features.append(new_col)
            except Exception:
                pass

            # Division
            new_col = f"{col1}_div_{col2}"

            try:
                df[new_col] = df[col1] / (df[col2] + 1e-6)
                created_features.append(new_col)
            except Exception:
                pass

    print(f"\nCreated {len(created_features)} interaction features")

    return df


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_user_dataset(userid):

    user_meta_path = os.path.join(META_DIR, userid)
    user_output_path = os.path.join(OUTPUT_DIR, userid)

    if not os.path.exists(user_meta_path):
        print("User metadata folder not found.")
        return

    if not os.path.exists(user_output_path):
        print("User output folder not found.")
        return

    profiling_files = [
        f for f in os.listdir(user_meta_path) if f.endswith("_profiling.json")
    ]

    if not profiling_files:
        print("No profiling metadata found.")
        return

    for profiling_file in profiling_files:

        base_name = profiling_file.replace("_profiling.json", "")

        metadata_path = os.path.join(user_meta_path, profiling_file)
        scaling_path = os.path.join(
            user_output_path, f"{base_name}_scaling.csv"
        )

        if not os.path.exists(scaling_path):
            print(f"Scaling file not found: {scaling_path}")
            continue

        print(f"\nProcessing dataset: {base_name}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        numeric_columns = get_numeric_columns_from_metadata(metadata)

        if len(numeric_columns) < 2:
            print("Not enough numeric columns for interactions.")
            continue

        print(f"Numeric columns detected: {numeric_columns}")

        # Load dataset
        df = pd.read_csv(scaling_path)

        # Apply interactions
        df = create_feature_interactions(df, numeric_columns)

        # Save back
        df.to_csv(scaling_path, index=False)

        print(f"Updated dataset saved to: {scaling_path}")


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    print("\nAutoML Feature Interaction Engine\n")

    userid = input("Enter User ID: ").strip()

    process_user_dataset(userid)


if __name__ == "__main__":
    main()