import os
import pandas as pd


def clean_ratings(ratings_df, min_interactions=5):
    """
    Clean the MovieLens ratings dataset.

    Steps:
    1. Check missing values
    2. Remove duplicate rows
    3. Sort by UserID and Timestamp
    4. Remove users with fewer than min_interactions
    5. Save cleaned dataset
    """

    print("\n" + "=" * 60)
    print("STEP 1 : DATA CLEANING")
    print("=" * 60)

    # Dataset information
    print(f"\nOriginal Shape: {ratings_df.shape}")

    print("\nMissing Values:")
    print(ratings_df.isnull().sum())

    # Remove duplicates
    before = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates()
    after = len(ratings_df)

    print(f"\nDuplicate Rows Removed: {before - after}")

    # Sort chronologically
    ratings_df = ratings_df.sort_values(
        by=["UserID", "Timestamp"]
    ).reset_index(drop=True)

    print("\nDataset sorted by UserID and Timestamp.")

    # Filter users with minimum interactions
    user_counts = ratings_df["UserID"].value_counts()

    valid_users = user_counts[user_counts >= min_interactions].index

    ratings_df = ratings_df[
        ratings_df["UserID"].isin(valid_users)
    ].reset_index(drop=True)

    print(f"\nUsers remaining after filtering: {ratings_df['UserID'].nunique()}")

    # Save
    os.makedirs("datasets/processed", exist_ok=True)

    ratings_df.to_csv(
        "datasets/processed/clean_ratings.csv",
        index=False
    )

    print("\nClean dataset saved to:")
    print("datasets/processed/clean_ratings.csv")

    print("\nFinal Shape:", ratings_df.shape)

    return ratings_df