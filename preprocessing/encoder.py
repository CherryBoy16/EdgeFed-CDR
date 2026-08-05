import os
import pickle
from sklearn.preprocessing import LabelEncoder


def encode_ids(ratings_df):
    """
    Encode UserID and MovieID into consecutive integers.
    """

    print("\n" + "=" * 60)
    print("STEP 2 : ID ENCODING")
    print("=" * 60)

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings_df["UserID"] = user_encoder.fit_transform(ratings_df["UserID"])
    ratings_df["MovieID"] = movie_encoder.fit_transform(ratings_df["MovieID"])

    os.makedirs("datasets/processed", exist_ok=True)

    ratings_df.to_csv(
        "datasets/processed/encoded_ratings.csv",
        index=False
    )

    with open("datasets/processed/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)

    with open("datasets/processed/movie_encoder.pkl", "wb") as f:
        pickle.dump(movie_encoder, f)

    print(f"Encoded Users  : {ratings_df['UserID'].nunique()}")
    print(f"Encoded Movies : {ratings_df['MovieID'].nunique()}")

    print("\nEncoded dataset saved.")
    print("User encoder saved.")
    print("Movie encoder saved.")

    return ratings_df