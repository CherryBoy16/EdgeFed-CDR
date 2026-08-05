import os
import pickle


def generate_sequences(ratings_df, min_sequence_length=2):
    """
    Generate prefix sequences for sequential recommendation.
    """

    print("\n" + "=" * 60)
    print("STEP 3 : SEQUENCE GENERATION")
    print("=" * 60)

    sequences = []

    grouped = ratings_df.groupby("UserID")

    for user_id, group in grouped:

        movies = group["MovieID"].tolist()

        if len(movies) < min_sequence_length:
            continue

        for i in range(1, len(movies)):

            input_sequence = movies[:i]
            target = movies[i]

            sequences.append(
                {
                    "user": user_id,
                    "input": input_sequence,
                    "target": target
                }
            )

    print(f"\nTotal Training Samples: {len(sequences)}")

    os.makedirs("datasets/processed", exist_ok=True)

    with open(
        "datasets/processed/sequences.pkl",
        "wb"
    ) as f:

        pickle.dump(sequences, f)

    print("\nSequence dataset saved.")

    return sequences