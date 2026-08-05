import os
import pickle
from sklearn.model_selection import train_test_split


def split_sequences(sequences, test_size=0.2):
    """
    Split generated sequences into training and testing sets.
    """

    print("\n" + "=" * 60)
    print("STEP 4 : TRAIN / TEST SPLIT")
    print("=" * 60)

    train_sequences, test_sequences = train_test_split(
        sequences,
        test_size=test_size,
        random_state=42,
        shuffle=False
    )

    os.makedirs("datasets/processed", exist_ok=True)

    with open("datasets/processed/train_sequences.pkl", "wb") as f:
        pickle.dump(train_sequences, f)

    with open("datasets/processed/test_sequences.pkl", "wb") as f:
        pickle.dump(test_sequences, f)

    print(f"Training Samples : {len(train_sequences)}")
    print(f"Testing Samples  : {len(test_sequences)}")

    return train_sequences, test_sequences