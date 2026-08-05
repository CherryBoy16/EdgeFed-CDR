from preprocessing.loader import load_movielens
from preprocessing.cleaner import clean_ratings
from preprocessing.encoder import encode_ids
from preprocessing.sequence_generator import generate_sequences
from preprocessing.splitter import split_sequences

DATA_PATH = "datasets/raw/ml-m1"

ratings, movies, users = load_movielens(DATA_PATH)
cleaned_ratings=clean_ratings(ratings)
encoded_ratings=encode_ids(cleaned_ratings)
sequences=generate_sequences(encoded_ratings)
train_sequences,test_sequences=split_sequences(sequences)

print("=" * 50)
print("MovieLens 1M Loaded Successfully")
print("=" * 50)

print("\nRatings Shape:")
print(ratings.shape)

print("\nMovies Shape:")
print(movies.shape)

print("\nUsers Shape:")
print(users.shape)

print("\nFirst 5 Ratings")
print(ratings.head())

print("\nFirst 5 Movies")
print(movies.head())

print("\nFirst 5 Users")
print(users.head())

#cleaning
print("\nFirst 5 Cleaned records:")
print(cleaned_ratings.head())

#encoded
print("\nFirst 5 Encoded records:")
print(encoded_ratings.head())

#sequence generation
print("\nFirst 5 Sequences:")
for sample in sequences[:5]:
    print(sample)

#Train test
print(f"\nTotal Training Samples: {len(train_sequences)}")
print(f"Total Testing Samples : {len(test_sequences)}")