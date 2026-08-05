import pandas as pd
def load_movielens(data_path):
    ratings=pd.read_csv(
        f"{data_path}/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID","MovieID","Rating","Timestamp"],
        encoding="latin-1"
    )
    movies = pd.read_csv(
        f"{data_path}/movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    users = pd.read_csv(
        f"{data_path}/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
        encoding="latin-1"
    )

    return ratings, movies, users