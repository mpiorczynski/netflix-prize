import pandas as pd


def load_ratings(paths, chunk_size=100_000):
    all_ratings = []
    for path in paths:
        print(f"Loading ratings from {path}...")
        for i, chunk in enumerate(parse_ratings_batch(path, chunk_size)):
            if i % 100 == 0:
                print(f"Processed {i * chunk_size} ratings...")
            all_ratings.append(chunk)
    return pd.concat(all_ratings, ignore_index=True)


def parse_ratings_batch(data_path, chunk_size=100_000):
    batch = []
    movie_id = None
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                movie_id = int(line[:-1])
            else:
                user_id, rating, date = line.split(",")
                timestamp = pd.Timestamp(date).timestamp()
                batch.append((movie_id, int(user_id), int(rating), int(timestamp)))

                if len(batch) >= chunk_size:
                    yield pd.DataFrame(batch, columns=["movie_id", "user_id", "rating", "timestamp"])
                    batch = []
    if batch:
        yield pd.DataFrame(batch, columns=["movie_id", "user_id", "rating", "timestamp"])


def load_movies(path):
    movies = pd.read_csv(path, header=None, names=["movie_id", "year", "title"], sep=";")
    return movies


def train_test_split(ratings, strategy="user", **split_kwargs):
    # TODO: use probe.txt
    ...
