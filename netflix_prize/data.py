import pandas as pd


def load_movies(path: str) -> pd.DataFrame:
    movies = pd.read_csv(path, header=None, names=["movie_id", "year", "title"], sep=";")
    return movies


def load_ratings(paths: str, chunk_size: int = 100_000) -> pd.DataFrame:
    all_ratings = []
    for path in paths:
        print(f"Loading ratings from {path}...")
        for i, chunk in enumerate(parse_ratings_batch(path, chunk_size)):
            if i % 100 == 0:
                print(f"Processed {i * chunk_size} ratings...")
            all_ratings.append(chunk)
    return pd.concat(all_ratings, ignore_index=True)


def load_probe(path: str, chunk_size: int = 100_000) -> pd.DataFrame:
    all_probes = []
    for i, chunk in enumerate(parse_probe_batch(path), chunk_size):
        if i % 100 == 0:
            print(f"Processed {i * chunk_size} ratings...")
        all_probes.append(chunk)
    return pd.concat(all_probes, ignore_index=True)


def train_test_split(ratings: pd.DataFrame, probe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_ratings = probe.merge(ratings, on=["movie_id", "user_id"])
    train_ratings = ratings[~ratings.index.isin(test_ratings.index)]
    return train_ratings, test_ratings


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


def parse_probe_batch(data_path: str, chunk_size: int = 100_000):
    batch = []
    movie_id = None
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                movie_id = int(line[:-1])
            else:
                user_id = int(line)
                batch.append((movie_id, user_id))

                if len(batch) >= chunk_size:
                    yield pd.DataFrame(batch, columns=["movie_id", "user_id"])
                    batch = []
    if batch:
        yield pd.DataFrame(batch, columns=["movie_id", "user_id"])
