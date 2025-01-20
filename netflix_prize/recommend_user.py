import numpy as np
from omegaconf import DictConfig
from surprise import Dataset, Reader

from netflix_prize.data import load_movies, load_ratings
from netflix_prize.utils import format_number, load_config, load_model


def build_testset_anti_testset(trainset, ruid, fill=None):
    """All pairs (u, i) that are NOT in the training set"""
    fill = trainset.global_mean if fill is None else float(fill)
    iuid = trainset.to_inner_uid(ruid)
    user_items = {j for (j, r) in trainset.ur[iuid]}
    testset = [
        (trainset.to_raw_uid(iuid), trainset.to_raw_iid(i), r)
        for (i, r) in trainset.ur[iuid]
    ]
    anti_testset = [
        (trainset.to_raw_uid(iuid), trainset.to_raw_iid(i), fill)
        for i in trainset.all_items()
        if i not in user_items
    ]
    return testset, anti_testset


@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    print(f"Looking for recommendations for the user {cfg.recommend.user_id}")

    print("Loading available movies...")
    movies_df = load_movies(cfg.data.movies_file)
    print(f"Number of movies in the database: {format_number(movies_df.shape[0])}")

    movie_id_to_title = movies_df.set_index("movie_id").to_dict("index")
    # add a year to the title
    movie_id_to_title = {
        k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})"
        for k, v in movie_id_to_title.items()
    }

    print(f"Loading model from {cfg.model_path}")
    model = load_model(cfg.model_path)

    print("Loading ratings data...")
    ratings_df = load_ratings(cfg.data.rating_files, cfg.data.chunk_size)
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]], reader
    ).build_full_trainset()
    del ratings_df  # free memory

    print("Looking for the movies rated by the user...")
    testset, anti_testset = build_testset_anti_testset(trainset, cfg.recommend.user_id)

    user_ratings = []
    for user_id, movie_id, r_true in testset:
        user_ratings.append((movie_id, r_true))

    del testset  # free memory

    user_ratings.sort(key=lambda x: x[1], reverse=True)
    highest_rated_movies = user_ratings[: cfg.recommend.num_recommendations]
    highest_rated_movies = [
        movie_id_to_title[movie_id] for movie_id, _ in highest_rated_movies
    ]

    print("Highest rated movies by the user:")
    print("-" * 40)
    for i, movie in enumerate(highest_rated_movies):
        print(f"{i+1}. ", movie)

    print("Predicting ratings for unrated movies...")
    predictions = model.test(anti_testset)

    user_ratings = []
    for user_id, movie_id, r_true, r_est, _ in predictions:
        if user_id == cfg.recommend.user_id:
            user_ratings.append((movie_id, r_est))

    user_ratings.sort(key=lambda x: x[1], reverse=True)
    recommended_movies = user_ratings[: cfg.recommend.num_recommendations]
    recommended_movies = [
        movie_id_to_title[movie_id] for movie_id, _ in recommended_movies
    ]

    print("Recommended movies:")
    print("-" * 40)
    for i, movie in enumerate(recommended_movies):
        print(f"{i+1}. ", movie)


if __name__ == "__main__":
    main()
