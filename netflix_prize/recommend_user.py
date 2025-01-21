import numpy as np
from loguru import logger
from omegaconf import DictConfig

from netflix_prize.data import load_movies
from netflix_prize.utils import format_number, load_config, load_model


def build_testset_anti_testset(trainset, ruid, fill=None):
    """All pairs (u, i) that are NOT in the training set"""
    fill = trainset.global_mean if fill is None else float(fill)
    iuid = trainset.to_inner_uid(ruid)
    user_items = {j for (j, _) in trainset.ur[iuid]}
    testset = [(ruid, trainset.to_raw_iid(i), r) for (i, r) in trainset.ur[iuid]]
    anti_testset = [
        (ruid, trainset.to_raw_iid(i), fill)
        for i in trainset.all_items()
        if i not in user_items
    ]
    return testset, anti_testset


@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    logger.info(f"Looking for recommendations for the user {cfg.recommend.user_id}")

    logger.info("Loading available movies...")
    movies_df = load_movies(cfg.data.movies_file)
    logger.info(f"Number of movies in the database: {format_number(movies_df.shape[0])}")

    movie_id_to_title = movies_df.set_index("movie_id").to_dict("index")
    # add a year to the title
    movie_id_to_title = {
        k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})"
        for k, v in movie_id_to_title.items()
    }

    logger.info(f"Loading model from {cfg.model_path}")
    model = load_model(cfg.model_path)

    logger.info("Looking for the movies rated by the user...")
    testset, anti_testset = build_testset_anti_testset(
        model.trainset, cfg.recommend.user_id
    )

    logger.info("Highest rated movies by the user:")
    logger.info("-" * 40)
    user_ratings = [(movie_id, r_true) for user_id, movie_id, r_true in testset]
    del testset  # free memory
    user_ratings.sort(key=lambda x: x[1], reverse=True)

    for i, (movie_id, rating) in enumerate(user_ratings):
        logger.info(f"{i+1}. {movie_id_to_title[movie_id]} ({rating} stars)")
        if i == cfg.recommend.num_recommendations - 1:
            break

    logger.info("Predicting ratings for unrated movies...")
    predictions = model.test(anti_testset)
    del anti_testset

    logger.info("Recommended movies:")
    logger.info("-" * 40)
    user_ratings = [
        (movie_id, r_est) for user_id, movie_id, r_true, r_est, _ in predictions
    ]
    del predictions
    user_ratings.sort(key=lambda x: x[1], reverse=True)

    for i, (movie_id, rating) in enumerate(user_ratings):
        logger.info(f"{i+1}. {movie_id_to_title[movie_id]} ({rating:.1f} stars)")
        if i == cfg.recommend.num_recommendations - 1:
            break


if __name__ == "__main__":
    main()
