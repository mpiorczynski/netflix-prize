import numpy as np
from loguru import logger
from omegaconf import DictConfig

from netflix_prize.data import load_movies
from netflix_prize.utils import format_number, load_config, load_model


@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    logger.info(f"Looking for recommendations after watching {cfg.recommend.movie_title}")

    logger.info("Loading available movies...")
    movies_df = load_movies(cfg.data.movies_file)
    logger.info(f"Number of movies in the database: {format_number(movies_df.shape[0])}")

    movies_filtered = movies_df[movies_df["title"] == cfg.recommend.movie_title].to_dict(
        "records"
    )

    if len(movies_filtered) == 0:
        logger.info(f"Movie with title {cfg.recommend.movie_title} not found.")
    else:
        if len(movies_filtered) > 1:
            logger.info(
                f"Multiple versions of the {cfg.recommend.movie_title} found. Selecting the first one."
            )
        movie_raw_id = movies_filtered[0]["movie_id"]
        movie_year = int(movies_filtered[0]["year"])

    movie_raw_id_to_title = movies_df.set_index("movie_id").to_dict("index")
    # add a year to the title
    movie_raw_id_to_title = {
        k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})"
        for k, v in movie_raw_id_to_title.items()
    }

    logger.info(f"Loading model from {cfg.model_path}")
    model = load_model(cfg.model_path)

    logger.info("Finding recommendations...")
    movie_inner_id = model.trainset.to_inner_iid(movie_raw_id)
    similar_movies = model.get_neighbors(
        movie_inner_id, k=cfg.recommend.num_recommendations
    )

    logger.info(f"Recommended after watching {cfg.recommend.movie_title} ({movie_year}):")
    logger.info("-" * 40)
    for i, movie_iid in enumerate(similar_movies):
        logger.info(
            f"{i+1}. {movie_raw_id_to_title[model.trainset.to_raw_iid(movie_iid)]}"
        )


if __name__ == "__main__":
    main()
