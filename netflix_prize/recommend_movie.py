import numpy as np
from omegaconf import DictConfig

from netflix_prize.data import load_movies
from netflix_prize.utils import format_number, load_config, load_model


@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    print(f"Looking for recommendations for the {cfg.recommend.movie_title}")

    print("Loading available movies...")
    movies_df = load_movies(cfg.data.movies_file)
    print(f"Number of movies in the database: {format_number(movies_df.shape[0])}")

    movies_filtered = movies_df[movies_df["title"] == cfg.recommend.movie_title].to_dict(
        "records"
    )

    if len(movies_filtered) == 0:
        print(f"Movie with title {cfg.recommend.movie_title} not found.")
    else:
        if len(movies_filtered) > 1:
            print(
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

    print(f"Loading model from {cfg.model_path}")
    model = load_model(cfg.model_path)

    print("Finding recommendations...")
    movie_inner_id = model.trainset.to_inner_iid(movie_raw_id)
    similar_movies = model.get_neighbors(
        movie_inner_id, k=cfg.recommend.num_recommendations
    )
    similar_movies = (model.trainset.to_raw_iid(inner_id) for inner_id in similar_movies)
    similar_movies = (movie_raw_id_to_title[raw_id] for raw_id in similar_movies)

    print()
    print(f"Recommended after watching {cfg.recommend.movie_title} ({movie_year}):")
    print("-" * 40)
    for i, movie in enumerate(similar_movies):
        print(f"{i+1}. ", movie)


if __name__ == "__main__":
    main()
