import numpy as np

from netflix_prize.data import load_movies
from netflix_prize.utils import format_number, load_model

movie_title = "Batman Begins"
model_path = "checkpoints/centered_knn.pkl"
top_k = 10

movies_df = load_movies("data/movie_titles_fixed.csv")
print(f"Number of movies in the database: {format_number(movies_df.shape[0])}")
print(f"The oldest movie: {int(movies_df['year'].min())}")
print(f"The newest movie: {int(movies_df['year'].max())}")

movies_filtered = movies_df[movies_df["title"] == movie_title].to_dict("records")

if len(movies_filtered) == 0:
    print(f"Movie with title {movie_title} not found.")
else:
    if len(movies_filtered) > 1:
        print(f"Multiple versions of the {movie_title} found. Selecting the first one.")
    movie_raw_id = movies_filtered[0]["movie_id"]
    movie_year = int(movies_filtered[0]["year"])

movie_raw_id_to_title = movies_df.set_index("movie_id").to_dict("index")
# add a year to the title
movie_raw_id_to_title = {k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})" for k, v in movie_raw_id_to_title.items()}

print(f"Loading model from {model_path}")
model = load_model(model_path)

movie_inner_id = model.trainset.to_inner_iid(movie_raw_id)

movie_neighbors_inner_ids = model.get_neighbors(movie_inner_id, k=top_k)
movie_neighbors_raw_ids = (
    model.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors_inner_ids
)
movie_neighbors_titles = (movie_raw_id_to_title[raw_id] for raw_id in movie_neighbors_raw_ids)

print()
print(f"Recommended after watching {movie_title} ({movie_year}):")
print("-" * 40)
for i, movie in enumerate(movie_neighbors_titles):
    print(f"{i+1}. ", movie)