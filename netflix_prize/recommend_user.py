import numpy as np
from surprise import Dataset, Reader

from netflix_prize.data import load_movies, load_ratings
from netflix_prize.utils import format_number, load_model

top_k = 10
target_user_id = 2380848
model_path = "checkpoints/centered_knn/model.pkl"
data_files = [
    "data/combined_data_1.txt",
    # "data/combined_data_2.txt",
    # "data/combined_data_3.txt",
    # "data/combined_data_4.txt",
]
movies_path = "data/movie_titles_fixed.csv"

movies_df = load_movies(movies_path)
print(f"Number of movies in the database: {format_number(movies_df.shape[0])}")
print(f"The oldest movie: {int(movies_df['year'].min())}")
print(f"The newest movie: {int(movies_df['year'].max())}")

movie_id_to_title = movies_df.set_index("movie_id").to_dict("index")
# add a year to the title
movie_id_to_title = {
    k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})" for k, v in movie_id_to_title.items()
}


model = load_model(model_path)

# data
print("Loading ratings data...")
ratings_df = load_ratings(data_files, chunk_size=100_000)
print("Loaded ratings data.")

reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader).build_full_trainset()
del ratings_df  # free memory


def build_testset_anti_testset(trainset, ruid, fill=None):
    """All pairs (u, i) that are NOT in the training set"""
    fill = trainset.global_mean if fill is None else float(fill)
    iuid = trainset.to_inner_uid(ruid)
    user_items = {j for (j, r) in trainset.ur[iuid]}
    testset = [(trainset.to_raw_uid(iuid), trainset.to_raw_iid(i), r) for (i, r) in trainset.ur[iuid]]
    anti_testset = [
        (trainset.to_raw_uid(iuid), trainset.to_raw_iid(i), fill) for i in trainset.all_items() if i not in user_items
    ]
    return testset, anti_testset


print("Looking for the movies rated by the user...")
testset, anti_testset = build_testset_anti_testset(trainset, target_user_id)

user_ratings = []
for user_id, movie_id, r_true in testset:
    user_ratings.append((movie_id, r_true))

user_ratings.sort(key=lambda x: x[1], reverse=True)
highest_rated_movies = user_ratings[:top_k]
highest_rated_movies = [movie_id_to_title[movie_id] for movie_id, _ in highest_rated_movies]

print("Highest rated movies by the user:")
for i, movie in enumerate(highest_rated_movies):
    print(f"{i+1}. ", movie)

del testset  # free memory

print("Predicting ratings for unrated movies...")
predictions = model.test(anti_testset)


user_ratings = []
for user_id, movie_id, r_true, r_est, _ in predictions:
    if user_id == target_user_id:
        user_ratings.append((movie_id, r_est))

user_ratings.sort(key=lambda x: x[1], reverse=True)
recommended_movies = user_ratings[:top_k]
recommended_movies = [movie_id_to_title[movie_id] for movie_id, _ in recommended_movies]

print("Recommended movies:")
for i, movie in enumerate(recommended_movies):
    print(f"{i+1}. ", movie)
