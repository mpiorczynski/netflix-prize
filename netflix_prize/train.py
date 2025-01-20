import time

from omegaconf import DictConfig
from surprise import Dataset, Reader

from netflix_prize.data import load_ratings
from netflix_prize.utils import (
    format_number,
    format_time,
    import_class_by_name,
    load_config,
    save_model,
)


def print_ratings_info(ratings):
    num_users = ratings["user_id"].nunique()
    num_items = ratings["movie_id"].nunique()
    num_ratings = len(ratings)

    print(f"Number of users: {format_number(num_users)}")
    print(f"Number of items: {format_number(num_items)}")
    print(f"Number of ratings (1-5 stars): {format_number(num_ratings)}")

    sparsity = num_ratings / (num_users * num_items)
    print(f"Sparsity ratio: {sparsity:.2%}")

    return num_users, num_items, num_ratings, sparsity



@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    # data
    load_start = time.time()
    ratings_df = load_ratings(cfg.data.files, chunk_size=100_000)
    load_end = time.time()
    load_time = load_end - load_start
    print(f"Ratings loaded in {format_time(load_time)}")

    num_users, num_items, num_ratings, sparsity = print_ratings_info(ratings_df)

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
    
    # model
    model_class = import_class_by_name(cfg.model_class)
    model = model_class(**cfg.model_kwargs)
    print(f"Model: {cfg.model_class}")
    print(f"Model parameters :\n{cfg.model_kwargs}")
    
    # train
    train_start = time.time()
    model.fit(dataset.build_full_trainset())
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Model trained in {format_time(train_time)}")


    # test
    # TODO: split data into train and test
    # save evaluation metrics
        
    # save
    save_model(cfg.model_save_path, model)
    print(f"Model saved to {cfg.model_save_path}")


if __name__ == "__main__":
    main()
