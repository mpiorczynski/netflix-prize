import json
import os
import time

from omegaconf import DictConfig, OmegaConf
from surprise import Dataset, Reader
from surprise import accuracy as surprise_metrics

from netflix_prize.data import load_probe, load_ratings, train_test_split
from netflix_prize.metrics import precision_recall_at_k
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


@load_config("configs/default.yaml")
def main(cfg: DictConfig):
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    cfg.output_dir = os.path.join(cfg.output_dir, timestamp)

    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    # model
    model_class = import_class_by_name(cfg.model.class_name)
    model = model_class(**cfg.model.hparams)

    # data
    print("Loading ratings...")
    load_start = time.time()
    ratings_df = load_ratings(cfg.data.rating_files, cfg.data.chunk_size)
    load_end = time.time()
    load_time = load_end - load_start
    print(f"Ratings loaded in {format_time(load_time)}")
    print_ratings_info(ratings_df)

    print("Splitting data...")
    probe_df = load_probe(cfg.data.probe_file, cfg.data.chunk_size)
    train_df, test_df = train_test_split(ratings_df, probe_df)
    print(f"Train set: {train_df.shape[0]} ratings")  # 99,072,112
    print(f"Test set: {test_df.shape[0]} ratings")  # 1,408,395
    del ratings_df, probe_df  # free memory

    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(
        train_df[["user_id", "movie_id", "rating"]], reader
    ).build_full_trainset()
    del train_df  # free memory

    # train
    print("Training...")
    train_start = time.time()
    model.fit(trainset)
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Model trained in {format_time(train_time)}")
    del trainset  # free memory

    # save
    save_model(cfg.output_dir, model)
    print(f"Model saved to {cfg.output_dir}")

    # test
    print("Evaluating...")
    testset = (
        Dataset.load_from_df(test_df[["user_id", "movie_id", "rating"]], reader)
        .build_full_trainset()
        .build_testset()
    )
    del test_df  # free memory
    predictions = model.test(testset)
    rmse = surprise_metrics.rmse(predictions)
    mae = surprise_metrics.mae(predictions)
    at_k = 5
    precision_at_k, recall_at_k = precision_recall_at_k(predictions, k=at_k, threshold=4)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        f"precision@{at_k}": precision_at_k,
        f"recall@{at_k}": recall_at_k,
    }
    print(f"Metrics:\n{metrics}")
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print("Done.")


if __name__ == "__main__":
    main()
