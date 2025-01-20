import importlib
import math
import pickle
import sys
import time
from collections import defaultdict
from functools import wraps

from omegaconf import OmegaConf


def save_model(f: str, model):
    obj = {"model": model}
    pickle.dump(obj, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_model(f: str):
    obj = pickle.load(open(f, "rb"))
    return obj["model"]


def format_time(seconds):
    return time.strftime("%Hh %Mm %Ss", time.gmtime(seconds))


def format_number(number):
    exponent = math.floor(math.log10(number)) // 3
    suffix = ["", "k", "M", "B", "T", "Q"]
    short_number = number / (10 ** (3 * exponent))
    return f"{short_number:.1f}{suffix[exponent]}"



def get_top_k(predictions, k=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_k.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k[uid] = user_ratings[:k]

    return top_k

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls




def load_config(default_config_path: str):
    """
    Load config from default path, merge it with custom config (if provided) and CLI arguments.
    Then call the decorated function with the config as an only argument.
    """

    def _is_yaml_file(file_path: str) -> bool:
        """Check if the file is a YAML file."""
        return file_path.endswith(".yaml") or file_path.endswith(".yml")

    def decorator(func):
        @wraps(func)
        def wrapper():
            config = OmegaConf.load(default_config_path)

            args = sys.argv[1:]
            if len(args) > 0 and _is_yaml_file(args[0]):
                custom_config_path = args.pop(0)
                print(custom_config_path)
                custom_config = OmegaConf.load(custom_config_path)
                config = OmegaConf.merge(config, custom_config)

            if len(args) > 0:
                cli_config = OmegaConf.from_cli(args)
                config = OmegaConf.merge(config, cli_config)

            func(config)

        return wrapper

    return decorator


def import_class_by_name(class_name: str):
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

