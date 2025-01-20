import importlib
import math
import os
import pickle
import sys
import time
from functools import wraps

from omegaconf import OmegaConf


def save_model(dir: str, model, name="model.pkl"):
    path = os.path.join(dir, name)
    obj = {"model": model}
    pickle.dump(obj, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str):
    obj = pickle.load(open(path, "rb"))
    return obj["model"]


def format_time(seconds):
    return time.strftime("%Hh %Mm %Ss", time.gmtime(seconds))


def format_number(number):
    exponent = math.floor(math.log10(number)) // 3
    suffix = ["", "k", "M", "B", "T", "Q"]
    short_number = number / (10 ** (3 * exponent))
    return f"{short_number:.1f}{suffix[exponent]}"


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
                print(f"Config path: {custom_config_path}")
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
