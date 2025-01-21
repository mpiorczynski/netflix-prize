import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from omegaconf import OmegaConf

from netflix_prize.data import load_movies
from netflix_prize.recommend_user import build_testset_anti_testset
from netflix_prize.utils import load_model


@st.cache_resource
def find_model_files(directory, pattern="**/model.pkl"):
    """Recursively search for model.pkl files in the given directory."""
    return glob.glob(os.path.join(directory, pattern), recursive=True)


@st.cache_resource
def load_configuration(config_path="configs/default.yaml"):
    """Load the configuration file."""
    return OmegaConf.load(config_path)


@st.cache_resource
def load_movies_cached(movies_file):
    """Load movies data and preprocess movie titles."""
    movies_df = load_movies(movies_file)
    movie_id_to_title = movies_df.set_index("movie_id").to_dict("index")
    movie_id_to_title = {
        k: f"{v['title']} ({int(v['year']) if not np.isnan(v['year']) else '-'})"
        for k, v in movie_id_to_title.items()
    }
    return movie_id_to_title


@st.cache_resource
def load_model_cached(model_path):
    """Load a machine learning model."""
    return load_model(model_path)


st.title("Movie Recommendation System")

# Load configuration and movies
cfg = load_configuration()
logger.info("Loading available movies...")
movie_id_to_title = load_movies_cached(cfg.data.movies_file)

# Display model options
model_options = find_model_files(cfg.output_dir)
selected_model = st.sidebar.selectbox("Select a Model", model_options)

if st.sidebar.button("Load Model"):
    if selected_model:
        logger.info(f"Loading model from {selected_model}")
        model = load_model_cached(selected_model)

        # Load model metrics
        with open(os.path.join(os.path.dirname(selected_model), "metrics.json")) as f:
            metrics = json.load(f)
        st.sidebar.success("Model Loaded Successfully!")
        st.subheader("Model Metrics:")
        st.write(metrics)

        user_id = st.sidebar.number_input("Select User ID", min_value=1, value=440949)
        if st.sidebar.button("Recommend Movies"):
            logger.info("Looking for the movies rated by the user...")

            st.cache_resource

            def build_testset_anti_testset_cached(user_id):
                return build_testset_anti_testset(model.trainset, user_id)

            testset, anti_testset = build_testset_anti_testset_cached(user_id)

            # Display liked movies
            liked_movies = [(movie_id, r_true) for user_id, movie_id, r_true in testset]
            liked_movies = pd.DataFrame(liked_movies, columns=["movie_id", "rating"])
            liked_movies = liked_movies.sort_values("rating", ascending=False).head(10)
            liked_movies["title"] = liked_movies["movie_id"].map(movie_id_to_title)
            logger.info(liked_movies)
            st.write("Movies rated highly by the user:", liked_movies)

            # Display recommended movies
            logger.info("Predicting ratings for unrated movies...")
            predictions = model.test(anti_testset)

            recommend_movies = [
                (movie_id, r_est) for user_id, movie_id, r_true, r_est, _ in predictions
            ]
            recommend_movies = pd.DataFrame(
                recommend_movies, columns=["movie_id", "rating"]
            )
            recommend_movies = recommend_movies.sort_values(
                "rating", ascending=False
            ).head(10)
            recommend_movies["title"] = recommend_movies["movie_id"].map(
                movie_id_to_title
            )
            logger.info(recommend_movies)
            st.write("Recommended movies:", recommend_movies)
