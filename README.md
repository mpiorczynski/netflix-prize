# netflix-prize

## Installation
```bash
conda env create -f environment.yml
conda activate netflix-prize
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pre-commit install
```

## Data
The data is available at [Netflix Prize](https://www.kaggle.com/netflix-inc/netflix-prize-data).

## Usage

### Training
Change the configuration file in `configs/` to the desired model and hyperparameters. Then run the following command:
```bash
python netflix_prize/train.py configs/centered_knn.yaml
```

### Recommendations
```bash
python netflix_prize/recommend_movie.py model_path="checkpoints/centered_knn/model.pkl" recommend.movie_title="Toy Story"
```

```bash
python netflix_prize/recommend_user.py model_path="checkpoints/centered_knn/model.pkl" recommend.user_id=1377724
```


## References
[1] https://en.wikipedia.org/wiki/Netflix_Prize
