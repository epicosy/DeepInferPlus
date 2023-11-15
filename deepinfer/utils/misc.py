import pandas as pd

from deepinfer.utils.paths import models_path, datasets_path
from tensorflow import keras
from dataclasses import dataclass


@dataclass
class DatasetSplit:
    name: str
    file: str


@dataclass
class TrainSplit(DatasetSplit):
    name: str = 'train'
    file: str = 'x.csv'


@dataclass
class TestSplit(DatasetSplit):
    name: str = 'val'
    file: str = 'x.csv'


@dataclass
class UnseenSplit(DatasetSplit):
    name: str = 'unseen'
    file: str = 'unseen.csv'


def get_model(model: str, version: str) -> keras.Model:
    model_path = models_path / f"{model}{version}.h5"

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))


def get_dataset(dataset: str, split: DatasetSplit) -> pd.DataFrame:
    dataset_path = datasets_path / dataset / split.name / split.file

    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist")

    return pd.read_csv(str(dataset_path), delimiter=',', encoding='utf-8')
