from pathlib import Path
from tensorflow import keras


def get_model(model_path: str) -> keras.Model:
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))
