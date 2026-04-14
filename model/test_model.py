import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_random_test_data
from model.preprocessing import preprocess_data

import common

MODEL_PATH = common.CONFIG["paths"]["model_path"]
TARGET_COLUMN = common.CONFIG["dataset"]["target_column"]


def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print("Done")
    return model


def test_model(model):
    print("Testing the model")
    X, y = load_random_test_data()
    X_preprocessed = preprocess_data(X)
    y_pred = model.predict(X_preprocessed)
    df = X.copy()
    df[TARGET_COLUMN] = y
    df["prediction"] = y_pred
    print(df)


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_model(model)