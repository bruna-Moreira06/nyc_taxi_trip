from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_train_data, load_test_data
from model.preprocessing import preprocess_data

import common

MODEL_PATH = common.CONFIG["paths"]["model_path"]


def train_model():
    print("Building a model")

    X_train, y_train = load_train_data()

    model = LinearRegression()
    X_train_preprocessed = preprocess_data(X_train)
    model.fit(X_train_preprocessed, y_train)

    y_pred = model.predict(X_train_preprocessed)
    score = mean_squared_error(y_train, y_pred)
    print(f"Score on train data {score:.2f}")

    return model


def evaluate_model(model):
    print("Evaluating the model")

    X_test, y_test = load_test_data()
    X_test_preprocessed = preprocess_data(X_test)
    y_pred = model.predict(X_test_preprocessed)
    score = mean_squared_error(y_test, y_pred)
    print(f"Score on test data {score:.2f}")

    return score


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print("Done")


if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)
    persist_model(model, MODEL_PATH)