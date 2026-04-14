from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_train_data, load_test_data
from model.features import build_abnormal_dates
from model.preprocessing import add_model_features, transform_target

import common

MODEL_PATH = common.CONFIG["paths"]["model_ridge_path"]

NUM_FEATURES = [
    "log_distance_haversine",
    "hour",
    "hour_sin",
    "hour_cos",
    "abnormal_period",
    "is_weekend",
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_high_traffic_trip",
    "is_high_speed_trip",
    "is_rare_pickup_point",
    "is_rare_dropoff_point",
]
CAT_FEATURES = ["weekday", "month", "time_bucket"]
TRAIN_FEATURES = NUM_FEATURES + CAT_FEATURES


def build_model():
    column_transformer = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("scaling", StandardScaler(), NUM_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("ohe_and_scaling", column_transformer),
            ("regression", Ridge()),
        ]
    )
    return pipeline


def train_model():
    print("Building ridge model")

    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    abnormal_dates = build_abnormal_dates(X_train)

    X_train = add_model_features(X_train, abnormal_dates)
    X_test = add_model_features(X_test, abnormal_dates)

    y_train = transform_target(y_train)
    y_test = transform_target(y_test)

    model = build_model()
    model.fit(X_train[TRAIN_FEATURES], y_train)

    y_pred_train = model.predict(X_train[TRAIN_FEATURES])
    y_pred_test = model.predict(X_test[TRAIN_FEATURES])

    print(f"Train RMSLE = {root_mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"Test RMSLE = {root_mean_squared_error(y_test, y_pred_test):.4f}")
    print(f"Test R2 = {r2_score(y_test, y_pred_test):.4f}")

    return model


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
    persist_model(model, MODEL_PATH)