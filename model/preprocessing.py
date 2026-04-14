import pandas as pd

from model.features import add_distance_features, add_time_features, add_trip_type_features


def preprocess_data(X):
    print("Preprocessing data")
    X = X.copy()

    pickup_datetime = pd.to_datetime(X["pickup_datetime"])
    X["pickup_hour"] = pickup_datetime.dt.hour
    X["pickup_day"] = pickup_datetime.dt.day
    X["pickup_month"] = pickup_datetime.dt.month
    X["pickup_weekday"] = pickup_datetime.dt.weekday

    X["store_and_fwd_flag"] = X["store_and_fwd_flag"].map({"N": 0, "Y": 1})

    drop_columns = ["id", "pickup_datetime", "dropoff_datetime"]
    X = X.drop(columns=drop_columns, errors="ignore")

    return X


def transform_target(y):
    import numpy as np

    return np.log1p(y).rename("log_" + y.name)


def undo_step3_process_features(X):
    X = X.copy()
    X = X.drop(columns=["vendor_id", "store_and_fwd_flag", "passenger_count"], errors="ignore")
    return X


def add_model_features(X, abnormal_dates=None):
    X = add_time_features(X, abnormal_dates)
    X = add_distance_features(X)
    X = add_trip_type_features(X)
    X = undo_step3_process_features(X)
    return X