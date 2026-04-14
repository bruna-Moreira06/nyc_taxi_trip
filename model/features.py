import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def build_abnormal_dates(X, min_daily_trips=6300):
    pickup_datetime = pd.to_datetime(X["pickup_datetime"])
    pickup_date = pickup_datetime.dt.date.rename("pickup_date")
    trip_count_by_date = pickup_date.groupby(pickup_date).count()
    abnormal_dates = trip_count_by_date[trip_count_by_date < min_daily_trips]
    return abnormal_dates.index


def add_time_features(X, abnormal_dates=None):
    X = X.copy()
    pickup_datetime = pd.to_datetime(X["pickup_datetime"])
    pickup_dates = pickup_datetime.dt.normalize()
    holiday_dates = get_us_holiday_dates(pickup_dates.min(), pickup_dates.max())

    X["weekday"] = pickup_datetime.dt.weekday
    X["month"] = pickup_datetime.dt.month
    X["hour"] = pickup_datetime.dt.hour
    X["is_weekend"] = (X["weekday"] >= 5).astype(int)
    X["time_bucket"] = pickup_datetime.dt.hour.map(get_time_bucket)
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["is_holiday"] = pickup_dates.isin(holiday_dates).astype(int)
    X["is_pre_holiday"] = (pickup_dates + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    X["is_post_holiday"] = (pickup_dates - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)

    if abnormal_dates is None:
        X["abnormal_period"] = 0
    else:
        X["abnormal_period"] = pickup_datetime.dt.date.isin(abnormal_dates).astype(int)

    return X


def get_us_holiday_dates(start_date, end_date):
    # Use an official US federal holiday calendar to keep the feature deterministic.
    calendar = USFederalHolidayCalendar()
    return calendar.holidays(start=start_date, end=end_date)


def get_time_bucket(hour):
    # Keep buckets coarse so the linear model can learn stable traffic regimes.
    if 6 <= hour <= 9:
        return "morning_peak"
    if 10 <= hour <= 15:
        return "midday"
    if 16 <= hour <= 19:
        return "evening_peak"
    if 20 <= hour <= 23:
        return "night"
    return "late_night"


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    avg_earth_radius = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * avg_earth_radius * np.arcsin(np.sqrt(d))
    return h


def is_high_traffic_trip(X):
    return (
        ((X["hour"] >= 8) & (X["hour"] <= 19) & (X["weekday"] >= 0) & (X["weekday"] <= 4))
        | ((X["hour"] >= 13) & (X["hour"] <= 20) & (X["weekday"] == 5))
    )


def is_high_speed_trip(X):
    return (
        ((X["hour"] >= 2) & (X["hour"] <= 5) & (X["weekday"] >= 0) & (X["weekday"] <= 4))
        | ((X["hour"] >= 4) & (X["hour"] <= 7) & (X["weekday"] >= 5) & (X["weekday"] <= 6))
    )


def is_rare_point(X, latitude_column, longitude_column, qmin_lat, qmax_lat, qmin_lon, qmax_lon):
    lat_min = X[latitude_column].quantile(qmin_lat)
    lat_max = X[latitude_column].quantile(qmax_lat)
    lon_min = X[longitude_column].quantile(qmin_lon)
    lon_max = X[longitude_column].quantile(qmax_lon)

    return (
        (X[latitude_column] < lat_min)
        | (X[latitude_column] > lat_max)
        | (X[longitude_column] < lon_min)
        | (X[longitude_column] > lon_max)
    )


def add_distance_features(X):
    X = X.copy()
    distance_haversine = haversine_array(
        X["pickup_latitude"],
        X["pickup_longitude"],
        X["dropoff_latitude"],
        X["dropoff_longitude"],
    )
    X["log_distance_haversine"] = np.log1p(distance_haversine)
    return X


def add_trip_type_features(X):
    X = X.copy()
    X["is_high_traffic_trip"] = is_high_traffic_trip(X).astype(int)
    X["is_high_speed_trip"] = is_high_speed_trip(X).astype(int)
    X["is_rare_pickup_point"] = is_rare_point(
        X, "pickup_latitude", "pickup_longitude", 0.01, 0.995, 0, 0.95
    ).astype(int)
    X["is_rare_dropoff_point"] = is_rare_point(
        X, "dropoff_latitude", "dropoff_longitude", 0.01, 0.995, 0.005, 0.95
    ).astype(int)
    return X