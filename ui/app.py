from __future__ import annotations

import sqlite3
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NYC Taxi Trip UI", page_icon="🚕", layout="wide")

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_DB_PATH = str((Path(__file__).resolve().parents[1] / "data" / "taxi_trip_duration.db").resolve())
REQUIRED_BATCH_COLUMNS = [
    "id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
]


def _call_api(method: str, url: str, payload: dict[str, Any] | None) -> tuple[int, Any]:
    response = requests.request(method=method, url=url, json=payload, timeout=30)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def _to_iso_datetime(d: date, t: time) -> str:
    return datetime.combine(d, t).isoformat()


def _build_single_payload() -> dict[str, Any]:
    col1, col2, col3 = st.columns(3)

    default_pickup_date = date(2016, 3, 14)
    default_pickup_time = time(17, 24, 55)
    default_dropoff_date = date(2016, 3, 14)
    default_dropoff_time = time(17, 32, 30)

    with col1:
        trip_id = st.text_input("Trip ID", value="id_ui_001")
        vendor_id = st.number_input("Vendor ID", min_value=1, value=1, step=1)
        passenger_count = st.number_input("Passenger count", min_value=1, max_value=8, value=1, step=1)
        store_and_fwd_flag = st.selectbox("Store and fwd flag", options=["N", "Y"], index=0)
    with col2:
        pickup_date = st.date_input("Pickup date", value=default_pickup_date)
        pickup_time = st.time_input("Pickup time", value=default_pickup_time)
        pickup_longitude = st.slider(
            "Pickup longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-73.9821548461914,
            step=0.000001,
            format="%.6f",
        )
        pickup_latitude = st.slider(
            "Pickup latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.76793670654297,
            step=0.000001,
            format="%.6f",
        )
    with col3:
        dropoff_date = st.date_input("Dropoff date", value=default_dropoff_date)
        dropoff_time = st.time_input("Dropoff time", value=default_dropoff_time)
        dropoff_longitude = st.slider(
            "Dropoff longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-73.96463012695312,
            step=0.000001,
            format="%.6f",
        )
        dropoff_latitude = st.slider(
            "Dropoff latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.765602111816406,
            step=0.000001,
            format="%.6f",
        )

    return {
        "id": trip_id,
        "vendor_id": int(vendor_id),
        "pickup_datetime": _to_iso_datetime(pickup_date, pickup_time),
        "dropoff_datetime": _to_iso_datetime(dropoff_date, dropoff_time),
        "passenger_count": int(passenger_count),
        "pickup_longitude": float(pickup_longitude),
        "pickup_latitude": float(pickup_latitude),
        "dropoff_longitude": float(dropoff_longitude),
        "dropoff_latitude": float(dropoff_latitude),
        "store_and_fwd_flag": store_and_fwd_flag,
    }


def _extract_error_message(status_code: int, body: Any) -> str:
    if status_code == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        messages: list[str] = []
        for item in body["detail"]:
            location = " -> ".join(str(part) for part in item.get("loc", []))
            message = item.get("msg", "Invalid input")
            if location:
                messages.append(f"{location}: {message}")
            else:
                messages.append(str(message))
        return "\n".join(messages)

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        return str(body)

    return str(body)


@st.cache_data(show_spinner=False)
def _load_distributions(db_path: str) -> tuple[pd.Series, pd.Series]:
    with sqlite3.connect(db_path) as connection:
        train_df = pd.read_sql("SELECT trip_duration FROM train", connection)
        pred_df = pd.read_sql("SELECT predicted_trip_duration FROM predictions", connection)

    train_series = train_df["trip_duration"].dropna().astype(float)
    pred_series = pred_df["predicted_trip_duration"].dropna().astype(float)
    return train_series, pred_series


def _render_prediction_page(api_base_url: str, model_version: str) -> None:
    st.subheader("Single prediction")
    payload = _build_single_payload()

    map_df = pd.DataFrame(
        [
            {
                "point": "pickup",
                "lat": payload["pickup_latitude"],
                "lon": payload["pickup_longitude"],
            },
            {
                "point": "dropoff",
                "lat": payload["dropoff_latitude"],
                "lon": payload["dropoff_longitude"],
            },
        ]
    )
    st.caption("Pickup and dropoff locations")
    st.map(map_df, latitude="lat", longitude="lon")
    st.dataframe(map_df, use_container_width=True, hide_index=True)

    if st.button("Predict", type="primary"):
        endpoint = f"{api_base_url}/predict"
        if model_version.strip():
            endpoint += f"?model_version={model_version.strip()}"

        try:
            status_code, body = _call_api("POST", endpoint, payload)
        except requests.RequestException as exc:
            st.error(
                "Unable to reach the API. Check that the FastAPI server is running and the URL is correct."
            )
            st.caption(str(exc))
            return

        if 200 <= status_code < 300 and isinstance(body, dict):
            prediction = body.get("prediction", {})
            predicted_value = prediction.get("predicted_trip_duration")
            used_model_version = body.get("model_version", "unknown")

            st.success("Prediction completed.")
            if predicted_value is not None:
                st.metric("Predicted trip duration (seconds)", f"{float(predicted_value):.2f}")
            st.write(f"Model version used: {used_model_version}")
            st.caption(f"HTTP status: {status_code}")
        else:
            st.error("Prediction failed. Please check the input values and try again.")
            st.caption(_extract_error_message(status_code, body))
            with st.expander("Technical details"):
                st.write(f"HTTP status: {status_code}")
                st.json(body)

    st.divider()
    st.subheader("Batch prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to run batch predictions.")
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error("Unable to read CSV file.")
        st.caption(str(exc))
        return

    missing_columns = [col for col in REQUIRED_BATCH_COLUMNS if col not in batch_df.columns]
    if missing_columns:
        st.error("CSV is missing required columns for prediction.")
        st.caption("Missing: " + ", ".join(missing_columns))
        return

    st.caption(f"Detected {len(batch_df):,} rows in uploaded file.")
    st.dataframe(batch_df.head(10), use_container_width=True)

    if st.button("Run batch prediction", type="primary"):
        if batch_df.empty:
            st.warning("Uploaded CSV has no rows.")
            return

        records_df = batch_df[REQUIRED_BATCH_COLUMNS].copy()
        records_df = records_df.replace({np.nan: None})
        payload_batch = {"trips": records_df.to_dict(orient="records")}

        endpoint = f"{api_base_url}/predict_batch"
        if model_version.strip():
            endpoint += f"?model_version={model_version.strip()}"

        try:
            status_code, body = _call_api("POST", endpoint, payload_batch)
        except requests.RequestException as exc:
            st.error(
                "Unable to reach the API. Check that the FastAPI server is running and the URL is correct."
            )
            st.caption(str(exc))
            return

        if not (200 <= status_code < 300 and isinstance(body, dict)):
            st.error("Batch prediction failed. Please check CSV values and try again.")
            st.caption(_extract_error_message(status_code, body))
            with st.expander("Technical details"):
                st.write(f"HTTP status: {status_code}")
                st.json(body)
            return

        predictions = body.get("predictions", [])
        durations = [item.get("predicted_trip_duration") for item in predictions]
        output_df = batch_df.copy()

        if len(durations) == len(output_df):
            output_df["predicted_trip_duration"] = durations
        else:
            st.warning(
                "Prediction count does not match input row count. Appending available predictions by order."
            )
            output_df["predicted_trip_duration"] = pd.Series(durations)

        used_model_version = body.get("model_version", "unknown")
        st.success("Batch prediction completed.")
        st.write(f"Model version used: {used_model_version}")
        st.caption(f"HTTP status: {status_code}")

        st.dataframe(output_df, use_container_width=True)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="predictions_with_duration.csv",
            mime="text/csv",
        )


def _render_statistics_page(db_path: str) -> None:
    st.subheader("Data Distribution Statistics")
    st.caption("Compare training target distribution against predicted values distribution.")

    try:
        train_series, pred_series = _load_distributions(db_path)
    except sqlite3.Error as exc:
        st.error("Could not read the SQLite database.")
        st.caption(str(exc))
        return
    except Exception as exc:
        st.error("Could not build statistics from database tables.")
        st.caption(str(exc))
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train samples", f"{len(train_series):,}")
        if len(train_series) > 0:
            st.caption(
                f"Mean: {train_series.mean():.2f} | Median: {train_series.median():.2f}"
            )
    with col2:
        st.metric("Predicted samples", f"{len(pred_series):,}")
        if len(pred_series) > 0:
            st.caption(f"Mean: {pred_series.mean():.2f} | Median: {pred_series.median():.2f}")

    if len(train_series) == 0:
        st.warning("No training values found in table 'train'.")
        return
    if len(pred_series) == 0:
        st.info("No predicted values found yet. Make predictions first to populate the histogram.")
        return

    combined_min = min(float(train_series.min()), float(pred_series.min()))
    combined_max = max(float(train_series.max()), float(pred_series.max()))
    if combined_min == combined_max:
        combined_max = combined_min + 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    bin_edges = np.linspace(combined_min, combined_max, 41).tolist()
    ax.hist(train_series, bins=bin_edges, alpha=0.55, label="Training target", color="#1f77b4")
    ax.hist(pred_series, bins=bin_edges, alpha=0.55, label="Predicted values", color="#ff7f0e")
    ax.set_title("Training vs Predicted Trip Duration Distributions")
    ax.set_xlabel("Trip duration (seconds)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.title("NYC Taxi Trip Duration UI")
    st.caption("UI decoupled from training: this app only calls the prediction API.")

    with st.sidebar:
        page = st.radio("Page", options=["Predict", "Statistics"], index=0)
        st.header("Settings")
        api_base_url = st.text_input("Base URL", value=DEFAULT_API_URL).rstrip("/")
        model_version = st.text_input("Model version (optional)", value="")
        db_path = st.text_input("SQLite DB path", value=DEFAULT_DB_PATH)

    if page == "Predict":
        _render_prediction_page(api_base_url=api_base_url, model_version=model_version)
    else:
        _render_statistics_page(db_path=db_path)


if __name__ == "__main__":
    main()
