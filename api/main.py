from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import ClassVar, Iterable, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import API_CONFIG
from model.preprocessing import preprocess_data


class TripInput(BaseModel):
    NYC_MIN_LONGITUDE: ClassVar[float] = -74.25909
    NYC_MAX_LONGITUDE: ClassVar[float] = -73.70018
    NYC_MIN_LATITUDE: ClassVar[float] = 40.47740
    NYC_MAX_LATITUDE: ClassVar[float] = 40.91758
    MIN_TRIP_DISTANCE_METERS: ClassVar[float] = 50.0

    id: str = Field(..., description="Unique trip identifier")
    vendor_id: int = Field(..., ge=1, description="Taxi vendor identifier")
    pickup_datetime: str = Field(..., description="Pickup timestamp (ISO 8601)")
    dropoff_datetime: str = Field(..., description="Dropoff timestamp (ISO 8601)")
    passenger_count: int = Field(..., ge=1, le=8)
    pickup_longitude: float = Field(..., ge=-180, le=180)
    pickup_latitude: float = Field(..., ge=-90, le=90)
    dropoff_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: float = Field(..., ge=-90, le=90)
    store_and_fwd_flag: str = Field(..., pattern="^(Y|N)$")

    @model_validator(mode="after")
    def validate_geo_constraints(self) -> "TripInput":
        if not self._is_in_nyc_bbox(self.pickup_longitude, self.pickup_latitude):
            raise ValueError("pickup coordinates must be inside NYC bounding box")
        if not self._is_in_nyc_bbox(self.dropoff_longitude, self.dropoff_latitude):
            raise ValueError("dropoff coordinates must be inside NYC bounding box")

        distance_m = self._haversine_meters(
            self.pickup_latitude,
            self.pickup_longitude,
            self.dropoff_latitude,
            self.dropoff_longitude,
        )
        if distance_m <= self.MIN_TRIP_DISTANCE_METERS:
            raise ValueError("pickup and dropoff points must be more than 50 meters apart")

        return self

    @classmethod
    def _is_in_nyc_bbox(cls, longitude: float, latitude: float) -> bool:
        return (
            cls.NYC_MIN_LONGITUDE <= longitude <= cls.NYC_MAX_LONGITUDE
            and cls.NYC_MIN_LATITUDE <= latitude <= cls.NYC_MAX_LATITUDE
        )

    @staticmethod
    def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return r * c


class PredictRequest(BaseModel):
    trips: List[TripInput] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    id: str
    predicted_trip_duration: float


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]
    model_version: str
    persisted_count: int


class SinglePredictResponse(BaseModel):
    prediction: PredictionItem
    model_version: str
    persisted_count: int


class TaxiPredictionWrapper:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.model_version = self._build_model_version()
        self.model_created_at = datetime.fromtimestamp(
            self.model_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        with self.model_path.open("rb") as file:
            return pickle.load(file)

    def _build_model_version(self) -> str:
        stat = self.model_path.stat()
        return f"{stat.st_size}-{stat.st_mtime_ns}"

    def preprocess(self, records: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return preprocess_data(df)

    def postprocess(self, predictions: np.ndarray) -> list[float]:
        cleaned = np.maximum(predictions, 0)
        return [round(float(value), 2) for value in cleaned]

    def predict(self, records: Iterable[dict]) -> list[float]:
        features = self.preprocess(records)
        raw_predictions = self.model.predict(features)
        return self.postprocess(raw_predictions)


class PredictionRepository:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trip_id TEXT NOT NULL,
                    predicted_trip_duration REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    inference_timestamp TEXT,
                    model_version TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_version TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_created_at TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "predictions", "inference_timestamp", "TEXT")
            self._ensure_column(connection, "predictions", "model_version", "TEXT")
            connection.commit()

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_names = {column[1] for column in columns}
        if column_name not in existing_names:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )

    def save_model_metadata(self, model_version: str, model_path: str, model_created_at: str) -> None:
        registered_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_metadata (
                    model_version,
                    model_path,
                    model_created_at,
                    registered_at
                ) VALUES (?, ?, ?, ?)
                """,
                (model_version, model_path, model_created_at, registered_at),
            )
            connection.commit()

    def get_latest_model_version(self) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_version
                FROM model_metadata
                ORDER BY registered_at DESC
                LIMIT 1
                """
            ).fetchone()
        return None if row is None else str(row[0])

    def get_model_metadata(self, model_version: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_version, model_path, model_created_at, registered_at
                FROM model_metadata
                WHERE model_version = ?
                """,
                (model_version,),
            ).fetchone()
        if row is None:
            return None
        return {
            "model_version": str(row[0]),
            "model_path": str(row[1]),
            "model_created_at": str(row[2]),
            "registered_at": str(row[3]),
        }

    def save_predictions(
        self,
        records: Iterable[dict],
        predictions: Iterable[float],
        model_version: str,
    ) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(record.get("id", "")),
                float(prediction),
                json.dumps(record),
                timestamp,
                timestamp,
                model_version,
            )
            for record, prediction in zip(records, predictions)
        ]

        if not rows:
            return 0

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO predictions (
                    trip_id,
                    predicted_trip_duration,
                    payload_json,
                    created_at,
                    inference_timestamp,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

        return len(rows)

app = FastAPI(
    title=API_CONFIG["app"]["title"],
    version=API_CONFIG["app"]["version"],
)

model_wrapper: TaxiPredictionWrapper | None = None
prediction_repository: PredictionRepository | None = None
model_load_error: str | None = None
model_wrappers_by_version: dict[str, TaxiPredictionWrapper] = {}


def resolve_wrapper(requested_model_version: str | None) -> tuple[TaxiPredictionWrapper, str]:
    if prediction_repository is None:
        raise HTTPException(status_code=503, detail="Prediction repository not initialized")
    if model_wrapper is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model service not initialized",
        )

    version_to_use = requested_model_version or prediction_repository.get_latest_model_version()
    if version_to_use is None:
        version_to_use = model_wrapper.model_version

    if version_to_use == model_wrapper.model_version:
        return model_wrapper, version_to_use

    cached_wrapper = model_wrappers_by_version.get(version_to_use)
    if cached_wrapper is not None:
        return cached_wrapper, version_to_use

    metadata = prediction_repository.get_model_metadata(version_to_use)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{version_to_use}' not found",
        )

    candidate_wrapper = TaxiPredictionWrapper(model_path=metadata["model_path"])
    if candidate_wrapper.model_version != version_to_use:
        raise HTTPException(
            status_code=409,
            detail=(
                "Model metadata exists but current file does not match requested model version"
            ),
        )

    model_wrappers_by_version[version_to_use] = candidate_wrapper
    return candidate_wrapper, version_to_use


@app.on_event("startup")
def startup_event() -> None:
    global model_wrapper, prediction_repository, model_load_error, model_wrappers_by_version

    model_path = API_CONFIG["paths"]["model_path"]
    db_path = API_CONFIG["paths"]["db_path"]

    prediction_repository = PredictionRepository(db_path=db_path)
    try:
        model_wrapper = TaxiPredictionWrapper(model_path=model_path)
        model_wrappers_by_version = {model_wrapper.model_version: model_wrapper}
        prediction_repository.save_model_metadata(
            model_version=model_wrapper.model_version,
            model_path=str(model_wrapper.model_path),
            model_created_at=model_wrapper.model_created_at,
        )
        model_load_error = None
    except FileNotFoundError as exc:
        model_wrapper = None
        model_load_error = str(exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model_wrapper is not None,
        "model_error": model_load_error,
    }


@app.post("/predict", response_model=SinglePredictResponse)
def predict(
    payload: TripInput,
    model_version: str | None = Query(
        default=None,
        description="Model version to use. If omitted, latest available version is used.",
    ),
) -> SinglePredictResponse:
    active_wrapper, active_model_version = resolve_wrapper(model_version)
    records = [payload.model_dump()]

    try:
        predictions = active_wrapper.predict(records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    persisted_count = prediction_repository.save_predictions(
        records=records,
        predictions=predictions,
        model_version=active_model_version,
    )

    return SinglePredictResponse(
        prediction=PredictionItem(
            id=records[0]["id"],
            predicted_trip_duration=predictions[0],
        ),
        model_version=active_model_version,
        persisted_count=persisted_count,
    )


@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(
    payload: PredictRequest,
    model_version: str | None = Query(
        default=None,
        description="Model version to use. If omitted, latest available version is used.",
    ),
) -> PredictResponse:
    active_wrapper, active_model_version = resolve_wrapper(model_version)
    records = [trip.model_dump() for trip in payload.trips]

    try:
        predictions = active_wrapper.predict(records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    persisted_count = prediction_repository.save_predictions(
        records=records,
        predictions=predictions,
        model_version=active_model_version,
    )

    response_items = [
        PredictionItem(id=record["id"], predicted_trip_duration=prediction)
        for record, prediction in zip(records, predictions)
    ]

    return PredictResponse(
        predictions=response_items,
        model_version=active_model_version,
        persisted_count=persisted_count,
    )
