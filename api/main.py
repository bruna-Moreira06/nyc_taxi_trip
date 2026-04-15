from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import API_CONFIG
from model.preprocessing import preprocess_data


class TripInput(BaseModel):
    id: str = Field(..., description="Unique trip identifier")
    vendor_id: int = Field(..., ge=1, description="Taxi vendor identifier")
    pickup_datetime: str = Field(..., description="Pickup timestamp (ISO 8601)")
    dropoff_datetime: str = Field(..., description="Dropoff timestamp (ISO 8601)")
    passenger_count: int = Field(..., ge=1, le=8)
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str = Field(..., pattern="^(Y|N)$")


class PredictRequest(BaseModel):
    trips: List[TripInput] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    id: str
    predicted_trip_duration: float


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]
    persisted_count: int


class TaxiPredictionWrapper:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        with self.model_path.open("rb") as file:
            return pickle.load(file)

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
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_predictions(self, records: Iterable[dict], predictions: Iterable[float]) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(record.get("id", "")),
                float(prediction),
                json.dumps(record),
                timestamp,
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
                    created_at
                ) VALUES (?, ?, ?, ?)
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


@app.on_event("startup")
def startup_event() -> None:
    global model_wrapper, prediction_repository, model_load_error

    model_path = API_CONFIG["paths"]["model_path"]
    db_path = API_CONFIG["paths"]["db_path"]

    prediction_repository = PredictionRepository(db_path=db_path)
    try:
        model_wrapper = TaxiPredictionWrapper(model_path=model_path)
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


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if prediction_repository is None:
        raise HTTPException(status_code=503, detail="Prediction repository not initialized")
    if model_wrapper is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model service not initialized",
        )

    records = [trip.model_dump() for trip in payload.trips]

    try:
        predictions = model_wrapper.predict(records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    persisted_count = prediction_repository.save_predictions(records, predictions)

    response_items = [
        PredictionItem(id=record["id"], predicted_trip_duration=prediction)
        for record, prediction in zip(records, predictions)
    ]

    return PredictResponse(predictions=response_items, persisted_count=persisted_count)
