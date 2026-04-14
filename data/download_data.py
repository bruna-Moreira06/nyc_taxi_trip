"""
Charge les fichiers CSV bruts dans la base SQLite du projet.

Usage :
    python -m data.download_data

Produit :
    data/processed/nyc_taxi.db  — tables : train, test
"""

import sqlite3
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
DB_PATH = Path(__file__).parent / "processed" / "nyc_taxi.db"

DTYPE_MAP = {
    "vendor_id":          "int8",
    "passenger_count":    "int8",
    "store_and_fwd_flag": "category",
}


def charger_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Lecture de train.csv …")
    train = pd.read_csv(
        RAW_DIR / "train.csv",
        parse_dates=["pickup_datetime", "dropoff_datetime"],
        dtype=DTYPE_MAP,
    )
    print("Lecture de test.csv …")
    test = pd.read_csv(
        RAW_DIR / "test.csv",
        parse_dates=["pickup_datetime"],
        dtype=DTYPE_MAP,
    )
    return train, test


def ecrire_base(train: pd.DataFrame, test: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    print("Écriture de la table train …")
    train.to_sql("train", con, if_exists="replace", index=False)
    print("Écriture de la table test …")
    test.to_sql("test", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_train_pickup ON train(pickup_datetime)")
    con.commit()
    con.close()


def main() -> None:
    train, test = charger_csv()
    print(f"  train : {train.shape[0]:,} lignes × {train.shape[1]} colonnes")
    print(f"  test  : {test.shape[0]:,} lignes × {test.shape[1]} colonnes")
    ecrire_base(train, test)
    print(f"Base créée : {DB_PATH}")


if __name__ == "__main__":
    main()
