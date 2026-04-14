import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common

DB_PATH = Path(common.CONFIG["paths"]["db_path"])
RAW_DATA_DIR = Path(common.CONFIG["paths"]["raw_data"])
CSV_FILE = common.CONFIG["dataset"]["csv_file"]
RANDOM_STATE = int(common.CONFIG["ml"]["random_state"])
TEST_SIZE = float(common.CONFIG["ml"]["test_size"])


def download_data() -> None:
    csv_path = RAW_DATA_DIR / CSV_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading CSV file: {csv_path}")
    data = pd.read_csv(csv_path)
    data_train, data_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving train and test data to database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        data_train.to_sql(name="train", con=con, if_exists="replace", index=False)
        data_test.to_sql(name="test", con=con, if_exists="replace", index=False)


def test_download_data() -> None:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        print(f"Reading train data from database: {DB_PATH}")
        res = cur.execute("SELECT COUNT(*) FROM train")
        n_rows = res.fetchone()[0]
        res = cur.execute("SELECT * FROM train LIMIT 1")
        n_cols = len(res.description)
        print(f"Train data: {n_rows} x {n_cols}")

        print(f"Reading test data from database: {DB_PATH}")
        res = cur.execute("SELECT COUNT(*) FROM test")
        n_rows = res.fetchone()[0]
        res = cur.execute("SELECT * FROM test LIMIT 1")
        n_cols = len(res.description)
        print(f"Test data: {n_rows} x {n_cols}")


if __name__ == "__main__":
    download_data()
    test_download_data()