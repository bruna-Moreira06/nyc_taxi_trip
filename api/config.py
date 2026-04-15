from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def _resolve_project_paths(config: dict) -> dict:
    paths = config.get("paths", {})
    config["paths"] = {
        key: str((PROJECT_ROOT / value).resolve()) for key, value in paths.items()
    }
    return config


def load_api_config() -> dict:
    with API_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return _resolve_project_paths(config)


API_CONFIG = load_api_config()
