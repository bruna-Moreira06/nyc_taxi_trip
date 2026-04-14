from __future__ import annotations

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yml"


def _resolve_project_paths(config: dict) -> dict:
    paths = config.get("paths", {})
    config["paths"] = {
        key: str((ROOT_DIR / value).resolve()) for key, value in paths.items()
    }
    return config


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return _resolve_project_paths(config)


CONFIG = load_config()