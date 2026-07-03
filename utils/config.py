from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/default_config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_dir(path: str | Path) -> Path:
    path = resolve_project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configured_device(config: dict[str, Any]) -> str:
    requested = config.get("training", {}).get("device", "auto")
    if requested != "auto":
        return requested

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
