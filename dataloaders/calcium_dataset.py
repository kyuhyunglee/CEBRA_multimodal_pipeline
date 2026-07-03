from __future__ import annotations

from pathlib import Path

import numpy as np


class CalciumDataset:
    """Thin loader for standardized calcium feature arrays."""

    def __init__(self, subject_dir: str | Path):
        self.subject_dir = Path(subject_dir)

    def load(self) -> np.ndarray:
        path = self.subject_dir / "calcium.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing calcium data: {path}")
        return np.load(path)
