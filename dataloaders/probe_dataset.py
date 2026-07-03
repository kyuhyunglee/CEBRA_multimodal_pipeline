from __future__ import annotations

from pathlib import Path

import numpy as np


class ProbeDataset:
    """Thin loader for standardized probe arrays."""

    def __init__(self, subject_dir: str | Path):
        self.subject_dir = Path(subject_dir)

    def load(self) -> np.ndarray:
        path = self.subject_dir / "probe.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing probe data: {path}")
        return np.load(path)
