from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedModalities:
    probe: np.ndarray
    calcium: np.ndarray
    labels: np.ndarray | None = None
    timestamps: np.ndarray | None = None


def _as_2d(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")
    return array


def truncate_to_shared_length(
    probe: np.ndarray,
    calcium: np.ndarray,
    labels: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> AlignedModalities:
    """Align modalities by truncating all arrays to the shortest time length."""
    probe = _as_2d("probe", probe)
    calcium = _as_2d("calcium", calcium)
    lengths = [len(probe), len(calcium)]

    if labels is not None:
        labels = _as_2d("labels", labels)
        lengths.append(len(labels))
    if timestamps is not None:
        timestamps = np.asarray(timestamps).reshape(-1)
        lengths.append(len(timestamps))

    shared = min(lengths)
    return AlignedModalities(
        probe=probe[:shared],
        calcium=calcium[:shared],
        labels=None if labels is None else labels[:shared],
        timestamps=None if timestamps is None else timestamps[:shared],
    )


def align_by_nearest_timestamps(
    source_values: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Resample source values onto target timestamps by nearest neighbor lookup."""
    source_values = _as_2d("source_values", source_values)
    source_timestamps = np.asarray(source_timestamps).reshape(-1)
    target_timestamps = np.asarray(target_timestamps).reshape(-1)

    if len(source_values) != len(source_timestamps):
        raise ValueError("source_values and source_timestamps must have the same length.")

    indices = np.searchsorted(source_timestamps, target_timestamps, side="left")
    indices = np.clip(indices, 0, len(source_timestamps) - 1)
    prev_indices = np.clip(indices - 1, 0, len(source_timestamps) - 1)

    choose_prev = (
        np.abs(target_timestamps - source_timestamps[prev_indices])
        <= np.abs(target_timestamps - source_timestamps[indices])
    )
    nearest = np.where(choose_prev, prev_indices, indices)
    return source_values[nearest]


def sliding_windows(array: np.ndarray, window_bins: int) -> np.ndarray:
    """Return windows with shape (samples, window_bins, features)."""
    array = _as_2d("array", array)
    if window_bins < 1:
        raise ValueError("window_bins must be >= 1.")
    if len(array) < window_bins:
        raise ValueError(
            f"Cannot build {window_bins}-bin windows from {len(array)} samples."
        )
    return np.lib.stride_tricks.sliding_window_view(
        array, window_shape=window_bins, axis=0
    ).transpose(0, 2, 1)


def zscore(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = _as_2d("array", array).astype(np.float32, copy=False)
    return (array - array.mean(axis=0, keepdims=True)) / (
        array.std(axis=0, keepdims=True) + eps
    )
