from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import sliding_windows, zscore


@dataclass(frozen=True)
class ModalitySession:
    subject_id: str
    modality: str
    neural: np.ndarray
    continuous_labels: np.ndarray | None
    discrete_labels: np.ndarray | None
    timestamps: np.ndarray | None
    metadata: dict


def _as_2d(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")
    return array


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _load_continuous_labels(subject_dir: Path) -> np.ndarray | None:
    # labels.npy is kept as the default alias for continuous labels such as
    # DINO video features.
    labels = _load_optional(subject_dir / "continuous_labels.npy")
    if labels is not None:
        return labels
    return _load_optional(subject_dir / "labels.npy")


def _load_metadata(subject_dir: Path) -> dict:
    metadata_path = subject_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_modality_session(
    preprocessed_dir: str | Path,
    subject_id: str,
    modality: str,
) -> ModalitySession:
    """Load one modality as an independent CEBRA session.

    This intentionally does not merge modalities into a single feature array.
    Joint training is handled by passing multiple sessions to CEBRA with shared
    behavioural/video labels or time labels.
    """
    subject_dir = Path(preprocessed_dir) / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Missing preprocessed subject directory: {subject_dir}")

    neural_path = subject_dir / f"{modality}.npy"
    if not neural_path.exists():
        raise FileNotFoundError(f"Missing required modality file: {neural_path}")

    neural = _as_2d(modality, np.load(neural_path))
    continuous_labels = _load_continuous_labels(subject_dir)
    discrete_labels = _load_optional(subject_dir / "discrete_labels.npy")
    timestamps = _load_optional(subject_dir / "timestamps.npy")

    lengths = [len(neural)]
    if continuous_labels is not None:
        continuous_labels = _as_2d("continuous_labels", continuous_labels)
        lengths.append(len(continuous_labels))
    if discrete_labels is not None:
        discrete_labels = np.asarray(discrete_labels).reshape(-1)
        lengths.append(len(discrete_labels))
    if timestamps is not None:
        timestamps = np.asarray(timestamps).reshape(-1)
        lengths.append(len(timestamps))

    shared = min(lengths)
    return ModalitySession(
        subject_id=subject_id,
        modality=modality,
        neural=neural[:shared],
        continuous_labels=None
        if continuous_labels is None
        else continuous_labels[:shared],
        discrete_labels=None if discrete_labels is None else discrete_labels[:shared],
        timestamps=None if timestamps is None else timestamps[:shared],
        metadata=_load_metadata(subject_dir),
    )


def build_session_windows(
    session: ModalitySession,
    window_bins: int,
    time_shift_bins: int = 0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    neural = zscore(session.neural) if normalize else session.neural
    windows = sliding_windows(neural, window_bins)
    features = windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)

    if session.continuous_labels is None and session.discrete_labels is None:
        return features, None, None

    label_start = window_bins - 1 + time_shift_bins
    label_end = label_start + len(features)
    label_lengths = []
    if session.continuous_labels is not None:
        label_lengths.append(len(session.continuous_labels))
    if session.discrete_labels is not None:
        label_lengths.append(len(session.discrete_labels))
    max_label_end = min(label_lengths)
    if label_end > max_label_end:
        usable = max_label_end - label_start
        if usable <= 0:
            raise ValueError("time_shift_bins leaves no usable labels.")
        features = features[:usable]
        label_end = max_label_end

    continuous = None
    discrete = None
    if session.continuous_labels is not None:
        continuous = session.continuous_labels[label_start:label_end].astype(
            np.float32, copy=False
        )
    if session.discrete_labels is not None:
        discrete = session.discrete_labels[label_start:label_end]

    return features, continuous, discrete


def load_joint_training_sessions(
    preprocessed_dir: str | Path,
    subject_ids: list[str],
    modalities: list[str],
    window_bins: int,
    time_shift_bins: int = 0,
    normalize: bool = True,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[dict],
]:
    """Return independent sessions for CEBRA joint/multi-session training."""
    feature_sessions: list[np.ndarray] = []
    continuous_label_sessions: list[np.ndarray] = []
    discrete_label_sessions: list[np.ndarray] = []
    session_metadata: list[dict] = []
    continuous_seen = False
    continuous_missing = False
    discrete_seen = False
    discrete_missing = False

    for subject_id in subject_ids:
        for modality in modalities:
            try:
                session = load_modality_session(preprocessed_dir, subject_id, modality)
            except FileNotFoundError:
                continue
            features, continuous, discrete = build_session_windows(
                session,
                window_bins=window_bins,
                time_shift_bins=time_shift_bins,
                normalize=normalize,
            )
            feature_sessions.append(features)
            session_metadata.append(
                {
                    "subject_id": subject_id,
                    "modality": modality,
                    "num_samples": int(len(features)),
                    "num_features": int(features.shape[1]),
                    "has_continuous_labels": continuous is not None,
                    "has_discrete_labels": discrete is not None,
                }
            )

            if continuous is None:
                continuous_missing = True
                if continuous_seen:
                    raise ValueError(
                        "Either every session must have continuous labels or none may have them."
                    )
            else:
                continuous_seen = True
                if continuous_missing:
                    raise ValueError(
                        "Either every session must have continuous labels or none may have them."
                    )
                continuous_label_sessions.append(continuous)

            if discrete is None:
                discrete_missing = True
                if discrete_seen:
                    raise ValueError(
                        "Either every session must have discrete labels or none may have them."
                    )
            else:
                discrete_seen = True
                if discrete_missing:
                    raise ValueError(
                        "Either every session must have discrete labels or none may have them."
                    )
                discrete_label_sessions.append(discrete)

    if not feature_sessions:
        raise FileNotFoundError(
            "No modality sessions were found. Check dataset.target_subjects, "
            "dataset.modalities, and paths.preprocessed_dir."
        )

    return (
        feature_sessions,
        continuous_label_sessions if continuous_seen else None,
        discrete_label_sessions if discrete_seen else None,
        session_metadata,
    )
