from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dataloaders.multimodal_dataset import load_joint_training_sessions
from utils.cebra_factory import build_cebra_model, fit_cebra_model
from utils.config import configured_subject_ids, ensure_dir, load_config, resolve_project_path


def _log(message: str) -> None:
    print(f"[train_joint] {message}", flush=True)


def _save_training_cache(
    model_dir: Path,
    feature_sessions: list[np.ndarray],
    continuous_label_sessions: list[np.ndarray] | None,
    discrete_label_sessions: list[np.ndarray] | None,
    session_metadata: list[dict],
) -> None:
    cache_payload = {
        f"session_{idx:03d}": features
        for idx, features in enumerate(feature_sessions)
    }
    if continuous_label_sessions is not None:
        cache_payload.update(
            {
                f"continuous_labels_{idx:03d}": labels
                for idx, labels in enumerate(continuous_label_sessions)
            }
        )
    if discrete_label_sessions is not None:
        cache_payload.update(
            {
                f"discrete_labels_{idx:03d}": labels
                for idx, labels in enumerate(discrete_label_sessions)
            }
        )
    np.savez(model_dir / "joint_training_cache.npz", **cache_payload)
    (model_dir / "joint_training_sessions.json").write_text(
        json.dumps(session_metadata, indent=2),
        encoding="utf-8",
    )


def _validate_cross_modal_joint_sessions(
    requested_modalities: list[str],
    session_metadata: list[dict],
) -> None:
    loaded_modalities = sorted({metadata["modality"] for metadata in session_metadata})
    requested_unique = sorted(set(requested_modalities))
    missing_modalities = sorted(set(requested_unique) - set(loaded_modalities))

    if len(requested_unique) < 2:
        raise ValueError(
            "train_joint.py is reserved for cross-modality joint training. "
            "Configure at least two modalities, such as ['probe', 'calcium']. "
            "Use scripts/train_unimodal.py for single-modality training."
        )
    if missing_modalities:
        raise FileNotFoundError(
            "Cross-modality joint training requires every requested modality to "
            f"be present in the preprocessed dataset. Missing: {missing_modalities}. "
            f"Loaded modalities: {loaded_modalities}."
        )
    if len(loaded_modalities) < 2:
        raise ValueError(
            "Cross-modality joint training needs at least two loaded modalities. "
            f"Loaded modalities: {loaded_modalities}. "
            "Use scripts/train_unimodal.py for single-modality training."
        )


def train_joint(config_path: str, use_labels: bool | None = None) -> Path:
    _log(f"Loading config: {config_path}")
    config = load_config(config_path)
    dataset_config = config["dataset"]
    preprocessed_dir = resolve_project_path(config["paths"]["preprocessed_dir"])
    model_dir = ensure_dir(config["paths"]["model_dir"])
    subject_ids = configured_subject_ids(config)

    _log(f"preprocessed_dir: {preprocessed_dir}")
    _log(f"model_dir: {model_dir}")
    _log(f"subjects: {len(subject_ids)}")
    _log(f"modalities: {dataset_config['modalities']}")
    _log(
        f"window_bins: {dataset_config['time_window_bins']}, "
        f"time_shift_bins: {dataset_config.get('time_shift_bins', 0)}"
    )

    (
        feature_sessions,
        continuous_label_sessions,
        discrete_label_sessions,
        session_metadata,
    ) = load_joint_training_sessions(
        preprocessed_dir=preprocessed_dir,
        subject_ids=subject_ids,
        modalities=dataset_config["modalities"],
        window_bins=dataset_config["time_window_bins"],
        time_shift_bins=dataset_config.get("time_shift_bins", 0),
        normalize=True,
    )
    _validate_cross_modal_joint_sessions(
        requested_modalities=dataset_config["modalities"],
        session_metadata=session_metadata,
    )
    total_samples = sum(len(features) for features in feature_sessions)
    total_features = sum(features.shape[1] for features in feature_sessions)
    _log(
        f"Loaded {len(feature_sessions)} training session(s), "
        f"total_samples={total_samples}, summed_feature_dims={total_features}."
    )
    for idx, metadata in enumerate(session_metadata):
        _log(
            f"session_{idx:03d}: subject={metadata['subject_id']}, "
            f"modality={metadata['modality']}, samples={metadata['num_samples']}, "
            f"features={metadata['num_features']}, "
            f"continuous={metadata['has_continuous_labels']}, "
            f"discrete={metadata['has_discrete_labels']}"
        )

    _log("Saving training cache.")
    _save_training_cache(
        model_dir,
        feature_sessions,
        continuous_label_sessions,
        discrete_label_sessions,
        session_metadata,
    )
    _log(f"Saved training cache: {model_dir / 'joint_training_cache.npz'}")

    _log("Building modality-aware CEBRA model.")
    model = build_cebra_model(config, session_metadata=session_metadata)
    session_architectures = getattr(model, "session_architectures", None)
    if session_architectures is not None:
        for idx, architecture in enumerate(session_architectures):
            _log(f"session_{idx:03d}: architecture={architecture}")
    has_labels = continuous_label_sessions is not None or discrete_label_sessions is not None
    should_use_labels = has_labels if use_labels is None else use_labels
    if (
        should_use_labels
        and len(feature_sessions) > 1
        and continuous_label_sessions is not None
        and discrete_label_sessions is not None
    ):
        _log(
            "Both continuous and discrete labels are available. "
            "Using continuous labels for multi-session CEBRA 0.4.0 compatibility."
        )
        discrete_label_sessions = None
    _log(f"Labels available: {has_labels}; using labels: {should_use_labels}")
    model_device = getattr(model, "device", config["training"].get("device", "auto"))
    _log(
        f"Starting CEBRA fit. max_iterations={config['training']['max_iterations']}, "
        f"batch_size={config['training']['batch_size']}, device={model_device}"
    )
    if should_use_labels:
        if not has_labels:
            raise ValueError(
                "--labels on was requested, but no continuous/discrete label files were found."
            )
        fit_cebra_model(model, feature_sessions, continuous_label_sessions, discrete_label_sessions)
    else:
        fit_cebra_model(model, feature_sessions)

    _log("CEBRA fit finished. Saving model.")
    save_path = model_dir / "joint_cebra_model.pt"
    model.save(save_path, backend="torch")
    _log(f"Saved joint CEBRA model: {save_path}")
    _log(f"Saved session metadata: {model_dir / 'joint_training_sessions.json'}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CEBRA jointly across independent sessions."
    )
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument(
        "--labels",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use shared behaviour/video labels when available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_mode = {"auto": None, "on": True, "off": False}[args.labels]
    train_joint(args.config, use_labels=label_mode)
