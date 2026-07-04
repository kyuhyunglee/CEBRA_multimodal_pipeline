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


def _select_label_sessions(
    continuous_label_sessions: list[np.ndarray] | None,
    discrete_label_sessions: list[np.ndarray] | None,
    label_type: str,
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
    if label_type == "continuous":
        if continuous_label_sessions is None:
            raise ValueError("--label-type continuous was requested, but no continuous labels were found.")
        return continuous_label_sessions, None
    if label_type == "discrete":
        if discrete_label_sessions is None:
            raise ValueError("--label-type discrete was requested, but no discrete labels were found.")
        return None, discrete_label_sessions
    return continuous_label_sessions, discrete_label_sessions


def train_unimodal(
    config_path: str,
    modality: str,
    use_labels: bool | None = None,
    label_type: str = "auto",
    subject_id: str | None = None,
) -> Path:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    preprocessed_dir = resolve_project_path(config["paths"]["preprocessed_dir"])
    model_dir = ensure_dir(Path(config["paths"]["model_dir"]) / f"{modality}_only")
    subject_ids = [subject_id] if subject_id is not None else configured_subject_ids(config)

    (
        feature_sessions,
        continuous_label_sessions,
        discrete_label_sessions,
        session_metadata,
    ) = load_joint_training_sessions(
        preprocessed_dir=preprocessed_dir,
        subject_ids=subject_ids,
        modalities=[modality],
        window_bins=dataset_config["time_window_bins"],
        time_shift_bins=dataset_config.get("time_shift_bins", 0),
        normalize=True,
    )

    cache_payload = {f"session_{idx:03d}": x for idx, x in enumerate(feature_sessions)}
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
    np.savez(model_dir / "training_cache.npz", **cache_payload)
    (model_dir / "training_sessions.json").write_text(
        json.dumps(session_metadata, indent=2),
        encoding="utf-8",
    )

    model = build_cebra_model(config)
    has_labels = continuous_label_sessions is not None or discrete_label_sessions is not None
    should_use_labels = has_labels if use_labels is None else use_labels
    if should_use_labels:
        if not has_labels:
            raise ValueError(
                "--labels on was requested, but no continuous/discrete label files were found."
            )
        fit_continuous_labels, fit_discrete_labels = _select_label_sessions(
            continuous_label_sessions,
            discrete_label_sessions,
            label_type=label_type,
        )
        if label_type == "discrete" and len(feature_sessions) > 1:
            raise NotImplementedError(
                "CEBRA 0.4.0 supports discrete labels for single-session training, "
                "but not for multi-session training. Pass --subject-id to train one "
                "session with discrete labels, or use --label-type continuous for "
                "multi-session unimodal training."
            )
        if (
            label_type == "auto"
            and len(feature_sessions) > 1
            and fit_continuous_labels is not None
            and fit_discrete_labels is not None
        ):
            print(
                "Both continuous and discrete labels are available. "
                "Using continuous labels by default for multi-session CEBRA. "
                "Pass --label-type discrete to use discrete labels instead."
            )
            fit_discrete_labels = None
        fit_cebra_model(model, feature_sessions, fit_continuous_labels, fit_discrete_labels)
    else:
        fit_cebra_model(model, feature_sessions)

    save_path = model_dir / f"{modality}_cebra_model.pt"
    model.save(save_path)
    print(f"Saved {modality}-only CEBRA model: {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CEBRA on one modality only.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--modality", required=True)
    parser.add_argument(
        "--labels",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use shared behaviour/video labels when available.",
    )
    parser.add_argument(
        "--label-type",
        choices=("auto", "continuous", "discrete"),
        default="auto",
        help="Choose which label set to use when both continuous and discrete labels exist.",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Optional preprocessed subject/session id. Required for true discrete multi-session avoidance.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_mode = {"auto": None, "on": True, "off": False}[args.labels]
    train_unimodal(
        args.config,
        args.modality,
        use_labels=label_mode,
        label_type=args.label_type,
        subject_id=args.subject_id,
    )
