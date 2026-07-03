from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dataloaders.multimodal_dataset import build_session_windows, load_modality_session
from utils.config import ensure_dir, load_config, resolve_project_path


def _transform_with_optional_session_id(model, features: np.ndarray, session_id: int):
    try:
        return model.transform(features, session_id=session_id)
    except TypeError:
        return model.transform(features)


def _find_training_session_id(model_dir: Path, subject_id: str, modality: str) -> int:
    metadata_path = model_dir / "joint_training_sessions.json"
    if not metadata_path.exists():
        return 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for idx, session in enumerate(metadata):
        if session["subject_id"] == subject_id and session["modality"] == modality:
            return idx
    return 0


def embed_subject(
    config_path: str,
    subject_id: str | None = None,
    modality: str | None = None,
    model_path: str | None = None,
) -> list[Path]:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    subject_id = subject_id or dataset_config["target_subjects"][0]
    modalities = [modality] if modality else list(dataset_config["modalities"])

    preprocessed_dir = resolve_project_path(config["paths"]["preprocessed_dir"])
    model_dir = ensure_dir(config["paths"]["model_dir"])
    model_path = (
        resolve_project_path(model_path)
        if model_path is not None
        else model_dir / "joint_cebra_model.pt"
    )

    import cebra

    try:
        model = cebra.CEBRA.load(model_path)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        model = cebra.CEBRA.load(model_path, weights_only=False)
    outputs = []
    for current_modality in modalities:
        session = load_modality_session(preprocessed_dir, subject_id, current_modality)
        features, continuous_labels, discrete_labels = build_session_windows(
            session,
            window_bins=dataset_config["time_window_bins"],
            time_shift_bins=dataset_config.get("time_shift_bins", 0),
            normalize=True,
        )

        session_id = _find_training_session_id(model_dir, subject_id, current_modality)
        embedding = _transform_with_optional_session_id(model, features, session_id=session_id)

        embedding_dir = ensure_dir(Path(config["paths"]["embedding_dir"]) / subject_id)
        save_path = embedding_dir / f"{current_modality}_embedding.npz"
        np.savez(
            save_path,
            embedding=embedding,
            continuous_labels=np.array([])
            if continuous_labels is None
            else continuous_labels,
            discrete_labels=np.array([]) if discrete_labels is None else discrete_labels,
            subject_id=subject_id,
            modality=current_modality,
        )
        print(f"Saved {current_modality} embedding: {save_path}")
        outputs.append(save_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CEBRA embeddings for one subject/modality.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--modality", choices=("probe", "calcium"), default=None)
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed_subject(
        args.config,
        subject_id=args.subject_id,
        modality=args.modality,
        model_path=args.model_path,
    )
