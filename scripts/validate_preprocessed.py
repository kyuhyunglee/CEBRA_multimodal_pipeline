from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders.multimodal_dataset import build_session_windows, load_modality_session
from utils.config import configured_subject_ids, load_config, resolve_project_path


def validate_preprocessed(config_path: str) -> None:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    preprocessing = config.get("preprocessing", {})
    preprocessed_dir = resolve_project_path(config["paths"]["preprocessed_dir"])
    window_bins = dataset_config["time_window_bins"]
    time_shift_bins = dataset_config.get("time_shift_bins", 0)
    require_labels = preprocessing.get("require_labels", False)

    found_sessions = 0
    errors = []

    for subject_id in configured_subject_ids(config):
        for modality in dataset_config["modalities"]:
            try:
                session = load_modality_session(preprocessed_dir, subject_id, modality)
            except FileNotFoundError:
                continue

            found_sessions += 1
            try:
                features, continuous_labels, discrete_labels = build_session_windows(
                    session,
                    window_bins=window_bins,
                    time_shift_bins=time_shift_bins,
                    normalize=False,
                )
            except Exception as exc:
                errors.append(f"{subject_id}/{modality}: {exc}")
                continue

            if require_labels and continuous_labels is None and discrete_labels is None:
                errors.append(
                    f"{subject_id}/{modality}: continuous_labels.npy, labels.npy, "
                    "or discrete_labels.npy is required but missing."
                )

            continuous_shape = None if continuous_labels is None else continuous_labels.shape
            discrete_shape = None if discrete_labels is None else discrete_labels.shape
            print(
                f"{subject_id}/{modality}: neural={session.neural.shape}, "
                f"windows={features.shape}, continuous={continuous_shape}, "
                f"discrete={discrete_shape}"
            )

    if found_sessions == 0:
        errors.append("No sessions found. Check paths.preprocessed_dir and dataset settings.")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"\nValidation passed for {found_sessions} session(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate standardized CEBRA input arrays.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_preprocessed(args.config)
