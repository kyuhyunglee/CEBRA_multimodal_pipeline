from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders.sampler import (
    DEFAULT_CONTINUOUS_COLUMNS,
    discover_ibl_sessions,
    preprocess_ibl_session,
)
from utils.config import ensure_dir, load_config, resolve_project_path


STANDARD_FILES = (
    "probe.npy",
    "labels.npy",
    "continuous_labels.npy",
    "discrete_labels.npy",
    "timestamps.npy",
    "metadata.json",
)


def _candidate_subject_dirs(drive_base: Path, subject_id: str) -> list[Path]:
    return [
        drive_base / "raw" / subject_id,
        drive_base / subject_id,
    ]


def _copy_subject_directory(source_dir: Path, target_dir: Path) -> list[str]:
    copied = []
    for file_name in STANDARD_FILES:
        source = source_dir / file_name
        if source.exists():
            shutil.copy2(source, target_dir / file_name)
            copied.append(file_name)
    return copied


def _prepare_from_manifest(manifest_path: Path, drive_base: Path, preprocessed_dir: Path) -> None:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "modality", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest must contain columns {sorted(required)}; got {reader.fieldnames}."
            )

        for row in reader:
            subject_id = row["subject_id"]
            modality = row["modality"]
            source = Path(row["path"])
            if not source.is_absolute():
                source = drive_base / source

            target_name = row.get("target_name") or f"{modality}.npy"
            target_dir = preprocessed_dir / subject_id
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_dir / target_name)


def prepare_data(config_path: str) -> None:
    config = load_config(config_path)
    paths = config["paths"]
    dataset = config["dataset"]
    preprocessing = config.get("preprocessing", {})

    drive_base = Path(paths["drive_base_path"])
    preprocessed_dir = ensure_dir(paths["preprocessed_dir"])
    manifest_path = resolve_project_path(paths["manifest_path"])

    if dataset.get("provider") == "ibl_one":
        lab_subjects_dir = Path(dataset.get("lab_subjects_dir", drive_base))
        probe_name = dataset.get("probe_name", "probe00")
        continuous_columns = tuple(
            dataset.get("continuous_columns", DEFAULT_CONTINUOUS_COLUMNS)
        )
        sessions = discover_ibl_sessions(lab_subjects_dir, probe_name=probe_name)
        if not sessions:
            raise FileNotFoundError(f"No valid IBL sessions found under {lab_subjects_dir}.")

        written = []
        for session in sessions:
            target_dir = preprocessed_dir / session.session_id
            metadata = preprocess_ibl_session(
                files=session,
                output_dir=target_dir,
                bin_size=dataset["bin_size"],
                continuous_columns=continuous_columns,
            )
            written.append(metadata)

        summary_path = preprocessed_dir / "ibl_sessions.json"
        summary_path.write_text(json.dumps(written, indent=2), encoding="utf-8")
        print(f"Prepared {len(written)} IBL session(s) under: {preprocessed_dir}")
        print(f"Session summary: {summary_path}")
        return

    if manifest_path.exists():
        _prepare_from_manifest(manifest_path, drive_base, preprocessed_dir)
    else:
        for subject_id in dataset["target_subjects"]:
            target_dir = preprocessed_dir / subject_id
            target_dir.mkdir(parents=True, exist_ok=True)

            source_dir = next(
                (candidate for candidate in _candidate_subject_dirs(drive_base, subject_id) if candidate.exists()),
                None,
            )
            if source_dir is None:
                raise FileNotFoundError(
                    f"No source directory found for {subject_id}. Expected one of: "
                    + ", ".join(str(path) for path in _candidate_subject_dirs(drive_base, subject_id))
                )

            copied = _copy_subject_directory(source_dir, target_dir)
            copied_modalities = {
                file_name.removesuffix(".npy")
                for file_name in copied
                if file_name.endswith(".npy")
            }
            requested_modalities = set(dataset["modalities"])
            if not copied_modalities.intersection(requested_modalities):
                raise FileNotFoundError(
                    f"{subject_id} needs at least one of {sorted(requested_modalities)} in {source_dir}."
                )
            has_labels = bool(
                {"labels.npy", "continuous_labels.npy", "discrete_labels.npy"}.intersection(copied)
            )
            if preprocessing.get("require_labels", False) and not has_labels:
                raise FileNotFoundError(
                    f"{subject_id} needs labels.npy, continuous_labels.npy, "
                    f"or discrete_labels.npy in {source_dir}."
                )

            metadata_path = target_dir / "metadata.json"
            if not metadata_path.exists():
                metadata_path.write_text(
                    json.dumps({"subject_id": subject_id, "source_dir": str(source_dir)}, indent=2),
                    encoding="utf-8",
                )

    print(f"Prepared data under: {preprocessed_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare standardized CEBRA multimodal data.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.config)
