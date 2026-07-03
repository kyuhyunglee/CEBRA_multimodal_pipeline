from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONTINUOUS_COLUMNS = (
    "choice",
    "feedbackType",
    "contrastLeft",
    "contrastRight",
    "probabilityLeft",
)


def _log(message: str) -> None:
    print(f"[sampler] {message}", flush=True)


@dataclass(frozen=True)
class IBLSessionFiles:
    subject_id: str
    session_date: str
    session_number: str
    session_id: str
    alf_dir: Path
    spike_times_path: Path
    spike_clusters_path: Path
    trials_table_path: Path
    stim_on_trigger_path: Path | None


def _versioned_file(alf_dir: Path, file_name: str) -> Path | None:
    direct = alf_dir / file_name
    if direct.exists():
        return direct
    candidates = sorted(alf_dir.glob(f"*/{file_name}"), reverse=True)
    return candidates[0] if candidates else None


def _probe_file(alf_dir: Path, file_name: str, probe_name: str = "probe00") -> Path | None:
    probe_root = alf_dir / probe_name / "pykilosort"
    direct = probe_root / file_name
    if direct.exists():
        return direct
    candidates = sorted(probe_root.glob(f"*/{file_name}"), reverse=True)
    return candidates[0] if candidates else None


def discover_ibl_sessions(
    lab_subjects_dir: str | Path,
    probe_name: str = "probe00",
) -> list[IBLSessionFiles]:
    """Find sessions containing the required IBL ALF files."""
    lab_subjects_dir = Path(lab_subjects_dir)
    sessions: list[IBLSessionFiles] = []

    for alf_dir in sorted(lab_subjects_dir.glob("*/*/*/alf")):
        try:
            subject_id = alf_dir.parents[2].name
            session_date = alf_dir.parents[1].name
            session_number = alf_dir.parents[0].name
        except IndexError:
            continue

        spike_times_path = _probe_file(alf_dir, "spikes.times.npy", probe_name=probe_name)
        spike_clusters_path = _probe_file(alf_dir, "spikes.clusters.npy", probe_name=probe_name)
        trials_table_path = _versioned_file(alf_dir, "_ibl_trials.table.pqt")
        stim_on_trigger_path = _versioned_file(alf_dir, "_ibl_trials.stimOnTrigger_times.npy")

        if spike_times_path is None or spike_clusters_path is None or trials_table_path is None:
            continue

        sessions.append(
            IBLSessionFiles(
                subject_id=subject_id,
                session_date=session_date,
                session_number=session_number,
                session_id=f"{subject_id}__{session_date}__{session_number}",
                alf_dir=alf_dir,
                spike_times_path=spike_times_path,
                spike_clusters_path=spike_clusters_path,
                trials_table_path=trials_table_path,
                stim_on_trigger_path=stim_on_trigger_path,
            )
        )

    return sessions


def bin_spikes(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spike_times = np.asarray(spike_times).reshape(-1)
    spike_clusters = np.asarray(spike_clusters).reshape(-1)
    if len(spike_times) != len(spike_clusters):
        raise ValueError("spike_times and spike_clusters must have the same length.")
    if len(spike_times) == 0:
        raise ValueError("Cannot bin an empty spike train.")

    start = float(np.nanmin(spike_times) if start_time is None else start_time)
    end = float(np.nanmax(spike_times) if end_time is None else end_time)
    if end <= start:
        raise ValueError(f"Invalid time range: start={start}, end={end}.")

    edges = np.arange(start, end + bin_size, bin_size)
    if len(edges) < 2:
        edges = np.array([start, start + bin_size], dtype=float)

    clusters = np.unique(spike_clusters)
    binned = np.zeros((len(edges) - 1, len(clusters)), dtype=np.float32)
    for column, cluster_id in enumerate(clusters):
        counts, _ = np.histogram(spike_times[spike_clusters == cluster_id], bins=edges)
        binned[:, column] = counts

    timestamps = edges[:-1] + (bin_size / 2.0)
    return binned, timestamps.astype(np.float32), clusters


def _series_or_nan(trials: pd.DataFrame, column: str, length: int) -> np.ndarray:
    if column not in trials.columns:
        return np.full(length, np.nan)
    return trials[column].to_numpy()


def _trial_start_times(trials: pd.DataFrame, stim_on_trigger: np.ndarray | None) -> np.ndarray:
    if stim_on_trigger is not None:
        return np.asarray(stim_on_trigger).reshape(-1)
    for column in ("stimOnTrigger_times", "stimOn_times", "goCue_times"):
        if column in trials.columns:
            return trials[column].to_numpy()
    raise ValueError(
        "Could not find trial start times. Provide _ibl_trials.stimOnTrigger_times.npy "
        "or a stimOn/stimOnTrigger/goCue column in _ibl_trials.table.pqt."
    )


def _trial_end_times(trials: pd.DataFrame) -> np.ndarray:
    for column in ("firstMovement_times", "response_times", "feedback_times"):
        if column in trials.columns:
            values = trials[column].to_numpy()
            if np.isfinite(values).any():
                return values
    raise ValueError("Could not find firstMovement_times, response_times, or feedback_times.")


def build_behavior_labels(
    trials: pd.DataFrame,
    timestamps: np.ndarray,
    stim_on_trigger: np.ndarray | None = None,
    continuous_columns: tuple[str, ...] = DEFAULT_CONTINUOUS_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create CEBRA behaviour labels for stimulus-to-movement decision windows."""
    timestamps = np.asarray(timestamps).reshape(-1)
    num_bins = len(timestamps)
    starts = _trial_start_times(trials, stim_on_trigger)
    ends = _trial_end_times(trials)
    num_trials = min(len(trials), len(starts), len(ends))

    continuous = np.zeros((num_bins, len(continuous_columns) + 1), dtype=np.float32)
    discrete = np.zeros(num_bins, dtype=np.int64)

    trial_values = {
        column: _series_or_nan(trials, column, len(trials))
        for column in continuous_columns
    }

    for trial_idx in range(num_trials):
        start = starts[trial_idx]
        end = ends[trial_idx]
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        mask = (timestamps >= start) & (timestamps <= end)
        if not mask.any():
            continue

        duration = max(float(end - start), 1e-6)
        continuous[mask, 0] = (timestamps[mask] - start) / duration
        for col_idx, column in enumerate(continuous_columns, start=1):
            value = trial_values[column][trial_idx]
            continuous[mask, col_idx] = 0.0 if pd.isna(value) else float(value)

        choice = trial_values.get("choice", np.full(len(trials), np.nan))[trial_idx]
        reward = trial_values.get("feedbackType", np.full(len(trials), np.nan))[trial_idx]
        choice_code = 1 if pd.isna(choice) else int(choice) + 1
        reward_code = 1 if pd.isna(reward) else int(reward) + 1
        discrete[mask] = (choice_code * 10) + reward_code

    return continuous, discrete


def preprocess_ibl_session(
    files: IBLSessionFiles,
    output_dir: str | Path,
    bin_size: float,
    continuous_columns: tuple[str, ...] = DEFAULT_CONTINUOUS_COLUMNS,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Loading spike times: {files.spike_times_path}")
    spike_times = np.load(files.spike_times_path)
    _log(f"Loaded spike times shape={spike_times.shape}, dtype={spike_times.dtype}")
    _log(f"Loading spike clusters: {files.spike_clusters_path}")
    spike_clusters = np.load(files.spike_clusters_path)
    _log(f"Loaded spike clusters shape={spike_clusters.shape}, dtype={spike_clusters.dtype}")
    _log(f"Loading trials table: {files.trials_table_path}")
    trials = pd.read_parquet(files.trials_table_path)
    _log(f"Loaded trials shape={trials.shape}")
    stim_on_trigger = (
        np.load(files.stim_on_trigger_path)
        if files.stim_on_trigger_path is not None
        else None
    )
    if files.stim_on_trigger_path is not None:
        _log(
            f"Loaded stimOnTrigger shape={stim_on_trigger.shape} "
            f"from {files.stim_on_trigger_path}"
        )
    else:
        _log("No stimOnTrigger file found; using trial table timing columns.")

    start_times = _trial_start_times(trials, stim_on_trigger)
    end_times = _trial_end_times(trials)
    finite_times = np.concatenate(
        [
            np.asarray(start_times)[np.isfinite(start_times)],
            np.asarray(end_times)[np.isfinite(end_times)],
        ]
    )
    start_time = float(np.nanmin(finite_times)) if len(finite_times) else None
    end_time = float(np.nanmax(finite_times)) if len(finite_times) else None
    _log(f"Binning spikes with bin_size={bin_size}, start_time={start_time}, end_time={end_time}")

    probe, timestamps, clusters = bin_spikes(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        bin_size=bin_size,
        start_time=start_time,
        end_time=end_time,
    )
    _log(
        f"Binned probe shape={probe.shape}, timestamps shape={timestamps.shape}, "
        f"clusters={len(clusters)}"
    )
    _log(f"Building behavior labels with columns={continuous_columns}")
    continuous, discrete = build_behavior_labels(
        trials=trials,
        timestamps=timestamps,
        stim_on_trigger=stim_on_trigger,
        continuous_columns=continuous_columns,
    )
    _log(f"Built continuous labels shape={continuous.shape}, discrete labels shape={discrete.shape}")

    _log(f"Saving arrays to: {output_dir}")
    np.save(output_dir / "probe.npy", probe)
    np.save(output_dir / "continuous_labels.npy", continuous)
    np.save(output_dir / "discrete_labels.npy", discrete)
    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "cluster_ids.npy", clusters)
    _log("Saved probe, labels, timestamps, and cluster ids.")

    metadata = {
        "subject_id": files.subject_id,
        "session_date": files.session_date,
        "session_number": files.session_number,
        "session_id": files.session_id,
        "alf_dir": str(files.alf_dir),
        "spike_times_path": str(files.spike_times_path),
        "spike_clusters_path": str(files.spike_clusters_path),
        "trials_table_path": str(files.trials_table_path),
        "stim_on_trigger_path": None
        if files.stim_on_trigger_path is None
        else str(files.stim_on_trigger_path),
        "bin_size": bin_size,
        "num_bins": int(len(probe)),
        "num_clusters": int(probe.shape[1]),
        "continuous_columns": ("decision_progress",) + tuple(continuous_columns),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _log(f"Wrote metadata: {output_dir / 'metadata.json'}")
    return metadata
