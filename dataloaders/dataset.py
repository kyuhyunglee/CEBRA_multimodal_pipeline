from __future__ import annotations

from pathlib import Path

from .sampler import IBLSessionFiles, discover_ibl_sessions


class IBLSubjectDataset:
    """Compatibility wrapper around the IBL ALF session discovery utilities."""

    def __init__(self, subject_path: str | Path, probe_name: str = "probe00"):
        self.subject_path = Path(subject_path)
        self.probe_name = probe_name
        self.sessions: list[IBLSessionFiles] = discover_ibl_sessions(
            self.subject_path,
            probe_name=probe_name,
        )

    def get_sessions(self) -> list[IBLSessionFiles]:
        return self.sessions
