"""Persistent run-directory logging."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    """JSON fallback serializer."""
    return str(value)


class ExperimentLogger:
    """Creates run directories and stores config, metrics, and summaries."""

    def __init__(self, base_dir: str | Path, config: dict[str, Any]) -> None:
        """Initializes a persistent run directory."""
        self.base_dir = Path(base_dir)
        self.config = config
        serialized = json.dumps(config, sort_keys=True, default=_json_default)
        self.config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.config_hash}"
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.write_json("config.json", config)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Writes a JSON file inside the run directory."""
        with (self.run_dir / name).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=_json_default)

    def append_csv(self, name: str, row: dict[str, Any]) -> None:
        """Appends one row to a CSV file."""
        path = self.run_dir / name
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
        rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def metadata(self) -> dict[str, Any]:
        """Returns reproducibility metadata."""
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            git_commit = None
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_commit_hash": git_commit,
            "python_version": sys.version,
            "platform": platform.platform(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
