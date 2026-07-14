"""Persistent run-directory logging."""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    """JSON fallback serializer.
    
    Args:
        value: Value value.
    
    Returns:
        Json default result.

    """
    return str(value)


def _safe_run_id_component(value: Any, default: str) -> str:
    """Returns a filesystem-safe run id component.
    
    Args:
        value: Value value.
        default: Default value.
    
    Returns:
        Safe run id component result.

    """
    text = str(value or default)
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text
    )
    return cleaned or default


class ExperimentLogger:
    """Creates run directories and stores config, metrics, and summaries."""

    def __init__(self, base_dir: str | Path, config: dict[str, Any]) -> None:
        """Initializes a persistent run directory.
        
        Args:
            base_dir: Parent directory for generated artifacts.
            config: Experiment configuration dictionary.

        """
        self.base_dir = Path(base_dir)
        self.config = config
        serialized = json.dumps(config, sort_keys=True, default=_json_default)
        self.config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
        self.environment_name = _safe_run_id_component(
            config.get("environment_config", {}).get("environment_name"),
            "environment",
        )
        self.algorithm_name = _safe_run_id_component(
            config.get("agent_config", {}).get("algorithm_name"),
            "algorithm",
        )
        self.run_id = (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{self.environment_name}-{self.algorithm_name}-{self.config_hash}"
        )
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.write_json("config.json", config)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Writes a JSON file inside the run directory.
        
        Args:
            name: Name value.
            payload: Serializable payload to persist.

        """
        with (self.run_dir / name).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=_json_default)

    def append_csv(self, name: str, row: dict[str, Any]) -> None:
        """Appends one row to a CSV file.
        
        Args:
            name: Name value.
            row: Row value.

        """
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
        """Returns reproducibility metadata.
        
        Returns:
            Computed result.

        """
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            git_commit = None
        try:
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            git_dirty = bool(git_status.strip())
        except Exception:
            git_dirty = None
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "environment_name": self.environment_name,
            "algorithm_name": self.algorithm_name,
            "git_commit_hash": git_commit,
            "git_dirty": git_dirty,
            "user": getpass.getuser(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def finalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Adds the run end time and rewrites metadata.json.
        
        Args:
            metadata: Metadata payload to persist or enrich.
        
        Returns:
            Computed result.

        """
        finalized = dict(metadata)
        finalized["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.write_json("metadata.json", finalized)
        return finalized
