"""Small stderr progress reporter for command-line runs."""

from __future__ import annotations

import sys
import time


class CliProgress:
    """Prints compact phase and percentage progress to stderr."""

    def __init__(self, label: str, total: int | None = None, enabled: bool = True):
        """Initializes the progress reporter."""
        self.label = label
        self.total = total
        self.enabled = enabled
        self.started = time.time()
        self.last_value: int | None = None
        self.last_message: str | None = None
        if self.enabled:
            self.update(0, "started")

    def update(self, value: int | None = None, message: str | None = None) -> None:
        """Prints a progress update."""
        if not self.enabled:
            return
        if value is not None:
            value = int(value)
        if value == self.last_value and message == self.last_message:
            return
        self.last_value = value
        self.last_message = message
        elapsed = time.time() - self.started
        parts = [f"[{self.label}]"]
        if value is not None and self.total:
            percent = min(100.0, 100.0 * value / self.total)
            parts.append(f"{value}/{self.total} ({percent:5.1f}%)")
        elif value is not None:
            parts.append(str(value))
        if message:
            parts.append(message)
        parts.append(f"{elapsed:.1f}s")
        print(" ".join(parts), file=sys.stderr, flush=True)

    def step(self, value: int, message: str | None = None) -> None:
        """Prints a completed step update."""
        self.update(value, message)

    def finish(self, message: str = "done") -> None:
        """Prints a final progress update."""
        self.update(self.total, message)
