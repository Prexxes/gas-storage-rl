"""Tqdm progress reporter for command-line runs."""

from __future__ import annotations

from tqdm.auto import tqdm


class CliProgress:
    """Wraps tqdm with an absolute-value progress API."""

    def __init__(self, label: str, total: int | None = None, enabled: bool = True):
        """Initializes the progress reporter."""
        self.label = label
        self.total = total
        self.enabled = enabled
        self.current_value = 0
        self.last_message: str | None = None
        self._bar = None
        if enabled:
            self._bar = tqdm(
                total=total,
                desc=label,
                unit="step",
                leave=True,
                dynamic_ncols=True,
                disable=not enabled,
            )

    def update(self, value: int | None = None, message: str | None = None) -> None:
        """Updates the progress bar to an absolute value."""
        if not self.enabled or self._bar is None:
            return
        if value is not None:
            value = int(value)
            if self.total is not None:
                value = min(value, self.total)
            delta = max(value - self.current_value, 0)
            if delta:
                self._bar.update(delta)
                self.current_value = value
        if message:
            self._bar.set_postfix_str(message)
            self.last_message = message
        self._bar.refresh()

    def step(self, value: int, message: str | None = None) -> None:
        """Updates the progress bar to a completed absolute step."""
        self.update(value, message)

    def finish(self, message: str = "done") -> None:
        """Completes and closes the progress bar."""
        if not self.enabled or self._bar is None:
            return
        self.update(self.total, message)
        self._bar.close()
        self._bar = None
