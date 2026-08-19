"""Historical gas price CSV loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HistoricalPriceSeries:
    """Validated historical gas price series."""

    data: pd.DataFrame
    price_column: str

    @property
    def dates(self) -> pd.Series:
        """Returns sorted observation dates.
        
        Returns:
            Date index for the historical series.

        """
        return self.data["date"]

    @property
    def prices(self) -> pd.Series:
        """Returns positive prices.
        
        Returns:
            Price values for the historical series.

        """
        return self.data[self.price_column]


def load_historical_price_csv(
    path: str | Path,
    price_column: str | None = None,
    expected_split: str | None = None,
) -> HistoricalPriceSeries:
    """Loads a historical gas price CSV with date and positive price columns.

    Args:
        path: CSV path. Windows ``C:/`` paths are accepted when running on WSL.
        price_column: Optional explicit price column name.
        expected_split: Optional value required in a ``split`` column.

    Returns:
        Validated and date-sorted historical price series.

    Raises:
        ValueError: If required columns are missing or prices are invalid.

    """
    csv_path = _normalize_path(path)
    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns:
        raise ValueError(f"Historical price CSV must contain a date column: {csv_path}")
    if expected_split is not None and "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str))
        if splits != {expected_split}:
            raise ValueError(
                f"Expected split {expected_split!r} in {csv_path}, "
                f"found {sorted(splits)}"
            )

    selected_price_column = price_column or _infer_price_column(frame)
    selected_columns = ["date", selected_price_column]
    if "split" in frame.columns:
        selected_columns.append("split")
    frame = frame[selected_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False).dt.normalize()
    frame[selected_price_column] = pd.to_numeric(
        frame[selected_price_column],
        errors="raise",
    )
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame["date"].isna().any():
        raise ValueError(f"Historical price CSV contains missing dates: {csv_path}")
    if frame["date"].duplicated().any():
        raise ValueError(f"Historical price CSV contains duplicate dates: {csv_path}")
    if (frame[selected_price_column] <= 0.0).any():
        raise ValueError(
            f"Historical price CSV contains non-positive prices: {csv_path}"
        )

    return HistoricalPriceSeries(frame, selected_price_column)


def assert_date_range(
    series: HistoricalPriceSeries,
    *,
    max_date: str | pd.Timestamp | None = None,
    min_date: str | pd.Timestamp | None = None,
) -> None:
    """Validates inclusive date bounds for a historical price series.
    
    Args:
        series: Series value.
        max_date: Max date value.
        min_date: Min date value.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if max_date is not None:
        cutoff = pd.Timestamp(max_date)
        if (series.dates > cutoff).any():
            raise ValueError(
                f"Series contains dates after maximum date {cutoff.date()}"
            )
    if min_date is not None:
        start = pd.Timestamp(min_date)
        if (series.dates < start).any():
            raise ValueError(
                f"Series contains dates before backtest start {start.date()}"
            )


def _infer_price_column(frame: pd.DataFrame) -> str:
    """Infers the single non-date, non-split numeric price column.
    
    Args:
        frame: Frame value.
    
    Returns:
        Infer price column result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    candidates = [column for column in frame.columns if column not in {"date", "split"}]
    numeric_candidates = [
        column
        for column in candidates
        if pd.to_numeric(frame[column], errors="coerce").notna().all()
    ]
    if len(numeric_candidates) != 1:
        raise ValueError(
            "Could not infer price column. Pass price_column explicitly; "
            f"numeric candidates were {numeric_candidates}."
        )
    return numeric_candidates[0]


def _normalize_path(path: str | Path) -> Path:
    """Converts Windows drive paths to WSL mount paths when needed.
    
    Args:
        path: Filesystem path to read from or write to.
    
    Returns:
        Normalize path result.

    """
    path_text = str(path)
    original_path = Path(path_text)
    if original_path.exists():
        return original_path
    if len(path_text) >= 3 and path_text[1:3] in {":/", ":\\"}:
        drive = path_text[0].lower()
        rest = path_text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)
