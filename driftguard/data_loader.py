"""CSV ingestion and validation.

The loader never modifies values: missing values and duplicate rows are kept
exactly as uploaded and reported separately by the profiling / quality modules.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path, PurePath

import pandas as pd

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_COLUMNS = 500
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used for analysis."""


@dataclass
class LoadedDataset:
    name: str
    frame: pd.DataFrame
    size_bytes: int
    warnings: list[str] = field(default_factory=list)


def safe_display_name(name: str | None) -> str:
    """Strip any directory components and control characters from a file name."""
    raw = PurePath(str(name or "dataset.csv").replace("\\", "/")).name
    cleaned = "".join(ch for ch in raw if ch.isprintable()).strip()
    return (cleaned or "dataset.csv")[:120]


def _decode(data: bytes, warnings: list[str]) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        warnings.append("File is not valid UTF-8; decoded as Windows-1252 instead.")
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise DatasetValidationError(
                "Could not decode the file. Please save it as UTF-8 encoded CSV."
            ) from exc


def _check_header(text: str) -> None:
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    header = next(csv.reader([first_line]), [])
    names = [h.strip() for h in header]
    seen: set[str] = set()
    duplicates = sorted({n for n in names if n and (n in seen or seen.add(n))})
    if duplicates:
        raise DatasetValidationError(
            f"Duplicate column names in header: {', '.join(duplicates)}. "
            "Every column must have a unique name."
        )
    if len(names) > MAX_COLUMNS:
        raise DatasetValidationError(
            f"File has {len(names)} columns; the maximum supported is {MAX_COLUMNS}."
        )


def load_csv_bytes(
    data: bytes, name: str, max_bytes: int = MAX_FILE_SIZE_BYTES
) -> LoadedDataset:
    display_name = safe_display_name(name)
    if not display_name.lower().endswith(".csv"):
        raise DatasetValidationError(
            f"'{display_name}' is not a .csv file. Only CSV files are supported."
        )
    if len(data) == 0:
        raise DatasetValidationError(f"'{display_name}' is empty.")
    if len(data) > max_bytes:
        raise DatasetValidationError(
            f"'{display_name}' is {len(data) / 1_048_576:.1f} MB; "
            f"the limit is {max_bytes / 1_048_576:.0f} MB."
        )

    warnings: list[str] = []
    text = _decode(data, warnings)
    if not text.strip():
        raise DatasetValidationError(f"'{display_name}' contains only whitespace.")
    _check_header(text)

    try:
        frame = pd.read_csv(io.StringIO(text), low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError(f"'{display_name}' has no columns.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(
            f"'{display_name}' is not a well-formed CSV: {str(exc).splitlines()[0]}"
        ) from exc

    if frame.shape[1] == 0:
        raise DatasetValidationError(f"'{display_name}' has no columns.")
    if frame.shape[0] == 0:
        raise DatasetValidationError(
            f"'{display_name}' has a header but no data rows."
        )

    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.shape[1] == 1:
        warnings.append(
            "Only one column was detected. If the file uses ';' or tabs as "
            "separators, re-save it with commas."
        )
    unnamed = [c for c in frame.columns if c.startswith("Unnamed:")]
    if unnamed:
        warnings.append(
            f"{len(unnamed)} column(s) have no header name (e.g. an exported "
            "index). They are kept and analysed like any other column."
        )
    empty_cols = [c for c in frame.columns if frame[c].isna().all()]
    if empty_cols:
        warnings.append(f"Columns with no values at all: {', '.join(empty_cols[:10])}.")

    return LoadedDataset(display_name, frame, len(data), warnings)


def list_sample_files() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLES_DIR.glob("*.csv") if p.is_file())


def load_sample(file_name: str) -> LoadedDataset:
    """Load a bundled sample by name. Only files inside SAMPLES_DIR are allowed."""
    allowed = set(list_sample_files())
    if file_name not in allowed:
        raise DatasetValidationError(f"Unknown sample file: {safe_display_name(file_name)}")
    path = (SAMPLES_DIR / file_name).resolve()
    if path.parent != SAMPLES_DIR.resolve():
        raise DatasetValidationError("Invalid sample path.")
    return load_csv_bytes(path.read_bytes(), file_name)
