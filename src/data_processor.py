"""Load, validate, and clean representative adverse-event datasets."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from quality import apply_severity_flags, apply_synonym_normalization
from utils import OPTIONAL_AE_COLUMNS, REQUIRED_AE_COLUMNS


class DataValidationError(ValueError):
    """Raised when an uploaded or sample dataset cannot be used."""


def _read_csv(source: Any) -> pd.DataFrame:
    try:
        if isinstance(source, (str, StringIO)):
            return pd.read_csv(source)
        return pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("Invalid CSV file. The file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError("Invalid CSV file. Check delimiters and quoting.") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            "Invalid CSV file. Save the file as UTF-8 and try again."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface a readable UI error
        raise DataValidationError("Invalid CSV file.") from exc


def load_adverse_event_csv(source: Any) -> pd.DataFrame:
    """Load a CSV from a path, buffer, or Streamlit uploader."""
    frame = _read_csv(source)
    if frame.empty:
        raise DataValidationError("No usable records found.")
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def validate_required_columns(frame: pd.DataFrame) -> list[str]:
    """Return missing required columns."""
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    missing = [col for col in REQUIRED_AE_COLUMNS if col not in lookup]
    return missing


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for col in list(REQUIRED_AE_COLUMNS) + list(OPTIONAL_AE_COLUMNS):
        if col in lookup:
            rename[lookup[col]] = col
    return frame.rename(columns=rename)


def clean_adverse_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean missing values, duplicates, and invalid rows.

    Returns the cleaned frame and a report of dropped records.
    """
    missing = validate_required_columns(frame)
    if missing:
        raise DataValidationError(
            "Required column missing: "
            + ", ".join(missing)
            + ". Expected columns are drug_name and adverse_event."
        )

    cleaned = _standardize_columns(frame).copy()
    original_rows = len(cleaned)

    cleaned["drug_name"] = cleaned["drug_name"].astype(str).str.strip()
    cleaned["adverse_event"] = cleaned["adverse_event"].astype(str).str.strip()

    invalid_mask = (
        cleaned["drug_name"].eq("")
        | cleaned["adverse_event"].eq("")
        | cleaned["drug_name"].str.lower().isin(["nan", "none", "null"])
        | cleaned["adverse_event"].str.lower().isin(["nan", "none", "null"])
    )
    invalid_rows = int(invalid_mask.sum())
    cleaned = cleaned.loc[~invalid_mask].copy()

    cleaned, map_stats = apply_synonym_normalization(cleaned)
    cleaned = apply_severity_flags(cleaned)

    if "age" in cleaned.columns:
        cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")

    if "report_year" in cleaned.columns:
        cleaned["report_year"] = pd.to_numeric(cleaned["report_year"], errors="coerce")

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if cleaned.empty:
        raise DataValidationError("No usable records found.")

    report = {
        "original_rows": original_rows,
        "invalid_or_missing_required": invalid_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_cleaning": len(cleaned),
        **map_stats,
    }
    return cleaned, report


def dataset_statistics(frame: pd.DataFrame) -> dict[str, int]:
    """Basic counts for the dashboard metrics."""
    combos = frame.groupby(["drug_name", "adverse_event"], dropna=False).size()
    return {
        "total_reports": int(len(frame)),
        "n_drugs": int(frame["drug_name"].nunique()),
        "n_events": int(frame["adverse_event"].nunique()),
        "n_combinations": int(len(combos)),
    }


def detect_text_column(frame: pd.DataFrame) -> str | None:
    """Return a usable free-text column, if one exists."""
    candidates = [
        "narrative",
        "description",
        "adverse_event_description",
        "report_text",
    ]
    for col in candidates:
        if col not in frame.columns:
            continue
        series = frame[col].dropna().astype(str).str.strip()
        series = series[series.str.len() >= 20]
        if series.nunique() >= 8:
            return col
    return None
