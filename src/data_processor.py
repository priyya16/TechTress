"""Load, validate, and clean representative adverse-event datasets."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from quality import apply_severity_flags, apply_synonym_normalization
from utils import (
    CASE_ID_COLUMNS,
    DRUG_COLUMN_ALIASES,
    EVENT_COLUMN_ALIASES,
    OPTIONAL_AE_COLUMNS,
    REQUIRED_AE_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when an uploaded or sample dataset cannot be used."""


def _rewind(source: Any) -> None:
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:  # noqa: BLE001
            pass


def _read_csv(source: Any) -> pd.DataFrame:
    attempts = (
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin-1"},
    )
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            _rewind(source)
            return pd.read_csv(source, **kwargs)
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError("Invalid CSV file. The file is empty.") from exc
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except pd.errors.ParserError as exc:
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001 — try the next encoding/separator
            last_exc = exc
            continue
    if isinstance(last_exc, UnicodeDecodeError):
        raise DataValidationError(
            "Invalid CSV file. Save the file as UTF-8 and try again."
        ) from last_exc
    if isinstance(last_exc, pd.errors.ParserError):
        raise DataValidationError("Invalid CSV file. Check delimiters and quoting.") from last_exc
    raise DataValidationError("Invalid CSV file.") from last_exc


def load_adverse_event_csv(source: Any) -> pd.DataFrame:
    """Load a CSV from a path, buffer, or Streamlit uploader."""
    frame = _read_csv(source)
    if frame.empty:
        raise DataValidationError("No usable records found.")
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        lookup[key] = col
        lookup[str(col).strip().lower()] = col
    return lookup


def _looks_like_dossier(frame: pd.DataFrame) -> bool:
    keys = set(_column_lookup(frame))
    dossier_hints = {"section", "document", "status", "dossier_section", "module"}
    ae_hints = set(DRUG_COLUMN_ALIASES) | set(EVENT_COLUMN_ALIASES)
    return len(keys & dossier_hints) >= 2 and not (keys & ae_hints)


def validate_required_columns(frame: pd.DataFrame) -> list[str]:
    """Return missing required columns after alias matching."""
    lookup = _column_lookup(frame)
    missing = []
    if not any(alias in lookup for alias in DRUG_COLUMN_ALIASES):
        missing.append("drug_name")
    if not any(alias in lookup for alias in EVENT_COLUMN_ALIASES):
        missing.append("adverse_event")
    return missing


def _resolve_alias(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = _column_lookup(frame)
    rename: dict[str, str] = {}
    drug_col = _resolve_alias(lookup, DRUG_COLUMN_ALIASES)
    event_col = _resolve_alias(lookup, EVENT_COLUMN_ALIASES)
    if drug_col:
        rename[drug_col] = "drug_name"
    if event_col:
        rename[event_col] = "adverse_event"
    for col in OPTIONAL_AE_COLUMNS:
        if col in lookup and lookup[col] not in rename:
            rename[lookup[col]] = col
    return frame.rename(columns=rename)


def clean_adverse_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean missing values, duplicates, and invalid rows.

    Returns the cleaned frame and a report of dropped records.
    """
    if _looks_like_dossier(frame):
        raise DataValidationError(
            "This CSV looks like a CTD dossier outline (section/document/status), "
            "not adverse-event reports. Open Submission Readiness and upload it there."
        )

    missing = validate_required_columns(frame)
    if missing:
        found = ", ".join(str(col) for col in frame.columns)
        raise DataValidationError(
            "Required column missing: "
            + ", ".join(missing)
            + ". Expected a drug column (drug_name / drug / product) and an event "
            "column (adverse_event / event / pt). Found: "
            + found
            + "."
        )

    cleaned = _standardize_columns(frame).copy()
    original_rows = len(cleaned)

    drug_series = cleaned["drug_name"]
    event_series = cleaned["adverse_event"]

    invalid_mask = (
        drug_series.isna()
        | event_series.isna()
        | drug_series.fillna("").astype(str).str.strip().eq("")
        | event_series.fillna("").astype(str).str.strip().eq("")
        | drug_series.fillna("").astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
        | event_series.fillna("").astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
    )
    invalid_rows = int(invalid_mask.sum())
    cleaned = cleaned.loc[~invalid_mask].copy()

    cleaned["drug_name"] = cleaned["drug_name"].astype(str).str.strip()
    cleaned["adverse_event"] = cleaned["adverse_event"].astype(str).str.strip()

    cleaned, map_stats = apply_synonym_normalization(cleaned)
    cleaned = apply_severity_flags(cleaned)

    if "age" in cleaned.columns:
        cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")

    if "report_year" in cleaned.columns:
        cleaned["report_year"] = pd.to_numeric(cleaned["report_year"], errors="coerce")

    id_cols = [col for col in CASE_ID_COLUMNS if col in cleaned.columns]
    if id_cols:
        subset = id_cols + ["drug_name", "adverse_event"]
        duplicate_rows = int(cleaned.duplicated(subset=subset).sum())
        cleaned = cleaned.drop_duplicates(subset=subset).reset_index(drop=True)
    else:
        # Repeated drug-event rows are separate reports and must be kept for PRR.
        duplicate_rows = 0
        cleaned = cleaned.reset_index(drop=True)

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
