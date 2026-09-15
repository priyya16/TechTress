"""Orchestration layer used by the Streamlit frontend.

The UI never computes PRR, clusters, or CTD scores itself. It calls these
functions and renders the returned DataFrames and dictionaries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_processor import (
    DataValidationError,
    clean_adverse_events,
    dataset_statistics,
    load_adverse_event_csv,
)
from ml_screening import enrich_signal_table
from prr_analysis import analyze_prr, count_potential_signals
from quality import emerging_signal_table, pair_original_labels, pair_severity_summary
from signal_detection import cluster_reports
from submission_checker import evaluate_dossier, parse_dossier_text


def load_reports(source: Any) -> pd.DataFrame:
    """Load an adverse-event CSV from a path, buffer, or uploader."""
    return load_adverse_event_csv(source)


def run_signal_pipeline(
    frame: pd.DataFrame,
    min_a: int = 3,
    min_prr: float = 2.0,
) -> dict:
    """Validate, clean, cluster, and calculate PRR for one dataset."""
    cleaned, clean_report = clean_adverse_events(frame)
    clustering = cluster_reports(cleaned)
    results = analyze_prr(cleaned, min_a=min_a, min_prr=min_prr)
    results = enrich_signal_table(results, min_prr=min_prr, min_ror=min_prr)
    originals = pair_original_labels(cleaned)
    if not originals.empty:
        results = results.merge(originals, on=["Drug", "Adverse Event"], how="left")
    severity = pair_severity_summary(cleaned)
    if not severity.empty:
        results = results.merge(severity, on=["Drug", "Adverse Event"], how="left")
        for col in ("Cases with severity keywords", "hospitalized_mentions", "fatal_mentions"):
            if col in results.columns:
                results[col] = results[col].fillna(0).astype(int)
    emerging, emerging_reason = emerging_signal_table(cleaned, results, min_a=min_a)
    return {
        "cleaned": cleaned,
        "clean_report": clean_report,
        "stats": dataset_statistics(cleaned),
        "clustering": clustering,
        "results": results,
        "n_signals": count_potential_signals(results),
        "n_anomalies": int((results.get("ML Anomaly") == "Anomalous pair").sum())
        if "ML Anomaly" in results.columns
        else 0,
        "emerging": emerging,
        "emerging_reason": emerging_reason,
        "n_insufficient": int((results["Signal Status"] == "Insufficient Data").sum())
        if "Signal Status" in results.columns
        else 0,
    }


def run_dossier_pipeline(text: str) -> dict:
    """Parse an outline and score representative ICH M4 completeness."""
    sections = parse_dossier_text(text)
    return evaluate_dossier(sections)


def friendly_error(exc: Exception) -> str:
    """Map backend exceptions to short reviewer-facing messages."""
    if isinstance(exc, DataValidationError):
        return str(exc)
    if isinstance(exc, UnicodeDecodeError):
        return "Invalid CSV file. Save the file as UTF-8 and try again."
    message = str(exc).strip()
    if not message:
        return "Unable to process this input. Check the file or pasted text and try again."
    return message
