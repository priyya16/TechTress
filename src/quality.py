"""Normalize messy drug/event names and flag narrative severity keywords.

These maps are a small representative prototype, not a medical coding dictionary.
"""

from __future__ import annotations

import pandas as pd

from utils import normalize_label

# Lowercase lookup -> canonical display label
DRUG_SYNONYMS = {
    "heptraclear": "HeptraClear",
    "heptra clear": "HeptraClear",
    "neuravex": "Neuravex",
    "cardionil": "Cardionil",
    "amplitide": "Amplitide",
    "placebol": "Placebol",
}

EVENT_SYNONYMS = {
    "liver injury": "Liver Injury",
    "hepatic injury": "Liver Injury",
    "hepatotoxicity": "Liver Injury",
    "hepatocellular injury": "Liver Injury",
    "qt prolongation": "QT Prolongation",
    "qtc prolongation": "QT Prolongation",
    "anaphylaxis": "Anaphylaxis",
    "anaphylactic reaction": "Anaphylaxis",
    "anaphylactic shock": "Anaphylaxis",
    "headache": "Headache",
    "nausea": "Nausea",
    "rash": "Rash",
    "dizziness": "Dizziness",
}

SEVERITY_PATTERNS = {
    "hospitalized": ("hospitalized", "hospitalisation", "hospitalization", "hospital admission"),
    "fatal": ("fatal", "death", "died"),
    "not recovered": ("not recovered", "not recovering"),
    "serious lab": ("jaundice", "anaphylaxis", "hypotension", "icu"),
}


def canonicalize_term(value: str, synonyms: dict[str, str]) -> str:
    """Trim, collapse spaces, then map known variants to a canonical label."""
    collapsed = " ".join(str(value).strip().split())
    key = normalize_label(collapsed)
    if key in synonyms:
        return synonyms[key]
    return collapsed


def flag_severity_text(text: str) -> list[str]:
    """Return keyword flags found in free text. Rule-based, not an ML model."""
    blob = str(text).lower()
    hits = []
    for label, needles in SEVERITY_PATTERNS.items():
        if any(needle in blob for needle in needles):
            hits.append(label)
    return hits


def apply_synonym_normalization(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add raw columns and replace drug/event names using the synonym maps."""
    out = frame.copy()
    out["drug_name_raw"] = out["drug_name"].astype(str)
    out["adverse_event_raw"] = out["adverse_event"].astype(str)
    mapped_drug = out["drug_name_raw"].map(lambda v: canonicalize_term(v, DRUG_SYNONYMS))
    mapped_event = out["adverse_event_raw"].map(lambda v: canonicalize_term(v, EVENT_SYNONYMS))
    trimmed_drug = out["drug_name_raw"].map(lambda v: " ".join(str(v).strip().split()))
    trimmed_event = out["adverse_event_raw"].map(lambda v: " ".join(str(v).strip().split()))
    drug_changed = int((mapped_drug != trimmed_drug).sum())
    event_changed = int((mapped_event != trimmed_event).sum())
    out["drug_name"] = mapped_drug
    out["adverse_event"] = mapped_event
    stats = {
        "drug_labels_mapped": drug_changed,
        "event_labels_mapped": event_changed,
        "unique_drugs_after_map": int(out["drug_name"].nunique()),
        "unique_events_after_map": int(out["adverse_event"].nunique()),
    }
    return out, stats


def apply_severity_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a severity_flags column from narrative and outcome text."""
    out = frame.copy()
    parts = []
    for col in ("narrative", "description", "adverse_event_description", "report_text", "outcome"):
        if col in out.columns:
            parts.append(out[col].fillna("").astype(str))
    if not parts:
        out["severity_flags"] = ""
        out["severity_flag_count"] = 0
        return out
    combined = parts[0]
    for extra in parts[1:]:
        combined = combined + " " + extra
    flags = combined.map(flag_severity_text)
    out["severity_flags"] = flags.map(lambda items: ", ".join(items) if items else "")
    out["severity_flag_count"] = flags.map(len)
    return out


def pair_severity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Roll case-level flags up to drug-event pairs."""
    if "severity_flag_count" not in frame.columns:
        return pd.DataFrame(columns=["Drug", "Adverse Event", "Cases with severity keywords"])
    grouped = (
        frame.groupby(["drug_name", "adverse_event"], dropna=False)
        .agg(
            severity_cases=("severity_flag_count", lambda s: int((s > 0).sum())),
            hospitalized_mentions=("severity_flags", lambda s: int(s.str.contains("hospitalized", na=False).sum())),
            fatal_mentions=("severity_flags", lambda s: int(s.str.contains("fatal", na=False).sum())),
        )
        .reset_index()
        .rename(columns={"drug_name": "Drug", "adverse_event": "Adverse Event"})
    )
    grouped = grouped.rename(
        columns={"severity_cases": "Cases with severity keywords"}
    )
    return grouped


def counts_by_year(frame: pd.DataFrame, drug: str | None = None, event: str | None = None) -> pd.DataFrame:
    """Report counts by year for all rows or one drug-event pair."""
    if "report_year" not in frame.columns:
        return pd.DataFrame(columns=["report_year", "reports"])
    subset = frame.dropna(subset=["report_year"]).copy()
    subset["report_year"] = subset["report_year"].astype(int)
    if drug:
        subset = subset[subset["drug_name"] == drug]
    if event:
        subset = subset[subset["adverse_event"] == event]
    if subset.empty:
        return pd.DataFrame(columns=["report_year", "reports"])
    return (
        subset.groupby("report_year")
        .size()
        .reset_index(name="reports")
        .sort_values("report_year")
    )


def cases_for_pair(frame: pd.DataFrame, drug: str, event: str) -> pd.DataFrame:
    """Return the actual case rows for one drug-event pair."""
    mask = (frame["drug_name"] == drug) & (frame["adverse_event"] == event)
    cols = [
        col
        for col in (
            "case_id",
            "drug_name_raw",
            "adverse_event_raw",
            "drug_name",
            "adverse_event",
            "age",
            "sex",
            "outcome",
            "report_year",
            "narrative",
            "severity_flags",
        )
        if col in frame.columns
    ]
    return frame.loc[mask, cols].reset_index(drop=True)
