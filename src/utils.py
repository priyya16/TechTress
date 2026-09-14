"""Shared constants, paths, and helper text for the prototype."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"
SAMPLE_ADVERSE_EVENTS = SAMPLE_DATA_DIR / "adverse_events.csv"
SAMPLE_DOSSIER_OUTLINE = SAMPLE_DATA_DIR / "dossier_outline.txt"

APP_TITLE = "Drug Safety Signal Detector & Regulatory Submission Readiness Checker"
TEAM_NAME = "TechTress"

DISCLAIMER = (
    "This application is a hackathon prototype for reviewer support only. "
    "It is not a certified medical device, pharmacovigilance system, or "
    "regulatory compliance tool. Flagged PRR results are statistical screening "
    "findings that require expert and regulatory review. The CTD checklist is "
    "representative, not exhaustive."
)

PRR_HELP = (
    "Proportional Reporting Ratio (PRR) compares how often an event is reported "
    "for one drug versus all other drugs in the uploaded dataset. "
    "It is a disproportionality screening measure, not proof of causation."
)

REQUIRED_AE_COLUMNS = ("drug_name", "adverse_event")
OPTIONAL_AE_COLUMNS = (
    "case_id",
    "narrative",
    "description",
    "adverse_event_description",
    "report_text",
    "age",
    "sex",
    "outcome",
    "report_year",
)

STATUS_SIGNAL = "Potential Safety Signal"
STATUS_BELOW = "Below Threshold"
STATUS_INSUFFICIENT = "Insufficient Data"
STATUS_UNDEFINED = "Undefined (division by zero)"


def normalize_label(value: str) -> str:
    """Normalize section or column labels for comparison."""
    return " ".join(str(value).strip().lower().split())
