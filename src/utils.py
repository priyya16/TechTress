"""Shared constants, paths, and helper text for the prototype."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"
SAMPLE_ADVERSE_EVENTS = SAMPLE_DATA_DIR / "adverse_events.csv"
SAMPLE_SIGNAL_DEMO = SAMPLE_DATA_DIR / "signal_detection_demo.csv"
SAMPLE_DOSSIER_OUTLINE = SAMPLE_DATA_DIR / "dossier_outline.txt"

APP_TITLE = "TechTress — Pharmacovigilance & Submission Readiness"
APP_SUBTITLE = "AI-Assisted Pharmacovigilance & Regulatory Submission Readiness"
TEAM_NAME = "TechTress"
VALUE_PROP = (
    "TechTress reduces manual review burden by automatically prioritizing "
    "potential drug-safety signals and identifying missing regulatory "
    "submission documents."
)
HOME_BLURB = (
    "Detect potential adverse-event safety signals and identify CTD "
    "submission gaps before regulatory review."
)
SYNTHETIC_BANNER = "Synthetic Demo Data — Not FDA FAERS Data"

DISCLAIMER = (
    "Hackathon prototype for automated safety-signal screening and CTD readiness "
    "assessment. It is not a certified medical device, pharmacovigilance system, "
    "or FDA/regulatory decision tool. Demo files are Synthetic Demo Data, not the "
    "FDA FAERS 20M+ database. The CTD scorer is an ICH M4-based demo checklist, "
    "not complete official eCTD validation."
)
CTD_DEMO_NOTE = (
    "This is an ICH M4-based demo checklist and is not a complete official "
    "eCTD/M4 regulatory validator."
)
DEMO_LEAF_NOTE = (
    "5.9 and 5.10 are project/demo checklist items and should not be "
    "interpreted as official ICH M4 leaf-code claims."
)
SPOTLIGHT_CTD_CODES = ("2.5", "3.2.A", "4.4", "5.4", "5.9", "5.10")

PRR_HELP = (
    "Proportional Reporting Ratio (PRR) compares how often an event is reported "
    "for one drug versus all other drugs in the uploaded dataset. "
    "It is a disproportionality screening measure, not proof of causation."
)

REQUIRED_AE_COLUMNS = ("drug_name", "adverse_event")
OPTIONAL_AE_COLUMNS = (
    "case_id",
    "report_id",
    "primaryid",
    "narrative",
    "description",
    "adverse_event_description",
    "report_text",
    "age",
    "sex",
    "outcome",
    "report_year",
)
DRUG_COLUMN_ALIASES = (
    "drug_name",
    "drug",
    "product",
    "product_name",
    "productname",
    "drugname",
    "medicinalproduct",
    "medicinal_product",
    "substance",
    "drug_product",
)
EVENT_COLUMN_ALIASES = (
    "adverse_event",
    "event",
    "reaction",
    "pt",
    "preferred_term",
    "preferredterm",
    "ae",
    "adverseevent",
    "reaction_pt",
    "event_name",
    "adverse_reaction",
)
CASE_ID_COLUMNS = ("case_id", "report_id", "primaryid", "isr")

STATUS_SIGNAL = "Potential Safety Signal"
STATUS_BELOW = "Below Threshold"
STATUS_INSUFFICIENT = "Insufficient Data"
STATUS_UNDEFINED = "Undefined (division by zero)"


def normalize_label(value: str) -> str:
    """Normalize section or column labels for comparison."""
    return " ".join(str(value).strip().lower().split())
