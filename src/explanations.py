"""Rule-based reviewer explanations. No external AI API is required."""

from __future__ import annotations

from utils import STATUS_SIGNAL


def explain_prr_row(row: dict) -> str:
    """Explain one drug-event PRR result in plain language."""
    status = row.get("Signal Status", "")
    drug = row.get("Drug", "this drug")
    event = row.get("Adverse Event", "this event")
    a = row.get("a")
    b = row.get("b")
    c = row.get("c")
    d = row.get("d")
    prr = row.get("PRR")

    table = (
        f"The 2x2 table is a={a} (drug + event), b={b} (drug + other events), "
        f"c={c} (other drugs + event), d={d} (other drugs + other events)."
    )

    if status == STATUS_SIGNAL and prr is not None:
        return (
            f"{drug} with {event} was flagged as a Potential Safety Signal because "
            f"PRR = {prr}. Statistically, this event appears about {prr} times as "
            f"often among reports for {drug} as among reports for other drugs in "
            f"this dataset. {table} PRR is a screening ratio only; it does not "
            "prove the drug caused the event. A pharmacovigilance reviewer should "
            "examine case quality, confounders, and additional evidence."
        )

    if prr is None:
        return (
            f"{drug} with {event} could not receive a valid PRR. {table} "
            f"{row.get('Notes', '')} Do not treat a missing PRR as a confirmed "
            "or dismissed safety issue."
        )

    return (
        f"{drug} with {event} has PRR = {prr}. {table} "
        f"{row.get('Notes', '')} This pair is still available for reviewer "
        "inspection but is not flagged under the current screening thresholds."
    )


def explain_gap(module_name: str, section: str, priority: str) -> str:
    """Explain why a missing CTD section is prioritized."""
    priority_text = {
        "High": (
            "High-priority gaps typically block a reviewer from assessing core "
            "quality, clinical, or summary evidence."
        ),
        "Medium": (
            "Medium-priority gaps matter for a complete dossier but may be "
            "addressable after the highest-impact missing summaries."
        ),
        "Low": (
            "Low-priority gaps are still expected in a representative outline, "
            "but they are usually addressed after high-impact modules."
        ),
    }
    return (
        f"{section} is missing from {module_name}. "
        f"{priority_text.get(priority, '')} "
        "This checklist is representative of ICH M4 CTD structure, not a "
        "complete regulatory filing requirement. Confirm the expected content "
        "with the applicable regional Module 1 guidance."
    )


def recommend_next_actions(missing_high: int, overall_score: float) -> list[str]:
    actions = []
    if overall_score < 70:
        actions.append(
            "Treat the dossier as not submission-ready. Close high-priority gaps "
            "before any completeness claim."
        )
    elif overall_score < 90:
        actions.append(
            "The representative outline is partially ready. Resolve remaining "
            "high- and medium-priority gaps, then re-score."
        )
    else:
        actions.append(
            "Representative completeness is high. Perform a human quality review "
            "of present sections; scoring does not assess scientific adequacy."
        )
    if missing_high:
        actions.append(
            f"Prioritize the {missing_high} high-priority missing section(s) "
            "in Modules 2, 3, and 5 first."
        )
    actions.append(
        "Do not submit based on this prototype score alone. Use it as a "
        "reviewer checklist against ICH M4 CTD structure."
    )
    return actions


def explain_signal_highlights(row: dict, min_a: int = 3, min_prr: float = 2.0) -> list[str]:
    """Short judge-facing bullets built from calculated PRR fields only."""
    a = row.get("a")
    prr = row.get("PRR")
    status = row.get("Signal Status", "")
    lines = [
        f"Reports: **{a}**",
        f"PRR: **{prr}**" if prr is not None else "PRR: **undefined**",
        f"Signal threshold: **{min_prr}**",
        f"Minimum cases: **{min_a}**",
    ]
    if status == STATUS_SIGNAL and prr is not None:
        lines.append("PRR is above the configured threshold.")
        lines.append(
            "Reporting is disproportionately high compared with other drugs in this file."
        )
    elif prr is None:
        lines.append("PRR could not be calculated from the 2x2 table (often c = 0).")
    else:
        lines.append(str(row.get("Notes", "Not flagged under the current screening rule.")))
    original = row.get("Original Event")
    canonical = row.get("Adverse Event")
    if original and str(original).strip() and str(original) != str(canonical):
        lines.append(f"**Original event:** {original}")
        lines.append(f"**Canonical event:** {canonical}")
    return lines
