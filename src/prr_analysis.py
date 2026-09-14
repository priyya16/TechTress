"""Proportional Reporting Ratio (PRR) using a 2x2 contingency table.

Contingency table
-----------------
                Event           Other Events
Drug            a               b
Other Drugs     c               d

PRR = [a / (a + b)] / [c / (c + d)]

PRR is a statistical screening measure. Flagged pairs are labeled
"Potential Safety Signal" and require expert/regulatory review.
"""

from __future__ import annotations

import math

import pandas as pd

from utils import STATUS_BELOW, STATUS_INSUFFICIENT, STATUS_SIGNAL, STATUS_UNDEFINED


def compute_prr(a: float, b: float, c: float, d: float) -> dict[str, float | str | None]:
    """Compute PRR for one 2x2 table with explicit invalid-case handling."""
    try:
        a = float(a)
        b = float(b)
        c = float(c)
        d = float(d)
    except (TypeError, ValueError):
        return {
            "prr": None,
            "status_reason": "Invalid numeric values in the contingency table.",
            "computable": False,
        }

    if any(math.isnan(x) or math.isinf(x) for x in (a, b, c, d)):
        return {
            "prr": None,
            "status_reason": "Non-finite values cannot be used in PRR.",
            "computable": False,
        }

    if min(a, b, c, d) < 0:
        return {
            "prr": None,
            "status_reason": "Counts cannot be negative.",
            "computable": False,
        }

    drug_total = a + b
    other_total = c + d
    if drug_total == 0:
        return {
            "prr": None,
            "status_reason": "Insufficient data: this drug has no reports (a + b = 0).",
            "computable": False,
        }
    if other_total == 0:
        return {
            "prr": None,
            "status_reason": "Insufficient data: other drugs have no reports (c + d = 0).",
            "computable": False,
        }

    p_drug = a / drug_total
    p_other = c / other_total
    if p_other == 0:
        return {
            "prr": None,
            "status_reason": (
                "Undefined PRR: the event was not reported for other drugs "
                "(c = 0), which would divide by zero."
            ),
            "computable": False,
        }

    return {
        "prr": p_drug / p_other,
        "status_reason": "PRR computed from the 2x2 table.",
        "computable": True,
    }


def build_contingency_tables(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a, b, c, d for every drug-event pair in the dataset."""
    total = len(frame)
    drug_totals = frame.groupby("drug_name").size().to_dict()
    event_totals = frame.groupby("adverse_event").size().to_dict()
    combo_counts = (
        frame.groupby(["drug_name", "adverse_event"]).size().reset_index(name="a")
    )

    rows = []
    for rec in combo_counts.itertuples(index=False):
        a = int(rec.a)
        drug_total = int(drug_totals[rec.drug_name])
        event_total = int(event_totals[rec.adverse_event])
        b = drug_total - a
        c = event_total - a
        d = total - a - b - c
        rows.append(
            {
                "drug_name": rec.drug_name,
                "adverse_event": rec.adverse_event,
                "a": a,
                "b": b,
                "c": c,
                "d": max(d, 0),
            }
        )
    return pd.DataFrame(rows)


def analyze_prr(
    frame: pd.DataFrame,
    min_a: int = 3,
    min_prr: float = 2.0,
) -> pd.DataFrame:
    """Calculate PRR for all combinations and rank potential signals."""
    if min_a < 1:
        raise ValueError("Minimum case count (a) must be at least 1.")
    if min_prr <= 0:
        raise ValueError("PRR threshold must be greater than 0.")

    tables = build_contingency_tables(frame)
    records = []
    for rec in tables.itertuples(index=False):
        result = compute_prr(rec.a, rec.b, rec.c, rec.d)
        prr = result["prr"]
        computable = bool(result["computable"])

        if not computable:
            status = (
                STATUS_UNDEFINED
                if "divide by zero" in str(result["status_reason"]).lower()
                or "undefined" in str(result["status_reason"]).lower()
                else STATUS_INSUFFICIENT
            )
        elif rec.a < min_a:
            status = STATUS_INSUFFICIENT
            result["status_reason"] = (
                "Insufficient data to calculate PRR as a screening flag: "
                f"a = {rec.a}, which is below the configured minimum of {min_a}."
            )
        elif prr is not None and prr >= min_prr:
            status = STATUS_SIGNAL
            result["status_reason"] = (
                f"PRR = {prr:.3f} is at or above the screening threshold "
                f"({min_prr}) with a = {rec.a}."
            )
        else:
            status = STATUS_BELOW
            result["status_reason"] = (
                f"PRR = {prr:.3f} is below the screening threshold ({min_prr})."
            )

        records.append(
            {
                "Drug": rec.drug_name,
                "Adverse Event": rec.adverse_event,
                "a": rec.a,
                "b": rec.b,
                "c": rec.c,
                "d": rec.d,
                "PRR": None if prr is None else round(float(prr), 4),
                "Signal Status": status,
                "Notes": result["status_reason"],
            }
        )

    result_frame = pd.DataFrame(records)
    result_frame["rank_prr"] = pd.to_numeric(result_frame["PRR"], errors="coerce").fillna(-1)
    result_frame = result_frame.sort_values(
        by=["Signal Status", "rank_prr", "a"],
        ascending=[True, False, False],
        key=lambda col: col
        if col.name != "Signal Status"
        else col.map(
            {
                STATUS_SIGNAL: 0,
                STATUS_UNDEFINED: 1,
                STATUS_INSUFFICIENT: 2,
                STATUS_BELOW: 3,
            }
        ),
    ).drop(columns=["rank_prr"])
    return result_frame.reset_index(drop=True)


def count_potential_signals(result_frame: pd.DataFrame) -> int:
    if result_frame.empty:
        return 0
    return int((result_frame["Signal Status"] == STATUS_SIGNAL).sum())
