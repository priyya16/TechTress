"""Local ML and extra statistical screening. No cloud AI API is used.

Adds Reporting Odds Ratio (ROR), chi-square on the same 2x2 table as PRR,
Isolation Forest anomaly flags, and a transparent reviewer priority score.
These methods support ranking only. They are not medical diagnoses.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def compute_ror(a: float, b: float, c: float, d: float) -> dict:
    """ROR = (a/c) / (b/d) = (a * d) / (b * c)."""
    try:
        a, b, c, d = float(a), float(b), float(c), float(d)
    except (TypeError, ValueError):
        return {"ror": None, "reason": "Invalid counts."}
    if min(a, b, c, d) < 0:
        return {"ror": None, "reason": "Negative counts."}
    if b == 0 or c == 0:
        return {
            "ror": None,
            "reason": "Undefined ROR (b = 0 or c = 0 would divide by zero).",
        }
    return {"ror": (a * d) / (b * c), "reason": "ROR computed."}


def compute_chi_square(a: float, b: float, c: float, d: float) -> dict:
    """Pearson chi-square for a 2x2 table, with 1-df p-value approximation."""
    try:
        a, b, c, d = float(a), float(b), float(c), float(d)
    except (TypeError, ValueError):
        return {"chi2": None, "p_value": None, "reason": "Invalid counts."}
    if min(a, b, c, d) < 0:
        return {"chi2": None, "p_value": None, "reason": "Negative counts."}
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if n == 0 or denom == 0:
        return {
            "chi2": None,
            "p_value": None,
            "reason": "Insufficient data to calculate chi-square.",
        }
    chi2 = n * (a * d - b * c) ** 2 / denom
    p_value = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 >= 0 else None
    return {"chi2": chi2, "p_value": p_value, "reason": "Chi-square computed."}


def method_agreement(prr, ror, min_prr: float, min_ror: float) -> str:
    """Compare PRR and ROR against the same style of screening threshold."""
    prr_high = prr is not None and pd.notna(prr) and float(prr) >= min_prr
    ror_high = ror is not None and pd.notna(ror) and float(ror) >= min_ror
    if prr_high and ror_high:
        return "PRR and ROR both high"
    if prr_high:
        return "Only PRR high"
    if ror_high:
        return "Only ROR high"
    return "Neither high"


def enrich_signal_table(
    results: pd.DataFrame,
    random_state: int = 42,
    min_prr: float = 2.0,
    min_ror: float | None = None,
) -> pd.DataFrame:
    """Add ROR, chi-square, Isolation Forest flags, and a 0-100 priority score."""
    if results.empty:
        return results

    enriched = results.copy()
    ror_vals = []
    chi_vals = []
    p_vals = []
    for rec in enriched.itertuples(index=False):
        ror = compute_ror(rec.a, rec.b, rec.c, rec.d)
        chi = compute_chi_square(rec.a, rec.b, rec.c, rec.d)
        ror_vals.append(None if ror["ror"] is None else round(float(ror["ror"]), 4))
        chi_vals.append(None if chi["chi2"] is None else round(float(chi["chi2"]), 4))
        p_vals.append(None if chi["p_value"] is None else round(float(chi["p_value"]), 4))

    enriched["ROR"] = ror_vals
    enriched["Chi-square"] = chi_vals
    enriched["Chi-square p (approx)"] = p_vals
    ror_cut = min_prr if min_ror is None else min_ror
    enriched["Method Agreement"] = [
        method_agreement(prr, ror, min_prr, ror_cut)
        for prr, ror in zip(enriched["PRR"], enriched["ROR"])
    ]

    feature_frame = pd.DataFrame(
        {
            "a": pd.to_numeric(enriched["a"], errors="coerce").fillna(0),
            "prr": pd.to_numeric(enriched["PRR"], errors="coerce").fillna(0),
            "ror": pd.to_numeric(enriched["ROR"], errors="coerce").fillna(0),
            "chi2": pd.to_numeric(enriched["Chi-square"], errors="coerce").fillna(0),
        }
    )
    feature_frame["log_a"] = np.log1p(feature_frame["a"])
    x = feature_frame[["log_a", "prr", "ror", "chi2"]].to_numpy()

    n_rows = len(enriched)
    if n_rows >= 8:
        contamination = min(0.25, max(0.08, 3 / n_rows))
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )
        labels = model.fit_predict(x)
        scores = model.decision_function(x)
        enriched["ML Anomaly"] = np.where(labels == -1, "Anomalous pair", "Typical pair")
        enriched["IsolationForest score"] = np.round(scores, 4)
    else:
        enriched["ML Anomaly"] = "Insufficient rows for Isolation Forest"
        enriched["IsolationForest score"] = None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        feature_frame[["prr", "chi2", "log_a"]].to_numpy()
    )
    anomaly_boost = np.where(enriched["ML Anomaly"] == "Anomalous pair", 1.0, 0.0)
    priority = (
        100
        * (0.40 * scaled[:, 0] + 0.30 * scaled[:, 1] + 0.20 * scaled[:, 2] + 0.10 * anomaly_boost)
    )
    enriched["Reviewer Priority Score"] = np.round(priority, 1)
    return enriched
