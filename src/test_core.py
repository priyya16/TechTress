"""Unit tests for PRR, validation, and CTD completeness logic."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processor import DataValidationError, clean_adverse_events, load_adverse_event_csv
from prr_analysis import analyze_prr, compute_prr
from submission_checker import evaluate_dossier, parse_dossier_text
from utils import STATUS_INSUFFICIENT, STATUS_SIGNAL, STATUS_UNDEFINED


def test_prr_standard_table():
    result = compute_prr(a=8, b=12, c=4, d=76)
    assert result["computable"] is True
    expected = (8 / 20) / (4 / 80)
    assert result["prr"] == pytest.approx(expected)


def test_prr_division_by_zero_when_c_is_zero():
    result = compute_prr(a=5, b=5, c=0, d=90)
    assert result["computable"] is False
    assert result["prr"] is None
    assert "zero" in result["status_reason"].lower()


def test_prr_insufficient_when_drug_has_no_reports():
    result = compute_prr(a=0, b=0, c=10, d=10)
    assert result["computable"] is False
    assert "a + b = 0" in result["status_reason"]


def test_prr_insufficient_when_other_drugs_empty():
    result = compute_prr(a=3, b=2, c=0, d=0)
    assert result["computable"] is False
    assert "c + d = 0" in result["status_reason"]


def test_prr_rejects_negative_counts():
    result = compute_prr(a=-1, b=2, c=3, d=4)
    assert result["computable"] is False


def test_analyze_prr_flags_signal_and_ranks():
    rows = []
    rows.extend([{"drug_name": "HeptraClear", "adverse_event": "Liver Injury"}] * 10)
    rows.extend([{"drug_name": "HeptraClear", "adverse_event": "Nausea"}] * 2)
    rows.extend([{"drug_name": "Neuravex", "adverse_event": "Headache"}] * 12)
    rows.extend([{"drug_name": "Neuravex", "adverse_event": "Liver Injury"}] * 1)
    rows.extend([{"drug_name": "Cardionil", "adverse_event": "Dizziness"}] * 10)
    frame = pd.DataFrame(rows)
    result = analyze_prr(frame, min_a=3, min_prr=2.0)
    flagged = result[result["Signal Status"] == STATUS_SIGNAL]
    assert not flagged.empty
    top = flagged.iloc[0]
    assert top["Drug"] == "HeptraClear"
    assert top["Adverse Event"] == "Liver Injury"
    assert top["PRR"] > 2


def test_analyze_prr_marks_small_a_insufficient():
    frame = pd.DataFrame(
        [
            {"drug_name": "A", "adverse_event": "X"},
            {"drug_name": "A", "adverse_event": "Y"},
            {"drug_name": "A", "adverse_event": "Z"},
            {"drug_name": "B", "adverse_event": "X"},
            {"drug_name": "B", "adverse_event": "Y"},
            {"drug_name": "B", "adverse_event": "Z"},
        ]
    )
    result = analyze_prr(frame, min_a=3, min_prr=2.0)
    assert (result["Signal Status"] == STATUS_INSUFFICIENT).all()


def test_undefined_status_when_event_only_for_one_drug():
    frame = pd.DataFrame(
        [{"drug_name": "Solo", "adverse_event": "Rare Event"}] * 5
        + [{"drug_name": "Other", "adverse_event": "Headache"}] * 10
    )
    result = analyze_prr(frame, min_a=3, min_prr=2.0)
    rare = result[
        (result["Drug"] == "Solo") & (result["Adverse Event"] == "Rare Event")
    ].iloc[0]
    assert rare["Signal Status"] == STATUS_UNDEFINED
    assert rare["PRR"] is None or pd.isna(rare["PRR"])


def test_csv_missing_required_columns():
    raw = StringIO("foo,bar\n1,2\n")
    frame = load_adverse_event_csv(raw)
    with pytest.raises(DataValidationError, match="Required column missing"):
        clean_adverse_events(frame)


def test_empty_csv_raises():
    with pytest.raises(DataValidationError):
        load_adverse_event_csv(StringIO(""))


def test_cleaning_drops_missing_and_duplicates():
    raw = StringIO(
        "drug_name,adverse_event\n"
        "A,Headache\n"
        "A,Headache\n"
        ",Nausea\n"
        "B,\n"
        "B,Rash\n"
    )
    frame = load_adverse_event_csv(raw)
    cleaned, report = clean_adverse_events(frame)
    assert report["duplicate_rows_removed"] == 1
    assert report["invalid_or_missing_required"] == 2
    assert set(cleaned["adverse_event"]) == {"Headache", "Rash"}


def test_parse_dossier_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_dossier_text("   \n")


def test_submission_completeness_and_missing_sections():
    provided = [
        "Cover Letter",
        "Application Form",
        "Quality Overall Summary",
        "Drug Substance",
        "Clinical Overview",
    ]
    result = evaluate_dossier(provided)
    assert result["present_total"] == 5
    assert "Nonclinical Overview" not in set(
        result["gap_report"]
        .loc[result["gap_report"]["Status"] == "Present", "Section"]
    )
    assert result["missing_total"] > 0
    gaps = result["gap_report"]
    missing = gaps[gaps["Status"] == "Missing"]
    assert "Stability" in set(missing["Section"])
    assert 0 <= result["overall_score"] <= 100
    module_3 = result["module_scores"][
        result["module_scores"]["Module"].str.contains("Module 3")
    ].iloc[0]
    assert module_3["Present"] == 1
    assert module_3["Completeness %"] == pytest.approx(12.5)
    assert result["missing_high"] >= 1
    assert result["missing_medium"] >= 0
    assert result["missing_low"] >= 0
    assert "weighted average" in result["score_formula"]


def test_pipeline_sample_data_produces_signals_and_readiness():
    from pipeline import load_reports, run_dossier_pipeline, run_signal_pipeline
    from utils import SAMPLE_ADVERSE_EVENTS, SAMPLE_DOSSIER_OUTLINE

    frame = load_reports(SAMPLE_ADVERSE_EVENTS)
    payload = run_signal_pipeline(frame, min_a=3, min_prr=2.0)
    assert payload["stats"]["total_reports"] > 0
    assert payload["n_signals"] >= 1
    assert "Drug" in payload["results"].columns
    assert "PRR" in payload["results"].columns
    assert "ROR" in payload["results"].columns
    assert "Chi-square" in payload["results"].columns
    assert "ML Anomaly" in payload["results"].columns
    assert "Reviewer Priority Score" in payload["results"].columns
    assert "Method Agreement" in payload["results"].columns
    assert "Cases with severity keywords" in payload["results"].columns
    assert payload["clean_report"]["event_labels_mapped"] >= 1
    assert payload["clustering"]["method"] in {"tfidf_kmeans", "categorical_frequency"}
    dossier = run_dossier_pipeline(SAMPLE_DOSSIER_OUTLINE.read_text(encoding="utf-8"))
    assert 0 <= dossier["overall_score"] <= 100
    assert dossier["missing_total"] > 0
    missing = dossier["gap_report"][dossier["gap_report"]["Status"] == "Missing"]
    assert set(missing["Priority"]).issubset({"High", "Medium", "Low"})


def test_invalid_csv_and_empty_file_messages():
    with pytest.raises(DataValidationError, match="Invalid CSV file"):
        load_adverse_event_csv(StringIO("this is not,valid\n\"unterminated"))
    with pytest.raises(DataValidationError, match="Invalid CSV file"):
        load_adverse_event_csv(StringIO(""))


def test_ror_and_chi_square_handle_zeros():
    from ml_screening import compute_chi_square, compute_ror, enrich_signal_table

    ok = compute_ror(8, 12, 4, 76)
    assert ok["ror"] == pytest.approx((8 * 76) / (12 * 4))
    bad = compute_ror(5, 5, 0, 90)
    assert bad["ror"] is None
    chi = compute_chi_square(8, 12, 4, 76)
    assert chi["chi2"] is not None and chi["chi2"] > 0
    empty_chi = compute_chi_square(0, 0, 0, 0)
    assert empty_chi["chi2"] is None

    mini = pd.DataFrame(
        {
            "Drug": [f"D{i}" for i in range(16)],
            "Adverse Event": ["X"] * 16,
            "a": [4] * 8 + [2] * 8,
            "b": [4] * 16,
            "c": [2] * 16,
            "d": [20] * 16,
            "PRR": [3.0] * 8 + [1.1] * 8,
            "Signal Status": ["Potential Safety Signal"] * 8 + ["Below Threshold"] * 8,
        }
    )
    out = enrich_signal_table(mini)
    assert len(out) == 16
    assert out["Reviewer Priority Score"].between(0, 100).all()
    assert set(out["ML Anomaly"].unique()).issubset({"Anomalous pair", "Typical pair"})
    assert "Method Agreement" in out.columns


def test_synonyms_severity_and_year_counts():
    from quality import (
        canonicalize_term,
        cases_for_pair,
        counts_by_year,
        DRUG_SYNONYMS,
        EVENT_SYNONYMS,
        flag_severity_text,
    )

    assert canonicalize_term("heptraclear", DRUG_SYNONYMS) == "HeptraClear"
    assert canonicalize_term("  HeptraClear  ", DRUG_SYNONYMS) == "HeptraClear"
    assert canonicalize_term("hepatic injury", EVENT_SYNONYMS) == "Liver Injury"
    assert "hospitalized" in flag_severity_text("Patient was hospitalized after dosing")
    assert "fatal" in flag_severity_text("Outcome listed as fatal")

    raw = StringIO(
        "drug_name,adverse_event,report_year,narrative,outcome\n"
        "heptraclear,hepatic injury,2024,hospitalized with jaundice,Hospitalized\n"
        "HeptraClear,Liver Injury,2025,mild,Recovered\n"
    )
    cleaned, report = clean_adverse_events(load_adverse_event_csv(raw))
    assert report["drug_labels_mapped"] >= 1
    assert report["event_labels_mapped"] >= 1
    assert set(cleaned["drug_name"]) == {"HeptraClear"}
    assert set(cleaned["adverse_event"]) == {"Liver Injury"}
    pair = cases_for_pair(cleaned, "HeptraClear", "Liver Injury")
    assert len(pair) == 2
    years = counts_by_year(cleaned, drug="HeptraClear", event="Liver Injury")
    assert set(years["report_year"]) == {2024, 2025}

    from ml_screening import method_agreement

    assert method_agreement(3, 4, 2, 2) == "PRR and ROR both high"
    assert method_agreement(3, 1.2, 2, 2) == "Only PRR high"
    assert method_agreement(1.1, 5, 2, 2) == "Only ROR high"
    assert method_agreement(1.1, 1.1, 2, 2) == "Neither high"


