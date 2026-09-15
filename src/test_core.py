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
    assert report["duplicate_rows_removed"] == 0
    assert report["invalid_or_missing_required"] == 2
    assert len(cleaned) == 3
    assert cleaned["adverse_event"].tolist().count("Headache") == 2
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


def test_repeated_reports_are_counted_for_prr():
    raw = StringIO(
        "drug_name,adverse_event\n"
        + "Drug-A,Headache\n" * 9
        + "Drug-A,Dizziness\n" * 4
        + "Drug-A,Nausea\n" * 2
        + "Drug-B,Nausea\n" * 2
        + "Drug-B,Rash\n" * 2
        + "Drug-B,Fatigue\n"
        + "Drug-B,Headache\n"
        + "Drug-C,Fatigue\n" * 3
        + "Drug-C,Nausea\n" * 3
        + "Drug-C,Rash\n" * 2
        + "Drug-C,Headache\n"
    )
    from pipeline import run_signal_pipeline

    frame = load_adverse_event_csv(raw)
    payload = run_signal_pipeline(frame, min_a=3, min_prr=2.0)
    assert payload["stats"]["total_reports"] == 30
    results = payload["results"]
    ha = results[(results["Drug"] == "Drug-A") & (results["Adverse Event"] == "Headache")].iloc[0]
    assert ha["a"] == 9
    assert ha["PRR"] == pytest.approx(4.5)
    assert ha["Signal Status"] == STATUS_SIGNAL
    cf = results[(results["Drug"] == "Drug-C") & (results["Adverse Event"] == "Fatigue")].iloc[0]
    assert cf["a"] == 3
    assert cf["PRR"] == pytest.approx(7.0)
    assert payload["n_signals"] >= 1


def test_alias_columns_are_accepted():
    raw = StringIO("Drug,Event\nAlpha,Rash\nBeta,Rash\n")
    cleaned, _report = clean_adverse_events(load_adverse_event_csv(raw))
    assert list(cleaned["drug_name"]) == ["Alpha", "Beta"]
    assert list(cleaned["adverse_event"]) == ["Rash", "Rash"]


def test_status_csv_maps_ich_documents_and_skips_missing_rows():
    raw = """Section,Document,Status,Notes
1.0,Administrative Information,Present,Application form and administrative details available
2.1,CTD Table of Contents,Present,CTD structure and table of contents included
2.2,Introduction,Present,Product introduction and development overview included
2.3,Quality Overall Summary,Present,Quality summary document available
2.4,Nonclinical Overview,Present,Nonclinical development overview available
2.5,Clinical Overview,Missing,Clinical overview requires finalization
3.2.S,Drug Substance,Present,Drug substance information available
3.2.P,Drug Product,Present,Drug product manufacturing information available
3.2.A,Appendices,Missing,Quality appendices are incomplete
4.2,Pharmacology,Present,Primary and secondary pharmacology reports available
4.3,Pharmacokinetics,Present,PK study reports available
4.4,Toxicology,Missing,Final toxicology report is missing
5.2,Tabular Listing of Clinical Studies,Present,Clinical study listing included
5.3,Clinical Study Reports,Present,Major clinical study reports included
5.4,Literature References,Missing,Final literature reference package is missing
5.5,Clinical Summary,Present,Clinical summary available
5.7,Safety Summary,Present,Integrated safety summary available
"""
    result = evaluate_dossier(parse_dossier_text(raw))
    present = set(
        result["gap_report"].loc[result["gap_report"]["Status"] == "Present", "Section"]
    )
    missing = set(
        result["gap_report"].loc[result["gap_report"]["Status"] == "Missing", "Section"]
    )
    assert "Application Form" in present
    assert "CTD Table of Contents" in present
    assert "Drug Substance" in present
    assert "Pharmacology Study Reports" in present
    assert "Clinical Summary" in present
    assert "Clinical Overview" in missing
    assert "Toxicology Study Reports" in missing
    assert "Appendices" in missing
    assert "Clinical Literature References" in missing
    assert result["present_total"] > 0
    assert result["overall_score"] > 0


def test_dossier_rejects_adverse_event_csv():
    with pytest.raises(ValueError, match="Signal Detection"):
        parse_dossier_text("drug_name,adverse_event\nDrug-A,Headache\n")


def test_bundled_novalexa_demo_prr():
    from pipeline import run_signal_pipeline
    from utils import SAMPLE_SIGNAL_DEMO, STATUS_BELOW, STATUS_INSUFFICIENT

    frame = load_adverse_event_csv(SAMPLE_SIGNAL_DEMO)
    payload = run_signal_pipeline(frame, min_a=3, min_prr=2.0)
    assert payload["stats"]["total_reports"] == 116
    results = payload["results"]
    row = results[(results["Drug"] == "Novalexa") & (results["Adverse Event"] == "Liver Injury")].iloc[0]
    assert row["a"] == 18
    assert row["PRR"] == pytest.approx(6.4054, rel=1e-3)
    assert row["Signal Status"] == STATUS_SIGNAL
    assert "Hepatic Injury" in str(row.get("Original Event", ""))
    common = results[(results["Drug"] == "Novalexa") & (results["Adverse Event"] == "Headache")].iloc[0]
    assert common["Signal Status"] == STATUS_BELOW
    sparse = results[results["a"] < 3]
    assert (sparse["Signal Status"] == STATUS_INSUFFICIENT).all()
    assert payload["emerging_reason"] in {"ok", "insufficient_years"}


def test_emerging_signal_requires_growth_and_prr():
    from pipeline import run_signal_pipeline

    rows = []
    rows.extend([{"drug_name": "Solo", "adverse_event": "Hepatic Injury", "report_year": 2022}] * 3)
    rows.extend([{"drug_name": "Solo", "adverse_event": "Hepatic Injury", "report_year": 2025}] * 10)
    rows.extend([{"drug_name": "Other", "adverse_event": "Hepatic Injury", "report_year": 2022}] * 2)
    rows.extend([{"drug_name": "Other", "adverse_event": "Headache", "report_year": 2022}] * 12)
    rows.extend([{"drug_name": "Other", "adverse_event": "Headache", "report_year": 2025}] * 12)
    payload = run_signal_pipeline(pd.DataFrame(rows), min_a=3, min_prr=2.0)
    emerging = payload["emerging"]
    assert payload["emerging_reason"] == "ok"
    assert not emerging.empty
    assert emerging.iloc[0]["Drug"] == "Solo"
    assert emerging.iloc[0]["Adverse Event"] == "Liver Injury"
    headache = payload["results"][
        (payload["results"]["Drug"] == "Other") & (payload["results"]["Adverse Event"] == "Headache")
    ].iloc[0]
    assert headache["Signal Status"] != STATUS_SIGNAL


def test_emerging_fallback_without_year():
    from pipeline import run_signal_pipeline

    rows = [{"drug_name": "A", "adverse_event": "X"}] * 5 + [{"drug_name": "B", "adverse_event": "Y"}] * 5
    payload = run_signal_pipeline(pd.DataFrame(rows), min_a=3, min_prr=2.0)
    assert payload["emerging_reason"] == "no_year"
    assert payload["emerging"].empty


def test_undefined_prr_and_synonym_columns():
    from pipeline import run_signal_pipeline

    frame = pd.DataFrame(
        [{"drug_name": "Solo", "adverse_event": "Rare Event"}] * 5
        + [{"drug_name": "Other", "adverse_event": "Headache"}] * 10
    )
    payload = run_signal_pipeline(frame, min_a=3, min_prr=2.0)
    rare = payload["results"][
        (payload["results"]["Drug"] == "Solo") & (payload["results"]["Adverse Event"] == "Rare Event")
    ].iloc[0]
    assert rare["Signal Status"] == STATUS_UNDEFINED
    assert "Original Event" in payload["results"].columns


def test_demo_checklist_5_9_and_5_10():
    raw = """Section,Document,Status,Notes
2.5,Clinical Overview,Present,ok
5.9,Efficacy Summary,Missing,demo
5.10,Integrated Benefit-Risk Summary,Present,demo
"""
    result = evaluate_dossier(parse_dossier_text(raw))
    gap = result["gap_report"]
    e9 = gap[gap["Section Code"] == "5.9"].iloc[0]
    e10 = gap[gap["Section Code"] == "5.10"].iloc[0]
    assert e9["Document"] == "Efficacy Summary"
    assert e9["Status"] == "Missing"
    assert "not a standard ICH M4" in e9["Notes"]
    assert e10["Status"] == "Present"
    assert result["module_scores"]["Completeness %"].between(0, 100).all()
    assert 0 <= result["overall_score"] <= 100
    assert "Efficacy Summary" in set(gap.loc[gap["Status"] == "Missing", "Section"])


def test_modules_one_through_five_present_in_gap():
    result = evaluate_dossier(["Cover Letter"])
    modules = result["module_scores"]["Module"].tolist()
    assert len(modules) == 5
    assert any("Module 1" in m for m in modules)
    assert any("Module 5" in m for m in modules)


def test_unmapped_csv_does_not_score_zero_silently():
    with pytest.raises(ValueError, match="Unrecognized dossier columns"):
        parse_dossier_text("id,target\n1,0\n2,1\n")


def test_unknown_ctd_code_errors():
    with pytest.raises(ValueError, match="Unknown section codes|No checklist"):
        parse_dossier_text("Section,Document,Status,Notes\n9.9,Not A Real Document,Present,x\n")


def test_empty_dossier_input():
    with pytest.raises(ValueError, match="empty"):
        parse_dossier_text("   \n")


def test_invalid_status_values():
    with pytest.raises(ValueError, match="Invalid Status"):
        parse_dossier_text("Section,Document,Status\n2.5,Clinical Overview,Maybe\n")


def test_duplicate_dossier_sections():
    with pytest.raises(ValueError, match="Duplicate dossier"):
        parse_dossier_text(
            "Section,Document,Status\n2.5,Clinical Overview,Present\n2.5,Clinical Overview,Present\n"
        )


def test_explain_signal_highlights_uses_calculated_fields():
    from explanations import explain_signal_highlights
    from pipeline import run_signal_pipeline
    from utils import SAMPLE_SIGNAL_DEMO

    frame = load_adverse_event_csv(SAMPLE_SIGNAL_DEMO)
    payload = run_signal_pipeline(frame, min_a=3, min_prr=2.0)
    row = payload["results"][
        (payload["results"]["Drug"] == "Novalexa")
        & (payload["results"]["Adverse Event"] == "Liver Injury")
    ].iloc[0]
    bullets = explain_signal_highlights(row.to_dict(), min_a=3, min_prr=2.0)
    joined = " ".join(bullets)
    assert "18" in joined
    assert "2.0" in joined
    assert "Hepatic Injury" in joined
    assert "Liver Injury" in joined
    assert row["PRR"] == pytest.approx(6.4054, rel=1e-3)


def test_dossier_csv_rejected_in_signal_mode():
    frame = pd.DataFrame(
        {
            "Section": ["2.5"],
            "Document": ["Clinical Overview"],
            "Status": ["Present"],
        }
    )
    with pytest.raises(DataValidationError, match="Submission Readiness"):
        clean_adverse_events(frame)


def test_tfidf_kmeans_exposes_cluster_summaries():
    from pipeline import run_signal_pipeline
    from utils import SAMPLE_SIGNAL_DEMO

    payload = run_signal_pipeline(load_adverse_event_csv(SAMPLE_SIGNAL_DEMO), min_a=3, min_prr=2.0)
    clustering = payload["clustering"]
    assert clustering["method"] == "tfidf_kmeans"
    assert clustering["n_clusters"] >= 2
    assert clustering["n_narrative_reports"] >= 8
    summaries = clustering["summaries"]
    assert "top_terms" in summaries.columns
    assert "reports" in summaries.columns
    assert "example_narratives" in summaries.columns
    first = summaries.iloc[0]
    assert str(first["top_terms"]).strip()
    assert int(first["reports"]) >= 1
    examples = first["example_narratives"]
    assert len(list(examples)) >= 1
    row = payload["results"][
        (payload["results"]["Drug"] == "Novalexa")
        & (payload["results"]["Adverse Event"] == "Liver Injury")
    ].iloc[0]
    assert row["PRR"] == pytest.approx(6.4054, rel=1e-3)
    assert row["Signal Status"] == STATUS_SIGNAL





