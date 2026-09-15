"""ICH M4 CTD representative completeness checking.

This checklist is a documented, representative outline for the prototype.
It is not a complete regulatory submission checklist.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from utils import normalize_label

# Representative expected sections by ICH M4 CTD module.
REPRESENTATIVE_CHECKLIST: dict[str, dict] = {
    "Module 1 — Administrative Information and Prescribing Information": {
        "weight": 0.10,
        "sections": [
            ("Cover Letter", "Medium"),
            ("Application Form", "High"),
            ("Product Information / Labeling", "High"),
            ("Pharmacovigilance System Summary", "Medium"),
            ("Environmental Risk Assessment", "Low"),
        ],
    },
    "Module 2 — Common Technical Document Summaries": {
        "weight": 0.20,
        "sections": [
            ("CTD Table of Contents", "Medium"),
            ("Introduction", "Medium"),
            ("Quality Overall Summary", "High"),
            ("Nonclinical Overview", "High"),
            ("Clinical Overview", "High"),
            ("Nonclinical Written and Tabulated Summaries", "Medium"),
            ("Clinical Summary", "High"),
        ],
    },
    "Module 3 — Quality": {
        "weight": 0.25,
        "sections": [
            ("Drug Substance", "High"),
            ("Drug Product", "High"),
            ("Pharmaceutical Development", "High"),
            ("Manufacture", "High"),
            ("Control of Drug Product", "High"),
            ("Stability", "High"),
            ("Container Closure System", "Medium"),
            ("Appendices", "Low"),
        ],
    },
    "Module 4 — Nonclinical Study Reports": {
        "weight": 0.15,
        "sections": [
            ("Pharmacology Study Reports", "High"),
            ("Pharmacokinetics Study Reports", "High"),
            ("Toxicology Study Reports", "High"),
            ("Nonclinical Literature References", "Low"),
        ],
    },
    "Module 5 — Clinical Study Reports": {
        "weight": 0.30,
        "sections": [
            ("Tabular Listing of Clinical Studies", "High"),
            ("Biopharmaceutic Study Reports", "Medium"),
            ("Clinical Safety Study Reports", "High"),
            ("Clinical Efficacy Study Reports", "High"),
            ("Case Report Forms", "Medium"),
            ("Clinical Literature References", "Low"),
        ],
    },
}

RECOMMENDATIONS = {
    "High": "Add the required representative section before claiming module completeness.",
    "Medium": "Include this representative section to strengthen the dossier outline.",
    "Low": "Add this section after high-priority gaps are closed.",
}


def expected_sections() -> list[tuple[str, str, str]]:
    rows = []
    for module, spec in REPRESENTATIVE_CHECKLIST.items():
        for section, priority in spec["sections"]:
            rows.append((module, section, priority))
    return rows


def parse_dossier_text(text: str) -> list[str]:
    """Parse pasted or uploaded outline lines into section labels."""
    if text is None or not str(text).strip():
        raise ValueError("Dossier input is empty. Paste an outline or upload a file.")

    cleaned_text = str(text).strip()
    # 1. Attempt to parse as structured CSV if commas or tabs exist
    if "," in cleaned_text or "\t" in cleaned_text:
        try:
            df = pd.read_csv(StringIO(cleaned_text))
            if not df.empty:
                col_lookup = {str(c).strip().lower(): c for c in df.columns}
                section_col = None
                for candidate in ["section", "section_name", "dossier_section", "sections", "title", "name"]:
                    if candidate in col_lookup:
                        section_col = col_lookup[candidate]
                        break

                if section_col is not None:
                    # If there is a 'status' column (e.g. from an exported gap report)
                    if "status" in col_lookup:
                        status_col = col_lookup["status"]
                        is_present = ~df[status_col].astype(str).str.strip().str.lower().isin(
                            ["missing", "absent", "gap", "0", "false", "no"]
                        )
                        df = df[is_present]

                    sections = [str(val).strip() for val in df[section_col].dropna() if str(val).strip()]
                    if sections:
                        res = []
                        for s in sections:
                            for sep in ("|", ":", " - ", " — "):
                                if sep in s:
                                    s = [p.strip() for p in s.split(sep) if p.strip()][-1]
                                    break
                            res.append(s)
                        return res
                elif len(df.columns) == 1:
                    first_col = df.columns[0]
                    vals = [str(val).strip() for val in df[first_col].dropna() if str(val).strip()]
                    if vals:
                        res = []
                        for s in vals:
                            for sep in ("|", ":", " - ", " — "):
                                if sep in s:
                                    s = [p.strip() for p in s.split(sep) if p.strip()][-1]
                                    break
                            res.append(s)
                        return res
        except Exception:
            pass

    # 2. Line-by-line parsing for text files / pasted text
    lines = []
    for raw in cleaned_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("module,section"):
            continue
        if (
            line.lower().startswith("module ")
            and ":" not in line
            and "|" not in line
            and " — " not in line
            and " - " not in line
            and "," not in line
        ):
            continue
        for sep in (",", "|", ":", " - ", " — "):
            if sep in line:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                if sep == "," and len(parts) >= 2:
                    line = parts[1]
                else:
                    line = parts[-1]
                break
        lines.append(line)
    if not lines:
        raise ValueError(
            "No dossier sections could be parsed. Use one section name per line."
        )
    return lines


def load_dossier_source(source: str | Path) -> list[str]:
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    return parse_dossier_text(text)


def _strip_module_prefix(label: str) -> str:
    text = normalize_label(label)
    for index in range(1, 6):
        prefix = f"module {index} "
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _is_present(section: str, provided: list[str]) -> bool:
    """Match expected sections without substring false positives.

    Example: "Clinical Overview" must not match "Nonclinical Overview".
    """
    target = normalize_label(section)
    for item in provided:
        candidate = _strip_module_prefix(item)
        if candidate == target:
            return True
        if candidate.startswith(f"{target} "):
            return True
    return False


def evaluate_dossier(provided_sections: list[str]) -> dict:
    """Score representative CTD completeness and build a gap report."""
    if not provided_sections:
        raise ValueError("No dossier sections were provided.")

    module_rows = []
    gap_rows = []
    present_total = 0
    expected_total = 0
    weighted = 0.0

    for module, spec in REPRESENTATIVE_CHECKLIST.items():
        expected = spec["sections"]
        present = []
        missing = []
        for section, priority in expected:
            expected_total += 1
            if _is_present(section, provided_sections):
                present.append(section)
                present_total += 1
                status = "Present"
                recommendation = "Section found in the representative outline."
            else:
                missing.append(section)
                status = "Missing"
                recommendation = RECOMMENDATIONS[priority]
            gap_rows.append(
                {
                    "Module": module,
                    "Section": section,
                    "Status": status,
                    "Priority": priority,
                    "Recommendation": recommendation,
                }
            )
        completeness = 100.0 * len(present) / len(expected)
        weighted += completeness * spec["weight"]
        module_rows.append(
            {
                "Module": module,
                "Expected sections": len(expected),
                "Present": len(present),
                "Missing": len(missing),
                "Completeness %": round(completeness, 1),
                "Missing section names": "; ".join(missing) if missing else "None",
            }
        )

    gap_frame = pd.DataFrame(gap_rows)
    missing = gap_frame["Status"] == "Missing"
    missing_high = int((missing & (gap_frame["Priority"] == "High")).sum())
    missing_medium = int((missing & (gap_frame["Priority"] == "Medium")).sum())
    missing_low = int((missing & (gap_frame["Priority"] == "Low")).sum())
    return {
        "overall_score": round(weighted, 1),
        "present_total": present_total,
        "expected_total": expected_total,
        "missing_total": expected_total - present_total,
        "missing_high": missing_high,
        "missing_medium": missing_medium,
        "missing_low": missing_low,
        "module_scores": pd.DataFrame(module_rows),
        "gap_report": gap_frame,
        "provided_count": len(provided_sections),
        "score_formula": (
            "Overall readiness is a weighted average of module completeness: "
            "Module 1 10%, Module 2 20%, Module 3 25%, Module 4 15%, Module 5 30%."
        ),
    }
