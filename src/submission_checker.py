"""ICH M4 CTD representative completeness checking.

This checklist is a documented, representative outline for the prototype.
It is not a complete regulatory submission checklist.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

from utils import normalize_label

# Representative expected sections by ICH M4 CTD module.
# Each item is (section_code, document, priority). 5.9 and 5.10 are demo
# checklist items, not official ICH M4 leaf codes.
REPRESENTATIVE_CHECKLIST: dict[str, dict] = {
    "Module 1 — Administrative Information and Prescribing Information": {
        "weight": 0.10,
        "sections": [
            ("1.1", "Cover Letter", "Medium"),
            ("1.2", "Application Form", "High"),
            ("1.3", "Product Information / Labeling", "High"),
            ("1.4", "Pharmacovigilance System Summary", "Medium"),
            ("1.5", "Environmental Risk Assessment", "Low"),
        ],
    },
    "Module 2 — Common Technical Document Summaries": {
        "weight": 0.20,
        "sections": [
            ("2.1", "CTD Table of Contents", "Medium"),
            ("2.2", "Introduction", "Medium"),
            ("2.3", "Quality Overall Summary", "High"),
            ("2.4", "Nonclinical Overview", "High"),
            ("2.5", "Clinical Overview", "High"),
            ("2.6", "Nonclinical Written and Tabulated Summaries", "Medium"),
            ("2.7", "Clinical Summary", "High"),
        ],
    },
    "Module 3 — Quality": {
        "weight": 0.25,
        "sections": [
            ("3.2.S", "Drug Substance", "High"),
            ("3.2.P", "Drug Product", "High"),
            ("3.2.P.2", "Pharmaceutical Development", "High"),
            ("3.2.P.3", "Manufacture", "High"),
            ("3.2.P.5", "Control of Drug Product", "High"),
            ("3.2.P.8", "Stability", "High"),
            ("3.2.P.7", "Container Closure System", "Medium"),
            ("3.2.A", "Appendices", "Low"),
        ],
    },
    "Module 4 — Nonclinical Study Reports": {
        "weight": 0.15,
        "sections": [
            ("4.2", "Pharmacology Study Reports", "High"),
            ("4.3", "Pharmacokinetics Study Reports", "High"),
            ("4.4", "Toxicology Study Reports", "High"),
            ("4.R", "Nonclinical Literature References", "Low"),
        ],
    },
    "Module 5 — Clinical Study Reports": {
        "weight": 0.30,
        "sections": [
            ("5.2", "Tabular Listing of Clinical Studies", "High"),
            ("5.3.1", "Biopharmaceutic Study Reports", "Medium"),
            ("5.3.6", "Clinical Safety Study Reports", "High"),
            ("5.3.5", "Clinical Efficacy Study Reports", "High"),
            ("5.3.7", "Case Report Forms", "Medium"),
            ("5.4", "Clinical Literature References", "Low"),
            ("5.9", "Efficacy Summary", "High"),
            ("5.10", "Integrated Benefit-Risk Summary", "High"),
        ],
    },
}

DEMO_CHECKLIST_CODES = {"5.9", "5.10"}
CHECKLIST_TITLE = "ICH M4 CTD Submission Readiness — Demo Checklist"
DOSSIER_FORMAT_HELP = (
    "Expected dossier CSV columns: Section, Document, Status, Notes. "
    "Status should be Present or Missing. Example:\n"
    "Section,Document,Status,Notes\n"
    "2.5,Clinical Overview,Missing,Needs finalization\n"
    "5.9,Efficacy Summary,Present,Demo checklist item (not a standard M4 leaf)\n"
    "5.10,Integrated Benefit-Risk Summary,Missing,Demo checklist item\n\n"
    "You can also paste one section per line, e.g. Module 3: Drug Substance."
)

RECOMMENDATIONS = {
    "High": "Add the required representative section before claiming module completeness.",
    "Medium": "Include this representative section to strengthen the dossier outline.",
    "Low": "Add this section after high-priority gaps are closed.",
}

# ICH M4 codes and common document titles -> representative checklist names.
ICH_CODE_MAP = {
    "1": ["Cover Letter", "Application Form"],
    "1.0": ["Cover Letter", "Application Form"],
    "1.1": ["Cover Letter"],
    "1.2": ["Application Form"],
    "1.3": ["Product Information / Labeling"],
    "2.1": ["CTD Table of Contents"],
    "2.2": ["Introduction"],
    "2.3": ["Quality Overall Summary"],
    "2.4": ["Nonclinical Overview"],
    "2.5": ["Clinical Overview"],
    "2.6": ["Nonclinical Written and Tabulated Summaries"],
    "2.7": ["Clinical Summary"],
    "3.2.s": ["Drug Substance"],
    "3.2.p": ["Drug Product", "Manufacture"],
    "3.2.a": ["Appendices"],
    "3.2.p.2": ["Pharmaceutical Development"],
    "3.2.p.5": ["Control of Drug Product"],
    "3.2.p.8": ["Stability"],
    "3.2.p.7": ["Container Closure System"],
    "4.2": ["Pharmacology Study Reports"],
    "4.3": ["Pharmacokinetics Study Reports"],
    "4.4": ["Toxicology Study Reports"],
    "5.2": ["Tabular Listing of Clinical Studies"],
    "5.3": ["Clinical Safety Study Reports", "Clinical Efficacy Study Reports"],
    "5.3.1": ["Biopharmaceutic Study Reports"],
    "5.3.5": ["Clinical Efficacy Study Reports"],
    "5.3.6": ["Clinical Safety Study Reports"],
    "5.4": ["Clinical Literature References"],
    "5.5": ["Clinical Summary"],
    "5.7": ["Clinical Safety Study Reports"],
    "5.9": ["Efficacy Summary"],
    "5.10": ["Integrated Benefit-Risk Summary"],
}

DOCUMENT_ALIASES = {
    "cover letter": ["Cover Letter"],
    "application form": ["Application Form"],
    "administrative information": ["Cover Letter", "Application Form"],
    "product information": ["Product Information / Labeling"],
    "labeling": ["Product Information / Labeling"],
    "product information / labeling": ["Product Information / Labeling"],
    "pharmacovigilance system summary": ["Pharmacovigilance System Summary"],
    "environmental risk assessment": ["Environmental Risk Assessment"],
    "ctd table of contents": ["CTD Table of Contents"],
    "table of contents": ["CTD Table of Contents"],
    "introduction": ["Introduction"],
    "quality overall summary": ["Quality Overall Summary"],
    "nonclinical overview": ["Nonclinical Overview"],
    "clinical overview": ["Clinical Overview"],
    "nonclinical written and tabulated summaries": ["Nonclinical Written and Tabulated Summaries"],
    "clinical summary": ["Clinical Summary"],
    "drug substance": ["Drug Substance"],
    "drug product": ["Drug Product"],
    "pharmaceutical development": ["Pharmaceutical Development"],
    "manufacture": ["Manufacture"],
    "manufacturing": ["Manufacture"],
    "control of drug product": ["Control of Drug Product"],
    "stability": ["Stability"],
    "container closure system": ["Container Closure System"],
    "appendices": ["Appendices"],
    "pharmacology": ["Pharmacology Study Reports"],
    "pharmacology study reports": ["Pharmacology Study Reports"],
    "pharmacokinetics": ["Pharmacokinetics Study Reports"],
    "pharmacokinetics study reports": ["Pharmacokinetics Study Reports"],
    "pk": ["Pharmacokinetics Study Reports"],
    "toxicology": ["Toxicology Study Reports"],
    "toxicology study reports": ["Toxicology Study Reports"],
    "nonclinical literature references": ["Nonclinical Literature References"],
    "tabular listing of clinical studies": ["Tabular Listing of Clinical Studies"],
    "biopharmaceutic study reports": ["Biopharmaceutic Study Reports"],
    "clinical safety study reports": ["Clinical Safety Study Reports"],
    "clinical efficacy study reports": ["Clinical Efficacy Study Reports"],
    "clinical study reports": ["Clinical Safety Study Reports", "Clinical Efficacy Study Reports"],
    "case report forms": ["Case Report Forms"],
    "clinical literature references": ["Clinical Literature References"],
    "literature references": ["Clinical Literature References"],
    "safety summary": ["Clinical Safety Study Reports"],
    "efficacy summary": ["Efficacy Summary"],
    "integrated benefit-risk summary": ["Integrated Benefit-Risk Summary"],
    "benefit-risk summary": ["Integrated Benefit-Risk Summary"],
    "integrated benefit risk summary": ["Integrated Benefit-Risk Summary"],
}

_GENERIC_FIRST_WORDS = {
    "clinical",
    "quality",
    "drug",
    "product",
    "table",
    "module",
    "study",
    "reports",
    "report",
    "summary",
    "information",
    "contents",
    "nonclinical",
}

_AE_HEADER_HINTS = {
    "drug_name",
    "drug",
    "adverse_event",
    "event",
    "pt",
    "preferred_term",
}

_MISSING_STATUS = {"missing", "absent", "gap", "0", "false", "no", "incomplete", "not present"}
_PRESENT_STATUS = {"present", "complete", "available", "yes", "true", "1", "included", "done"}


def expected_sections() -> list[tuple[str, str, str, str]]:
    rows = []
    for module, spec in REPRESENTATIVE_CHECKLIST.items():
        for code, section, priority in spec["sections"]:
            rows.append((module, code, section, priority))
    return rows


def _checklist_names() -> list[str]:
    return [section for _module, _code, section, _priority in expected_sections()]


def _ich_code(value: str) -> str | None:
    text = str(value).strip().lower().rstrip(".")
    if re.fullmatch(r"\d+(?:\.\d+)*\.?[a-z]?", text):
        return text
    return None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def map_label_to_checklist(label: str, notes: str = "") -> list[str]:
    """Map an uploaded section code or document title onto checklist names."""
    raw = str(label).strip()
    if not raw:
        return []
    stripped = raw
    for sep in ("|", ":", " - ", " — "):
        if sep in stripped:
            left, right = stripped.split(sep, 1)
            if _ich_code(left.strip()):
                stripped = right.strip() or stripped
            else:
                stripped = [p.strip() for p in stripped.split(sep) if p.strip()][-1]
            break

    hits: list[str] = []
    code = _ich_code(raw) or _ich_code(str(label).split()[0]) or _ich_code(stripped)
    if code and code in ICH_CODE_MAP:
        hits.extend(ICH_CODE_MAP[code])
    # 3.2.S style inside a longer string
    match = re.search(r"\b(\d+(?:\.\d+)*\.?[a-zA-Z]?)\b", raw)
    if match:
        found = match.group(1).lower()
        if found in ICH_CODE_MAP:
            hits.extend(ICH_CODE_MAP[found])

    key = normalize_label(stripped)
    key = re.sub(r"^module \d+\s*", "", key).strip()
    if key in DOCUMENT_ALIASES:
        hits.extend(DOCUMENT_ALIASES[key])

    blob = f"{raw} {notes}".lower()
    if "manufactur" in blob:
        hits.append("Manufacture")
    if "application form" in blob:
        hits.append("Application Form")
    if "label" in blob and "table of contents" not in key:
        hits.append("Product Information / Labeling")

    expected = _checklist_names()
    for section in expected:
        target = normalize_label(section)
        if key == target:
            hits.append(section)
            continue
        first = target.split()[0]
        if key == first and first not in _GENERIC_FIRST_WORDS:
            hits.append(section)
            continue
        if target.startswith(key + " ") and len(key.split()) >= 2:
            hits.append(section)

    return _unique([name for name in hits if name in set(expected)])


def _pick_column(lookup: dict[str, str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def _row_is_present(status: str | None) -> bool:
    if status is None or not str(status).strip():
        return True
    token = str(status).strip().lower()
    if token in _MISSING_STATUS:
        return False
    if token in _PRESENT_STATUS:
        return True
    return "miss" not in token and "absent" not in token


def _parse_structured_frame(df: pd.DataFrame) -> dict | None:
    if df.empty:
        raise ValueError("Dossier CSV is empty. " + DOSSIER_FORMAT_HELP)
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    lookup.update({str(c).strip().lower(): c for c in df.columns})
    if set(lookup) & _AE_HEADER_HINTS and not (
        {"section", "document", "dossier_section"} & set(lookup)
    ):
        raise ValueError(
            "This CSV looks like adverse-event reports, not a dossier outline. "
            "Open Signal Detection and upload it there."
        )

    section_col = _pick_column(
        lookup,
        ["section", "section_name", "dossier_section", "sections", "code", "ich", "module"],
    )
    document_col = _pick_column(
        lookup,
        ["document", "title", "name", "document_name", "item"],
    )
    status_col = _pick_column(lookup, ["status", "completeness", "present"])
    notes_col = _pick_column(lookup, ["notes", "comment", "comments", "remark"])

    if section_col is None and document_col is None:
        if len(df.columns) == 1:
            first_col = df.columns[0]
            mapped: list[str] = []
            unmapped: list[str] = []
            for val in df[first_col].dropna():
                hits = map_label_to_checklist(str(val))
                if hits:
                    mapped.extend(hits)
                else:
                    unmapped.append(str(val).strip())
            return {
                "present": _unique(mapped),
                "unmapped": _unique(unmapped),
                "unknown_codes": [],
                "invalid_status": [],
                "known_missing": 0,
                "row_count": int(len(df)),
                "structured": True,
            }
        found = ", ".join(str(c) for c in df.columns)
        raise ValueError(
            "Unrecognized dossier columns: "
            + found
            + ". "
            + DOSSIER_FORMAT_HELP
        )

    present: list[str] = []
    unmapped: list[str] = []
    unknown_codes: list[str] = []
    invalid_status: list[str] = []
    known_missing = 0
    seen_keys: list[str] = []
    for rec in df.to_dict(orient="records"):
        status_val = rec.get(status_col) if status_col else None
        token = "" if status_val is None else str(status_val).strip().lower()
        if token and token not in _MISSING_STATUS and token not in _PRESENT_STATUS and token not in {"", "nan"}:
            invalid_status.append(str(status_val).strip())
        notes = str(rec.get(notes_col, "") or "")
        section_raw = "" if section_col is None or rec.get(section_col) is None else str(rec[section_col]).strip()
        document_raw = "" if document_col is None or rec.get(document_col) is None else str(rec[document_col]).strip()
        key = f"{section_raw}|{document_raw}".lower()
        seen_keys.append(key)
        parts: list[str] = []
        if section_raw:
            parts.extend(map_label_to_checklist(section_raw, notes))
        if document_raw:
            parts.extend(map_label_to_checklist(document_raw, notes))
        parts = _unique(parts)
        code = _ich_code(section_raw)
        if code and code not in ICH_CODE_MAP and not parts:
            unknown_codes.append(section_raw)
        if not parts:
            label = document_raw or section_raw
            if label:
                unmapped.append(label)
            continue
        if not _row_is_present(None if status_val is None else str(status_val)):
            known_missing += 1
            continue
        present.extend(parts)

    duplicates = [k for k in seen_keys if seen_keys.count(k) > 1 and k != "|"]
    return {
        "present": _unique(present),
        "unmapped": _unique(unmapped),
        "unknown_codes": _unique(unknown_codes),
        "invalid_status": _unique(invalid_status),
        "known_missing": known_missing,
        "row_count": int(len(df)),
        "duplicates": _unique(duplicates),
        "structured": True,
    }


def parse_dossier_text(text: str) -> list[str]:
    """Parse pasted or uploaded outline lines into section labels."""
    if text is None or not str(text).strip():
        raise ValueError("Dossier input is empty. " + DOSSIER_FORMAT_HELP)

    cleaned_text = str(text).strip()
    first_line = cleaned_text.splitlines()[0].lower() if cleaned_text.splitlines() else ""
    looks_csv_table = ("," in first_line or "\t" in first_line) and not first_line.lstrip().startswith("#")
    if looks_csv_table:
        try:
            df = pd.read_csv(StringIO(cleaned_text))
            parsed = _parse_structured_frame(df)
            if parsed is not None:
                present = parsed["present"]
                if parsed.get("invalid_status"):
                    raise ValueError(
                        "Invalid Status values: "
                        + ", ".join(parsed["invalid_status"])
                        + ". Use Present or Missing. "
                        + DOSSIER_FORMAT_HELP
                    )
                if parsed.get("duplicates"):
                    raise ValueError(
                        "Duplicate dossier sections in the file: "
                        + ", ".join(parsed["duplicates"][:8])
                        + ". Keep one row per section. "
                        + DOSSIER_FORMAT_HELP
                    )
                if (
                    parsed["row_count"]
                    and not present
                    and not parsed["known_missing"]
                ):
                    extra = ""
                    if parsed["unknown_codes"]:
                        extra = " Unknown section codes: " + ", ".join(parsed["unknown_codes"]) + "."
                    if parsed["unmapped"]:
                        extra += " Unrecognized documents: " + ", ".join(parsed["unmapped"][:8]) + "."
                    raise ValueError(
                        "No checklist sections could be matched from this file."
                        + extra
                        + " "
                        + DOSSIER_FORMAT_HELP
                    )
                return present
        except ValueError:
            raise
        except Exception:
            pass

    lines = []
    unmapped_lines = []
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
        mapped = map_label_to_checklist(line)
        if mapped:
            lines.extend(mapped)
        else:
            lines.append(line)
            if line.lower() not in {normalize_label(n) for n in _checklist_names()}:
                unmapped_lines.append(line)
    lines = _unique([item for item in lines if item])
    if not lines:
        raise ValueError(
            "No dossier sections could be parsed. " + DOSSIER_FORMAT_HELP
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
        mapped = map_label_to_checklist(item)
        if section in mapped:
            return True
    return False


def evaluate_dossier(provided_sections: list[str] | None) -> dict:
    """Score representative CTD completeness and build a gap report."""
    if provided_sections is None:
        raise ValueError("No dossier sections were provided. " + DOSSIER_FORMAT_HELP)

    provided_sections = list(provided_sections)

    module_rows = []
    gap_rows = []
    present_total = 0
    expected_total = 0
    weighted = 0.0

    for module, spec in REPRESENTATIVE_CHECKLIST.items():
        expected = spec["sections"]
        present = []
        missing = []
        for code, section, priority in expected:
            expected_total += 1
            demo_note = (
                "Demo checklist item — not a standard ICH M4 leaf code."
                if code in DEMO_CHECKLIST_CODES
                else "Representative ICH M4-based demo checklist item."
            )
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
                    "Section Code": code,
                    "Document": section,
                    "Section": section,
                    "Status": status,
                    "Notes": demo_note,
                    "Priority": priority,
                    "Weight": spec["weight"],
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
            "Module 1 10%, Module 2 20%, Module 3 25%, Module 4 15%, Module 5 30%. "
            "This is the ICH M4 CTD Submission Readiness — Demo Checklist "
            "(not complete official eCTD validation)."
        ),
    }
