# Solution Overview

## What we built

TechTress built a dark-themed Streamlit dashboard with two modes:

- **Signal Detection** — clean a CSV of representative adverse-event reports, detect patterns, calculate PRR, rank **Potential Safety Signals**, visualize findings, and download results.
- **Submission Readiness** — map a dossier outline to a **representative** ICH M4 CTD checklist, score each module, compute an overall readiness score, and download a gap report.

The application runs locally. It does not require IBM Cloud, watsonx.ai, or any other external inference API.

## Solution architecture at a high level

The user interacts only with Streamlit. Pandas and NumPy prepare tables. scikit-learn optionally clusters narratives. Plotly draws charts. Rule-based text in `explanations.py` interprets PRR rows and CTD gaps. IBM Bob was used while designing and writing this code; it is not called when a reviewer clicks Run.

```
Reviewer → Streamlit (src/app.py)
            ├─ Signal path: data_processor → signal_detection → prr_analysis → CSV
            └─ CTD path:    submission_checker → gap report CSV
```

## Two application modes

1. Upload or load sample adverse-event data → validate → clean → cluster or group → PRR → rank → explain → download.
2. Upload, paste, or load a sample outline → parse sections → map to five modules → score → prioritize gaps → download.

## Core mechanisms

- **PRR / ROR / chi-square** use the same 2x2 table, with explicit zero handling.
- **Clustering** uses TF-IDF + KMeans, LDA topics, and cosine nearest neighbors when narratives exist; otherwise frequency grouping.
- **Isolation Forest** flags unusual drug-event pairs inside the current file and feeds a transparent reviewer priority score.
- **CTD scoring** is a weighted completeness percentage across a documented representative checklist (not an exhaustive filing list).
- **Explanations** are deterministic templates filled with those metrics. They are not calls to a hosted model.

## Differentiation

Many dashboards either plot raw counts or claim automated medical decisions. This prototype stays in the middle: reproducible screening math, visible 2x2 counts, and honest status labels (`Potential Safety Signal`, `Below Threshold`, `Insufficient Data`, `Undefined`).

## Design decisions

| Decision | Rationale |
|---|---|
| Streamlit only, no FastAPI/React | Matches the official Python prototype stack and stays inside `src/`. |
| No SQLite | The demo is file-in, table-out; persistence would not improve the judge flow. |
| No watsonx.ai at runtime | The core app must work without an external AI API. |
| Representative CTD list in code | Judges can read exactly what is scored. |
| Sample / synthetic data labeled as such | Avoids implying we redistributed FAERS. |
| IBM Bob for development | Used for planning, generation, refactoring, tests, and docs — not as a production model host. |

## User experience

The home page states both product names. The sidebar switches modes. Metrics, tables, expanders, progress bars, and Plotly charts keep the demo readable. Errors (bad CSV, missing columns, empty dossier text) surface as Streamlit messages instead of stack traces.
