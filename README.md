# Drug Safety Signal Detector & Regulatory Submission Readiness Checker

## Team

| Field | Value |
|---|---|
| **Team Name** | TechTress |
| **Track** | AI |
| **Team Lead** | Priya Pedhadiya — priyyapedhadiya1497@gmail.com |
| **Members** | Jiya Kothari, Dhruvi Garala, Krisha Thakor |

## Problem Statement

FDA FAERS-scale adverse-event collections contain millions of reports, so reviewers cannot reliably spot emerging drug-event patterns by hand. At the same time, a marketing-authorization CTD dossier spans five ICH M4 modules, and missing sections delay submission. Safety and regulatory teams need a transparent screening workspace that ranks potential signals and shows dossier gaps without pretending to be a certified pharmacovigilance or compliance system.

## Solution

The prototype is a two-mode **Streamlit** application. The UI (`src/app.py`) calls backend modules through `src/pipeline.py`. There is no FastAPI server and no external AI API at runtime.

1. **Signal Detection** loads a CSV (or bundled sample / synthetic data), cleans and validates it, maps simple name variants, groups or clusters patterns, calculates PRR (plus ROR and chi-square), flags **Potential Safety Signals**, and exports results.
2. **Submission Readiness** maps a dossier outline to a documented representative ICH M4 CTD checklist, scores each module, computes an overall readiness score, and downloads a prioritized gap report.

Rule-based explanations describe why a pair was flagged or why a gap is high priority. They are filled from calculated numbers, not from a hosted model.

## Key Features

- Adverse-event clustering (TF-IDF + KMeans), LDA topics, and similar-narrative search when text exists
- PRR, ROR, chi-square, and a **Method Agreement** column (both high vs only one method)
- Isolation Forest anomaly flags and a reviewer priority score
- Case-level drill-down, report-year trends, synonym normalization, and narrative severity keyword flags
- ICH M4 CTD completeness checking across five modules
- Module-wise readiness scoring and an overall readiness score
- Downloadable gap report with priority and recommendation columns
- Rule-based explanations from calculated metrics (offline)

## Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python |
| **Frameworks / libraries** | Streamlit, Pandas, NumPy, scikit-learn, Plotly |
| **IBM technologies (development)** | IBM Bob used for architecture planning, code generation, explanation, refactoring, tests, and documentation |
| **Databases** | None |
| **Other** | Git, GitHub, GitHub Actions, pytest |

IBM Bob is **not** a runtime dependency. FastAPI, React, Docker, PostgreSQL, and watsonx.ai are **not** used in the running app.

## How to Run

```bash
git clone https://github.com/priyya16/TechTress.git
cd TechTress

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

python -m pip install -r src/requirements.txt
python -m streamlit run src/app.py
```

Requires **Python 3.10+**. Open http://localhost:8501

No `.env` file is required.

```bash
python -m pytest src/test_core.py -v
```

More detail: [docs/setup-guide.md](docs/setup-guide.md) · [docs/architecture.md](docs/architecture.md) · [docs/problem-statement.md](docs/problem-statement.md) · [docs/solution-overview.md](docs/solution-overview.md)

## Demo

1. Start Streamlit and open **Signal Detection**.
2. Keep **Sample / Synthetic Data** selected (the default).
3. Review cleaning, statistics, clustering, PRR / ROR / agreement, case drill-down, and year charts.
4. Download `prr_signal_results.csv`.
5. Switch to **Submission Readiness**.
6. Keep the sample dossier outline selected.
7. Review module scores, overall readiness, and high-priority gaps.
8. Download `ctd_gap_report.csv`.

| Artifact | Location |
|---|---|
| Demo video | [demo/demo-video-link.txt](demo/demo-video-link.txt) |
| Live demo | [demo/live-demo-url.txt](demo/live-demo-url.txt) |
| Screenshots | [demo/screenshots/](demo/screenshots/) |
| Presentation | [presentation/](presentation/) |

## Known Limitations

- Prototype only — not production pharmacovigilance software.
- Sample / representative / synthetic data may be used. It is **not** the complete FDA FAERS dataset.
- Not a replacement for pharmacovigilance experts.
- Not a certified medical device or regulatory compliance system.
- The CTD checklist is representative rather than exhaustive.
- PRR is a statistical screening ratio, not a causal medical conclusion.
- Record a public demo video and put the URL on line 1 of `demo/demo-video-link.txt` before final submission if it is not there yet.

## What We're Most Proud Of

We combined statistically careful safety-signal screening (including invalid 2×2 handling and PRR/ROR agreement) with practical CTD readiness checking in one reviewer-facing dashboard. Honest limitation language matters more here than claiming a complete regulatory engine.
