# Setup Guide

This file is the exact runbook for judges and teammates.

## Prerequisites

- Python 3.10 or later (3.11+ recommended)
- `pip`
- Git
- A modern browser (Chrome, Edge, or Firefox)
- Optional: IBM Bob during development only (not required to run the app)

You do **not** need Node.js, Docker, PostgreSQL, IBM Cloud, or watsonx.ai credentials.

## Python version

```bash
python --version
```

Expect `Python 3.10` or newer.

## Clean-machine setup

```bash
git clone https://github.com/priyya16/TechTress.git
cd TechTress

python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

## Environment variables

None required. Do not create a `.env` file unless you want unused local overrides.

If you copy the example:

```bash
copy src\.env.example src\.env
```

or

```bash
cp src/.env.example src/.env
```

Never commit `.env`. The example file contains no secrets.

## How to run Streamlit

From the repository root:

```bash
python -m streamlit run src/app.py
```

Equivalent from `src/`:

```bash
cd src
python -m streamlit run app.py
```

The app is served at **http://localhost:8501**

## Verification steps

1. The home page shows both product titles and the TechTress disclaimer.
2. **Signal Detection** → Load sample / synthetic data → metrics populate → PRR table appears → charts render → CSV download works.
3. **Submission Readiness** → Load sample / synthetic outline → overall score and module table appear → gap report downloads.
4. Tests:

```bash
python -m pytest src/test_core.py -v
```

Expect all tests to pass.

## Expected output

- Signal mode labels the dataset **Sample / Synthetic Data** when the bundled CSV is used.
- At default thresholds (`a >= 3`, `PRR >= 2`), synthetic HeptraClear–Liver Injury, Cardionil–QT Prolongation, and Amplitide–Anaphylaxis pairs should appear as **Potential Safety Signals**.
- Submission mode shows a partial overall readiness score because the sample outline intentionally omits sections such as Stability and Clinical Safety Study Reports.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Activate the virtualenv and re-run `python -m pip install -r src/requirements.txt`. |
| `streamlit` is not recognized | Use `python -m streamlit run src/app.py`. |
| Port 8501 in use | `python -m streamlit run src/app.py --server.port 8502` |
| CSV rejected | Include `drug_name` and `adverse_event` headers. Save as UTF-8. |
| Empty PRR table | The file had no valid rows after cleaning. Check missing values. |
| PowerShell execution policy blocks venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then activate again. |

## IBM Bob (development only)

Bob is not started as part of this setup. Team members may use Bob while editing code or docs. Judges only need Python, pip, and Streamlit.
