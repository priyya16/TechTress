# Source code (`src/`)

This folder is the **entire application**. Judges and teammates should start with the root [README.md](../README.md) for the full story, then run from the repo root:

```bash
python -m pip install -r src/requirements.txt
python -m streamlit run src/app.py
```

## What each file does

| File | Role |
|---|---|
| `app.py` | Streamlit frontend (pages, widgets, charts, downloads) |
| `pipeline.py` | Backend functions the UI calls |
| `data_processor.py` | CSV load, validation, cleaning |
| `quality.py` | Synonyms, severity keywords, year counts, case rows |
| `prr_analysis.py` | 2×2 PRR |
| `ml_screening.py` | ROR, chi-square, Isolation Forest, method agreement |
| `signal_detection.py` | TF-IDF / KMeans / LDA / similar reports |
| `submission_checker.py` | Representative ICH M4 completeness |
| `explanations.py` | Rule-based text (no API) |
| `utils.py` | Paths and shared labels |
| `test_core.py` | pytest |
| `sample_data/` | Synthetic CSV and dossier outline |

`.env.example` documents that **no API keys are required**. Do not commit `.env`.

```bash
python -m pytest src/test_core.py -v
```
