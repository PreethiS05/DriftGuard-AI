# DriftGuard AI

A Streamlit app that compares a **reference** dataset with a **current** dataset and reports which features changed in a way that is both statistically significant and practically meaningful. Data-quality and schema problems are reported separately and never change a drift status.

> Data drift describes a change in input distributions. It does not by itself mean model accuracy has dropped; DriftGuard does not evaluate a model.

## Requirements

- Python 3.10+ (developed and tested on Python 3.13)
- Packages in `requirements.txt`: Streamlit, pandas, NumPy, SciPy, Plotly

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # app only
pip install -r requirements-dev.txt  # app + pytest
```

## Launch

```bash
streamlit run app.py
https://mainvenvpscusershpdownloadsdrift-guard-ai-development-k4vxbsga.streamlit.app/
```

Open http://localhost:8501. To pick a port or bind to all interfaces:

```bash
streamlit run app.py --server.port 3000 --server.address 0.0.0.0
```

`package.json` exists only so the hosted preview can start the app (`npm run dev`) and run the tests (`npm test`). It assumes a virtualenv at `.venv/`. You don't need Node to use DriftGuard.

## Run the tests

```bash
pytest            # configured in pytest.ini (testpaths = tests, -q)
```

## Using the app

Choose a data source in the sidebar, then click **Run analysis**:

| Source | What it does |
| --- | --- |
| **Synthetic demo** | Generates a reference/current pair for a chosen scenario, row count and random seed. The Overview also has one-click buttons for common scenarios. |
| **Bundled samples** | Loads CSVs from `data/samples/` (stable, drifted, schema change, tiny). |
| **Upload CSVs** | UTF-8 CSV with a header row, up to 50 MB each. |

Synthetic scenarios: `stable`, `mean_shift`, `variance_shift`, `category_shift`, `new_category`, `missing_increase`, `schema_change`, `combined`. Synthetic data is labelled as synthetic throughout the UI and in exports.

### Pages

- **Overview**: headline status, key metrics, PSI by feature, drifted features with plain-language explanations, and the top quality issues.
- **Dataset Comparison**: shape, schema differences and per-column profiles for both datasets.
- **Drift Analysis**: a filterable feature table and a per-feature detail view. It shows test statistics, adjusted p-values, effect sizes and an explanation, plus histogram/ECDF charts for numerical features and proportion charts for categorical features.
- **Data Quality**: missing-value changes, constant columns, unseen categories, duplicates, type mismatches, and missing or new columns, each with a severity.
- **Reports**: download feature results (CSV), the full summary (JSON) and an HTML report. This page also shows the thresholds and methodology used. Exports contain aggregate statistics only, never raw rows.
- **Analysis History**: past analysis summaries stored in local SQLite. You can compare two runs, download one as JSON, or delete them.

## Methodology (short version)

- **Numerical features**: two-sample Kolmogorov-Smirnov test, plus PSI and summary-statistic deltas.
- **Categorical features**: chi-square test of homogeneity with Cramér's V, plus PSI and unseen-category coverage.
- **Multiple testing**: p-values are adjusted across features (Benjamini-Hochberg by default; Bonferroni or none are selectable).
- **Decision rule**: a feature is **Drift Detected** only when the adjusted p-value is below alpha *and* the effect size exceeds its threshold. Features without enough non-missing rows are **Insufficient Data** rather than silently passing. Identifier-like columns are excluded.

The thresholds are in `driftguard/config.py` and can be adjusted under **Analysis settings** in the sidebar.

## Project layout

```
app.py                 Streamlit entry point and page navigation
driftguard/            Analysis engine (no Streamlit imports)
  config.py            Thresholds and settings
  data_loader.py       CSV loading and validation
  schema.py            Schema inference and comparison
  profiling.py         Per-column profiles
  drift.py             Statistical drift tests and status decisions
  quality.py           Data-quality checks
  synthetic.py         Synthetic scenario generator
  pipeline.py          End-to-end analysis orchestration
  reports.py           CSV / JSON / HTML exports
  history.py           SQLite history store
  methodology.py       Methodology text shown in the UI
  visualization.py     Plotly chart builders
ui/                    Session state, sidebar and shared components
views/                 One module per page
tests/                 pytest suite
scripts/generate_samples.py  Regenerates data/samples/
data/samples/          Bundled synthetic CSVs
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DRIFTGUARD_HISTORY_DB` | `data/history.db` | Path of the SQLite history database |

To regenerate the bundled samples, run `python scripts/generate_samples.py`.

## Limitations

- Drift is tested one feature at a time; DriftGuard does not test joint or multivariate drift.
- With very large samples, tiny differences become statistically significant. The effect-size threshold is there to filter those out.
- History is stored in a local SQLite file, so it isn't shared between machines or users.
