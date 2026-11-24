# Simply-Fit | Final M.Tech project

Estimate a weight-trend energy equivalent from dated measurements and withhold the number when the log falls outside the simulation-tested reporting policy. The project uses synthetic longitudinal ground truth. The app cannot measure a person's intake, expenditure or individual tissue energy density.

**Read first:** [Research report](research/REPORT.pdf) · [Methodology and results in Markdown](research/REPORT.md) · [Guide presentation outline](research/GUIDE_PRESENTATION.md) · [Pre-change audit](research/AUDIT.md)

## Run the application

From this directory, on Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

The core app and local coach work without an API key. The original Gemini configuration and optional LSTM remain available. The app defaults to a 28-day window. The empirical reporting policy was evaluated for observed spans up to 84 calendar days. A longer full-history request is withheld; select a supported trailing window. The research/reporting interval is distinct from the future-weight prediction band.

## Using the Gemini-first coach

Open **API Settings** in the Streamlit sidebar and enter a Google Gemini API key, or set GEMINI_API_KEY in the project .env file. A server key stays on the server. The question, selected conditions and summarized metrics are sent to Google only when the online coach is enabled. No raw daily log or key is included in the model prompt. GEMINI_MODEL can override the default gemini-3.6-flash when your account uses another model supporting JSON schema responses.

The primary coach accepts free-form questions, including questions the offline classifier has never seen. It reads a small bundled knowledge base with nine attributed summaries from NHS, CDC, NIDDK and NHLBI. All selected conditions reach both the answer and review requests. Health answers need recognized references. The app creates links from its own catalog and suppresses drafts that fail the second Gemini check. Unsupported medical topics receive an explanation of the evidence limit. Medication dose requests and urgent symptoms have additional safeguards. These checks reduce hallucination risk. They cannot establish clinical safety; the second model call can share the first call’s errors.

Ordinary successful online answers use two API calls. Missing credentials and API interruptions display different notices in the tracker and report. Offline answers use the local general recommendations. Source summaries are a curated snapshot, not a live web search. The original research results predate this coaching update.

## Reproduce the research

The shipped research/policy.json is sufficient to launch the app. To regenerate results, run from this directory. These commands overwrite only the new experiment outputs and frozen policy files. Use a fresh extracted copy to compare with the archived run.

```bash
python -m research.experiment select
python -m research.experiment calibrate
python -m research.experiment final
python -m research.experiment sensitivity
python -m research.experiment shift
python -m research.summarize
python -m research.diagnostics
```

The Python scripts contain no network calls. research/design.json contains the seed families and split design. research/selection.json records the development choice. research/policy.json records validation quantiles. The row for a user who failed minimum history remains in the CSV with eligible=False. Recreate any log using research.world.simulate(split, person). It returns calendar days, observed weights, realized balance, fixed-conversion target and parameters. Sweep rows require the matching reference settings and one changed value from research/experiment.py.

To run all automated tests including the optional LSTM:

```bash
python -m pip install -r requirements-full.txt
python -m pytest -q
```

research/environment.txt lists versions used for the final verification. For a lighter test installation use the app requirements plus pytest and statsmodels; the optional LSTM test skips without TensorFlow. The older requirements-full.txt also installs notebook tools for historical work.

## Evidence and provenance

| Location | Contents |
|---|---|
| research/results | Raw and summarized new experiment files, plots, and verification log |
| src/research_estimate.py and src/state_space.py | Shared deployed rule, score and state-space comparison |
| research/baselines | Exact input ZIPs for both versions; extract these for exact historical code and dependencies |
| src/legacy and results/experiments | Original BTP estimators and submitted MTP's historical A/B outputs |
| docs/SUBMITTED_MTP_README.md, EXPERIMENT_RESULTS.md and other inherited top-level reports | Documentation of the submitted MTP, predating the independent study |

The comparison with real NHANES adult weight and age distributions is inherited cross-sectional work. All new estimation and coverage experiments use synthetic weight histories.
