# Notebooks (internship-era, historical)

These five notebooks document the **original internship pipeline** and are kept unchanged as a record.
They were **not re-executed** for the M.Tech work, and some of their conclusions are superseded:

- `02_ml_layer.ipynb` reports a median error of about 8 kcal/day "vs realized balance". That
  comparison used a straight-line fit of the same noisy readings as the reference, not the
  simulator's true balance. See `CORRECTIONS_REPORT.md`, issue 1.
- `03_lstm_model.ipynb` is the source of `results/model_evaluation.json` (historical).

The current, reproducible analysis is in `experiments/` and `EXPERIMENT_RESULTS.md`.
`scripts/build_notebooks.py` regenerates these notebooks against the frozen v1 code in `src/legacy/`.
