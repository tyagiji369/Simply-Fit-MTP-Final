# Original vs Improved Pipeline

A reproducible, like-for-like comparison of the internship estimator and the corrected one. **Simulated data only.** Full detail: `EXPERIMENT_RESULTS.md`.

## How the comparison is kept fair

- The original code is frozen in `src/legacy/` and is byte-identical to the delivered ZIP (`tests/test_legacy_frozen.py` checks the SHA-256).
- Both pipelines receive exactly the same readings for every simulated user and window (`tests/test_experiments.py::test_all_methods_receive_identical_readings_and_truth`).
- The original is given a bare list of readings, which is all the original app stored.
- Users are generated from evaluation seeds that played no part in designing the new method. The original has no tuned parameters here, so it cannot be disadvantaged by the split.
- The clean condition (no missed days, no gross errors), which favours the original, is reported next to the others.

## Newly generated results (this project, evaluation seeds)

Mean absolute error of the energy-balance equivalent, kcal/day:

{{T:headline_before_after}}

Paired differences with 95% bootstrap intervals (negative = corrected method closer to the truth):

{{T:eval_paired}}

Other before/after measures:

| Measure | Original | Corrected |
|---|---|---|
| "95%" interval containing the truth (base condition, all windows) | {{S:coverage_pooled_base::original|.0%}} | {{S:coverage_pooled_base::huber_ar1|.0%}} |
| Deficit/surplus outputs pointing the wrong way, 28 days of data | {{S:legacy_wrong_sign::last 28 d|.0%}} (always outputs a signed number) | {{S:decisive::28::definite labels pointing the WRONG WAY|.1%}} (labels {{S:decisive::28::definite label given|.0%}} of users, says "not yet clear" for the rest) |
| Error in the 4 weeks after a change of behaviour, kcal/day | {{S:tracking::legacy_app_full_history::0-27 days after change|.0f}} | {{S:tracking::huber_trailing_28::0-27 days after change|.0f}} |
| Readings flagged on error-free data | {{A:A4_isolation_forest_on_error_free_data::mean_fraction_flagged_on_error_free_data|.1%}} flagged and overwritten | under 4% marked unusual, none overwritten |
| Noise-free -550 kcal/day line, 14 days | {{A:A1_ewma_attenuation_noise_free_line::0::v1_research_kcal|.0f}} (research module) | -550 |
| Same line with 40% of days missing | {{A:A3_missing_days_time_compression::2::v1_app_mean_kcal|.0f}} (app) | -550 |

![MAE by window](results/experiments/figures/fig1_mae_by_window.png)

![Stress sweeps](results/experiments/figures/fig2_stress_sweeps.png)

![Interval coverage](results/experiments/figures/fig3_interval_coverage.png)

![Tracking after a change](results/experiments/figures/fig4_tracking_after_change.png)

![Worked example](results/experiments/figures/fig7_worked_example.png)

The worked example shows what each pipeline fits for one simulated user. A single case says nothing about average accuracy; in this one the original happens to land closer to the truth.

## Where each pipeline succeeds and fails

| Situation | Original | Corrected |
|---|---|---|
| Complete daily logs, no gross errors, 4-8 weeks | Accurate; as good as the corrected method ({{S:headline::clean, last 28 d::legacy_app|.0f}} vs {{S:headline::clean, last 28 d::huber|.0f}} at 28 days) | Accurate |
| Two weeks of data | Research module has the lower average error ({{S:headline::base, last 14 d::legacy_research|.0f}} vs {{S:headline::base, last 14 d::huber|.0f}}) because it shrinks estimates; it still prints a confident number | No label, reliability "low"; honest that the data are insufficient |
| Missed days | Fails: error grows with the share of missed days ({{S:headline::missing 45%, last 28 d::legacy_app|.0f}} at 45%) | Unaffected in bias; error {{S:headline::missing 45%, last 28 d::huber|.0f}} |
| Gross errors | Degrades ({{S:headline::gross errors 10%, last 28 d::legacy_app|.0f}} at 10%) | Nearly flat ({{S:headline::gross errors 10%, last 28 d::huber|.0f}}) |
| Stating uncertainty | Interval right {{S:coverage_pooled_base::original|.0%}} of the time | {{S:coverage_pooled_base::huber_ar1|.0%}}; still short of 95% |
| After a change of behaviour | Whole-history average, slow to move | Trailing window follows the change; worse than a longer window when nothing changes |
| True conversion factor differs from 7,700 | Wrong | Equally wrong; stated as an assumption |
| Start or end of a diet (water shifts) | Biased | Equally biased; stated as a limitation |

## Historical saved results (not regenerated as new findings)

`results/model_evaluation.json`, written by the internship version on 2026-08-25, concerns the secondary forecasting feature on the internship synthetic cohort:

| Method | Saved MAE (kg) | Reproduced here from the saved model |
|---|---|---|
| LSTM | 0.537 | {{F:reproduction_same_cohort_and_split::mae_kg::saved_lstm_with_training_scaler|.3f}} |
| Persistence | 0.468 | {{F:reproduction_same_cohort_and_split::mae_kg::persistence|.3f}} |
| Linear extrapolation | 0.401 | {{F:reproduction_same_cohort_and_split::mae_kg::linear_14day_ols|.3f}} |

The internship README's energy-balance accuracy table ("median 8 kcal/day") is also historical, and its reference was circular (`CORRECTIONS_REPORT.md`, issue 1). New forecasting results are in `EXPERIMENT_RESULTS.md` section 11 and `results/forecast_reevaluation.json`.

## Reproduce

```bash
python -m experiments.audit_original
python -m experiments.run_experiments --stage all
python -m experiments.forecast_experiment
python -m experiments.forecast_reevaluation      # needs TensorFlow
python -m experiments.summarize
python -m experiments.build_report
```
