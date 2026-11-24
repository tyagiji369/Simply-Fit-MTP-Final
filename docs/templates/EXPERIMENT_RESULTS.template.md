# Experiment Results

**All results in this document come from computer-simulated weight logs.** They show how the methods behave under stated simulation assumptions. They are not real-world or clinical validation, and the kcal/day figures are approximate weight-trend-derived energy-balance equivalents, not measured physiological energy balance.

Every table and number below is generated from `results/experiments/` by `python -m experiments.build_report`; none is typed by hand. Reproduction commands are in section 15.

## 1. Summary of findings

1. **With missed days or gross errors, the corrected estimator is clearly more accurate than the original.** With 45% of days missing, mean absolute error over 28 days falls from {{S:headline::missing 45%, last 28 d::legacy_app|.0f}} (original app) to {{S:headline::missing 45%, last 28 d::huber|.0f}} kcal/day. With 10% gross errors it falls from {{S:headline::gross errors 10%, last 28 d::legacy_app|.0f}} to {{S:headline::gross errors 10%, last 28 d::huber|.0f}}. In the base condition (10% missed days, 2% gross errors) it falls from {{S:headline::base, last 28 d::legacy_app|.0f}} to {{S:headline::base, last 28 d::huber|.0f}} at 28 days and from {{S:headline::base, last 56 d::legacy_app|.0f}} to {{S:headline::base, last 56 d::huber|.0f}} at 56 days.
2. **On complete, error-free logs the corrected estimator is not more accurate than the original.** Clean condition, 28 days: original app {{S:headline::clean, last 28 d::legacy_app|.0f}}, corrected {{S:headline::clean, last 28 d::huber|.0f}} kcal/day. The paired difference is {{S:paired::clean, last 28 d vs legacy_app::diff|+.1f}} kcal/day (95% CI {{S:paired::clean, last 28 d vs legacy_app::lo|+.1f}} to {{S:paired::clean, last 28 d vs legacy_app::hi|+.1f}}), i.e. the original is marginally better there.
3. **With only 14 days of data the original research module has lower average error** ({{S:headline::base, last 14 d::legacy_research|.0f}} vs {{S:headline::base, last 14 d::huber|.0f}} kcal/day). Section 6 shows this is because its smoothing shrinks every estimate towards zero by about a fifth at that length, which also makes it under-report sustained deficits. No method is useful at 14 days: all errors are around 300 kcal/day.
4. **The original "95% confidence interval" contained the truth {{S:coverage_pooled_base::original|.0%}} of the time.** The corrected interval contained it {{S:coverage_pooled_base::huber_ar1|.0%}} of the time. That is a large improvement and still short of the nominal 95%; the software says so.
5. **The original app always printed a signed kcal number.** With 28 days of data, {{S:legacy_wrong_sign::last 28 d|.0%}} of its deficit/surplus outputs pointed the wrong way. The corrected app names a deficit or surplus only when the interval supports it and at least 25 days are covered; with 28 days it does so for {{S:decisive::28::definite label given|.0%}} of users and is wrong in direction for {{S:decisive::28::definite labels pointing the WRONG WAY|.1%}} of those.
6. **After a change in behaviour, the original headline number (fitted to the whole history) is far off**: {{S:tracking::legacy_app_full_history::0-27 days after change|.0f}} kcal/day error in the four weeks after a change, against {{S:tracking::huber_trailing_28::0-27 days after change|.0f}} for the corrected trailing 28-day estimate. The original 5-entry bar chart had an error of {{S:tracking::legacy_5day_bars::all|.0f}} kcal/day throughout.
7. **The 7,700 kcal/kg conversion is itself a source of error that weight data cannot remove.** In the second simulation model the true factor differs by person; even a perfectly known tissue trend, converted at 7,700, differs from the true balance by a median of {{S:mechB_conversion::median_excess|+.0%}} (5th to 95th percentile {{S:mechB_conversion::p5_excess|+.0%}} to {{S:mechB_conversion::p95_excess|+.0%}}) (section 8).

## 2. Evaluation design

| Item | Choice |
|---|---|
| Data | Simulated 90-day daily weight logs with known daily energy balance (`SYNTHETIC_DATA_METHODOLOGY.md`) |
| Simulation models | A ("ledger", shares the 7700 assumption) and B ("two-compartment, adaptive", does not) |
| Scenarios | stable, sustained deficit, sustained surplus, deficit to maintenance, deficit to surplus, gradual drift |
| Observation conditions | base (10% missed days, 2% gross errors), clean (none), and sweeps of noise, missed days and gross errors |
| Development data | seed base 100000. Used for every design decision. The original pipeline is never run on it. |
| Evaluation data | seed base 900000, disjoint simulated users. Used once, for the numbers reported here. |
| Estimation windows | last 14, 28 and 56 days of the series, and the first 28 days |
| Target (ground truth) | mean true energy balance over the same calendar window, kcal/day |
| Evaluation size | {{S:headline::base, last 28 d::n}} simulated users in the base condition (150 per model-scenario cell), {{S:headline::clean, last 28 d::n}} in the clean condition, 480 per sweep condition; {{S:n_series::eval_rows}} window-level rows in total |
| Fairness | Every method receives exactly the same readings for each window. The original code runs unchanged from `src/legacy/` (byte-identical to the delivered ZIP, checked by a test) and is given a bare list of readings, as the original app was. A window with fewer than 7 readings is skipped for all methods alike. |
| Uncertainty of comparisons | Each row is an independent simulated user. Differences in absolute error are paired by user, with a bootstrap 95% CI (2000 resamples) and a Wilcoxon signed-rank test. |

Real-time ("as of day t") estimates use only readings up to day t. No method in the corrected pipeline smooths with future data.

## 3. Methods compared

| Method | What it does | Assumptions |
|---|---|---|
| Original: research module | EWMA (span 7), Isolation Forest (contamination 0.1) on residuals, flagged readings replaced by EWMA values, EWMA again, least squares against list position. `src/legacy/ml_engine_v1.py` | Readings are consecutive days; 10% of readings are anomalies; smoothing does not change the slope |
| Original: app | Same anomaly step, then least squares on the anomaly-replaced raw series against list position. `src/legacy/app_estimator_v1.py` | As above |
| Original steps, real day index | Research module with the regression run against actual days | Diagnostic: separates the missed-days defect from the smoothing/anomaly steps |
| EWMA then regression | EWMA then least squares on actual days, no anomaly step | Smoothing does not change the slope |
| Least squares on raw readings | Straight line through raw readings on actual days | Constant balance in the window; independent, outlier-free errors |
| Theil-Sen | Median of pairwise slopes | Constant balance; robust to roughly 29% contamination |
| **Huber regression (recommended)** | Straight line fitted by iteratively reweighted least squares; readings far from the line get reduced weight, none is deleted or replaced. `src/energy_balance.py` | Constant balance in the window; most readings are ordinary |

All methods convert kg/day to kcal/day by multiplying by 7700. Input unit kg, time unit days, output kcal/day; negative means deficit.

Interval methods compared: the original interval; classical least squares; Newey-West HAC; an AR(1) sandwich on actual day gaps with and without small-sample correction; moving-block bootstrap.

## 4. Decisions made on development data

**Point estimator.** Development mean absolute error (kcal/day):

{{T:dev_point_mae}}

{{T:dev_sweep_mae}}

Huber regression was chosen: it matches plain least squares on clean data and is the most accurate under gross errors. EWMA-then-regression had lower error at 14 days on development data too; it was not chosen because that advantage is shrinkage (section 6).

**Interval method.** Share of development intervals containing the tissue-trend equivalent (nominal 0.95):

{{T:dev_interval_coverage}}

The AR(1) sandwich with small-sample correction was chosen as the best-calibrated option (pooled {{S:dev_interval_pooled::huber_ar1|.0%}} against {{S:dev_interval_pooled::ols_classical|.0%}} for the classical formula). None reached 95%.

**Default window for the current estimate.** Development error against the true balance of the last 14 days:

{{T:dev_tracking_window}}

28 days was chosen because it has the smallest worst-phase error. A 42-day window is better when behaviour is steady and worse just after a change.

**Minimum history for naming a deficit or surplus.** Development results, base condition, before any minimum-history rule:

{{T:dev_labels_ungated}}

Direction calls on 14 or 21 days of data were wrong 9-15% of the time. Grouped by actual days covered, the wrong-direction rate first dropped to 5% or less at 25-27 days, so the application gives no label below 25 days.

**Forecast method (secondary).** See section 11.

## 5. Results on evaluation data

### 5.1 Base condition: mean absolute error, kcal/day

{{T:eval_base_mae}}

Bias (mean of estimate minus truth):

{{T:eval_base_bias}}

RMSE:

{{T:eval_base_rmse}}

![MAE by window](results/experiments/figures/fig1_mae_by_window.png)

### 5.2 Clean condition (no missed days, no gross errors)

{{T:eval_clean_mae}}

In this condition "Original steps, real day index" equals the original research module because list position and day index coincide. Differences between methods are within about 10 kcal/day except at 14 days.

### 5.3 By scenario (base condition, last 28 days)

{{T:eval_by_scenario}}

### 5.4 Stress sweeps (last 28 days; the same 480 simulated users in every row of a table)

Noise level:

{{T:eval_sweep_noise}}

Missed days:

{{T:eval_sweep_missing}}

Gross errors:

{{T:eval_sweep_outliers}}

![Stress sweeps](results/experiments/figures/fig2_stress_sweeps.png)

The original methods degrade steeply with missed days because skipped days are treated as consecutive, which inflates the slope. Once the real day index is used ("Original steps, real day index") that degradation disappears. The original anomaly step does not protect against gross errors: its error grows from 0% to 10% contamination, where Huber's is almost flat.

![Missing days scatter](results/experiments/figures/fig6_missing_days_scatter.png)

### 5.5 Paired comparisons with uncertainty

Negative values mean the Huber estimate is closer to the truth.

{{T:eval_paired}}

Reading this table: improvements over both original variants are statistically clear in the base, missed-days, gross-error and high-noise conditions. In the clean condition they are absent or slightly reversed. Against plain least squares on raw readings, Huber's advantage is small unless gross errors are common.

### 5.6 Deficit / surplus / maintenance classification

Truth class uses a tolerance of ±100 kcal/day. Accuracy when every method is forced to give one of three labels from its point estimate:

{{T:eval_classification}}

Three-way accuracy is modest for every method at 14 and 28 days, because the tolerance band (±100) is narrower than the estimation error. This is why the application does not force a label. What the application actually shows:

{{T:eval_decisive_labels}}

For comparison, the original app's printed number, read as a label:

{{T:eval_legacy_wrong_sign}}

## 6. Why the original smoothing looks better at 14 days

EWMA with a fixed start value, followed by regression, multiplies the slope of a noise-free line by a factor k below one (about 0.81 at 14 days, 0.93 at 28, 0.98 at 56). Shrinking every estimate towards zero lowers average error when true values are scattered around zero, as they are in these scenarios. Two controls test whether that is the whole explanation: multiply the Huber estimate by the same k, and divide the EWMA estimate by k.

{{T:eval_shrinkage}}

"Size ratio" is the regression of the estimate on the truth over the sustained-deficit and sustained-surplus scenarios; 1.00 means sizes are reported correctly on average.

At 14 days the explicitly shrunk Huber estimate has lower error than the EWMA method, and the un-shrunk EWMA estimate has higher error than Huber. So the smoothing adds no information; it applies an undeclared, window-length-dependent shrinkage. The cost is that sustained deficits and surpluses are under-reported: size ratio {{S:shrinkage::base, last 14 d::size ratio Original (research)|.2f}} for the original research module at 14 days against {{S:shrinkage::base, last 14 d::size ratio Huber|.2f}} for Huber. The recommended estimator reports the unshrunk slope and states its uncertainty instead. A deliberate, documented shrinkage could be added later, but its benefit depends on how real users' balances are distributed, which this project cannot know.

## 7. Confidence intervals

Share of evaluation intervals (base condition) containing the tissue-trend equivalent; nominal 0.95:

{{T:eval_interval_coverage}}

Containing the true energy balance:

{{T:eval_interval_coverage_true}}

Mean interval width, kcal/day:

{{T:eval_interval_width}}

![Interval coverage](results/experiments/figures/fig3_interval_coverage.png)

Interpretation:

- The original interval is computed from residuals of an already-smoothed series. It is about {{S:coverage_pooled_base::original_mean_width}} kcal/day wide on average and contains the truth {{S:coverage_pooled_base::original|.0%}} of the time. It should not have been called a 95% interval.
- Classical least-squares intervals ignore day-to-day persistence of water weight and cover about {{S:coverage_pooled_base::ols_classical|.0%}}.
- Newey-West HAC and the block bootstrap, the usual autocorrelation-aware choices, are no better here ({{S:coverage_pooled_base::ols_hac|.0%}} and {{S:coverage_pooled_base::huber_block_bootstrap|.0%}}). Both are known to be unreliable with only 14-56 observations.
- The AR(1) sandwich covers {{S:coverage_pooled_base::huber_ar1|.0%}} overall. It still under-covers because (a) the autocorrelation is estimated from very few points, (b) real noise is not exactly AR(1), and (c) the balance is not constant inside the window, which the straight-line model assumes.

To reach 95% coverage the recommended interval would have needed to be wider by these factors:

{{T:eval_interval_inflation}}

No such factor is applied in the software. It would be calibrated to this simulator, not to real people. The application instead labels the interval "approximate" and states this shortfall.

Three different uncertainties are kept apart in code and interface: the trend-slope interval (kg/day), the energy-equivalent interval (the same interval multiplied by 7700; it excludes uncertainty in 7700), and the prediction interval for a future scale reading (section 10).

## 8. The conversion factor (simulation model B)

In model B a kilogram of tissue change is not worth 7700 kcal. For sustained scenarios, clean condition, last 56 days (n = {{S:mechB_conversion::n}}), the ratio of the tissue-trend equivalent to the true balance has median {{S:mechB_conversion::median_ratio|.2f}} (5th-95th percentile {{S:mechB_conversion::p5|.2f}} to {{S:mechB_conversion::p95|.2f}}). Huber's error against the true balance is {{S:mechB_conversion::mae_vs_true|.0f}} kcal/day, of which {{S:mechB_conversion::mae_vs_equiv|.0f}} is error in the trend itself.

{{T:eval_two_truths}}

No analysis of weight alone can remove this part of the error. It is the reason the output is described as an energy-balance *equivalent*.

## 9. Estimates over time ("as of today")

Error against the true mean balance of the most recent 14 days, using only readings available on each day:

{{T:eval_tracking}}

![Tracking after a change](results/experiments/figures/fig4_tracking_after_change.png)

The original app's headline number was a fit to the entire history, so it describes the average since day one. Its 5-entry bar chart divided the difference of two single readings by five; its error is larger than any plausible true value. The corrected default (trailing 28 days) is not the best in every phase: a 42-day window has lower error when nothing changes. The window is user-selectable.

## 10. Reliability indicator

The application shows a level (high / moderate / low / insufficient) derived from interval width and data-quality checks. Realised error on evaluation data by displayed level (five conditions, windows of 14-56 days, n = {{S:n_series::app_rows_eval}} user-windows):

{{T:eval_reliability}}

![Reliability vs error](results/experiments/figures/fig5_reliability_vs_error.png)

The ordering is as intended: median error {{S:reliability::high::median|.0f}}, {{S:reliability::moderate::median|.0f}} and {{S:reliability::low::median|.0f}} kcal/day for high, moderate and low. Note the last column: intervals in the "high" group cover least often ({{S:reliability::high::coverage|.0%}}), because a narrow interval is partly a sign that persistence was underestimated. The level describes data quality; it is not a probability of being right, and the interface says so.

How often each level appears, base condition:

{{T:eval_reliability_by_window}}

By condition (all window lengths):

{{T:eval_reliability_by_condition}}

## 11. Forecasting (secondary feature)

Seven-day-ahead error on scale readings, scenario simulator, base condition. Development data (used to choose the default):

{{T:forecast_dev}}

Evaluation data:

{{T:forecast_eval}}

80% prediction intervals on evaluation data:

{{T:forecast_pi}}

{{T:forecast_pi_by_horizon}}

On the internship cohort (independent noise, `results/forecast_reevaluation.json`) the same methods score: corrected LSTM {{F:corrected_optional_lstm_window_centred::mae_kg::lstm|.3f}}, recent level {{F:corrected_optional_lstm_window_centred::mae_kg::level|.3f}}, linear trend {{F:corrected_optional_lstm_window_centred::mae_kg::linear|.3f}}, persistence {{F:corrected_optional_lstm_window_centred::mae_kg::persistence|.3f}} kg.

Findings:

- The saved internship figures (LSTM 0.537, persistence 0.468, linear 0.401 kg) are reproduced exactly from the saved model and regenerated cohort. They are historical values from `results/model_evaluation.json`.
- The saved LSTM was evaluated with one normalisation and deployed with another. Scored the way the app called it, the same model gives {{F:deployment_mismatch_of_saved_lstm::mae_kg::as_deployed, 14-day log (scaler from the 14 inputs)|.3f}} kg (14-day log) or {{F:deployment_mismatch_of_saved_lstm::mae_kg::as_deployed, full history (scaler from all days so far)|.3f}} kg (full history), not 0.537. The published number did not describe the deployed model.
- The original fallback used when TensorFlow was missing is the least accurate method in every comparison ({{S:forecast::mae_legacy_fallback|.3f}} kg here).
- With consistent normalisation the retrained LSTM is the best method on the internship cohort and ties the recent-level rule on the scenario simulator ({{S:forecast::mae_lstm_subset|.3f}} vs {{S:forecast::mae_level_subset|.3f}} kg on the subset where it can run).
- The linear trend, best of the original baselines on the internship cohort, is worse than repeating the last reading on the scenario simulator. Rankings depend on the noise model; all differences are small next to single-reading scatter.
- The application default is the recent-level rule: it needs no TensorFlow, tolerates missed days, and its prediction band covered {{S:forecast::pi80_level|.0%}} against a nominal 80%. Its known weakness: it ignores the trend, so in the sustained-deficit scenario its forecasts are on average {{S:forecast::bias_level_sustained_deficit|.2f}} kg too high.

## 12. Failure cases

| Situation | Evidence | What the software does |
|---|---|---|
| Two weeks of data | Error about 250-350 kcal/day for every method; direction calls wrong 13-15% of the time | No label below 25 days; reliability held at "low"; no number at all below 10 readings / 14 days |
| Balance changes inside the window | Trailing 28-day estimate: error {{S:tracking::huber_trailing_28::0-27 days after change|.0f}} kcal/day in the 4 weeks after a change against {{S:tracking::huber_trailing_28::no change in scenario|.0f}} when nothing changes | Window is selectable; whole-period and recent figures shown side by side; stated in limitations |
| Start of a diet (model B, first 28 days) | Negative bias (see the bias table in 5.1): early water loss reads as extra deficit | Stated in limitations |
| Conversion factor wrong for the person | Section 8 | Output named "equivalent"; assumption stated next to the result |
| 45% of days missed | Error at 28 days rises from {{S:sweep_missing::base::huber|.0f}} to {{S:sweep_missing::missing_45::huber|.0f}} kcal/day; about a quarter of user-windows get no estimate | Missed-days check lowers reliability |
| Complete, clean logs | No accuracy gain over the original | Reported here |
| Interval calibration | {{S:coverage_pooled_base::huber_ar1|.0%}} instead of 95% | Labelled approximate; shortfall stated in the app |

## 13. Limitations

1. Simulation only. No real weight log was analysed.
2. Both simulation models were written by the author of the estimator. Using two models reduces, and does not remove, the risk of circularity. Model B's constants were not re-verified against their literature sources.
3. The original pipeline's results depend on the fixed `random_state=42` of its Isolation Forest, as shipped.
4. Thresholds (±100 kcal/day tolerance, reliability cut-offs at 150 and 300 kcal/day, 25-day minimum) were set on development simulations and may not transfer to real users.
5. Average-error comparisons favour shrinkage because simulated balances are centred near zero (section 6).
6. Only one conversion factor (7700) was evaluated.
7. Paired tests treat simulated users as independent, which they are by construction; the same user appears in several conditions, so different rows of the paired table are not independent of each other.

## 14. Evidence for the recommended method

Huber regression on raw readings against real days, with an AR(1)-sandwich interval, a 28-day default window, and a label only when the interval and a 25-day minimum support it. Supporting evidence: it is never materially worse than plain least squares and is better under gross errors (5.4, 5.5); it removes the missed-days defect (5.4); its interval is the best calibrated of those tested (7); its reliability level orders realised error (10); and its definite labels are rarely wrong in direction (5.6). Its limits are those in sections 12 and 13.

## 15. Reproduction

```bash
pip install -r requirements-full.txt
python -m experiments.audit_original            # evidence for CORRECTIONS_REPORT.md (about 3 min)
python -m experiments.run_experiments --stage all   # about 45 min on 2 cores
python -m experiments.forecast_experiment       # about 1 min; uses the optional LSTM if TensorFlow is present
python -m experiments.forecast_reevaluation     # needs TensorFlow; retrains the optional LSTM (about 2 min)
python -m experiments.summarize                 # tables, figures, summary.json
python -m experiments.build_report              # regenerates this file and the other reports
# one-minute smoke version of the pipeline:
python -m experiments.run_experiments --stage all --quick && python -m experiments.forecast_experiment --quick && python -m experiments.summarize --quick
```

Raw per-series results: `results/experiments/raw/*.csv`. With the fixed seeds in `config/scenarios.json` the simulation and all estimators are deterministic.
