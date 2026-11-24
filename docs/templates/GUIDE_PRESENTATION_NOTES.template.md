# Guide Presentation Notes

Speaking notes for presenting the M.Tech extension of Simply-Fit. Every number below is from the project's result files and is filled in automatically; say "in simulation" whenever you quote one. Nothing here has been validated on real people.

## 1. The problem statement (unchanged)

Estimate a person's approximate daily energy balance (calorie surplus, deficit or maintenance, in kcal/day) from a sequence of daily body-weight readings, allowing for short-term fluctuation, measurement noise and uncertainty.

Say this early: weight alone cannot identify calorie intake, expenditure or exact physiological energy balance. The output is an approximate *weight-trend-derived energy-balance equivalent*, obtained by multiplying the trend in kg/day by an assumed 7,700 kcal/kg.

## 2. What was wrong with the internship version

Lead with these five; each was reproduced by running the original code (`results/audit/audit_original.json`).

1. **The accuracy claim was circular.** The README's "median error 8 kcal/day" compared the estimate with a straight-line fit of the same readings, not with the simulator's true balance.
2. **The app and the tested module computed different numbers.** The app had its own copy of the estimator; on identical logs they disagreed by {{A:A2_two_code_paths_disagree::mean_abs_disagreement_kcal|.0f}} kcal/day on average.
3. **Missed days were treated as consecutive.** With 40% of days missing, a noise-free trend of {{A:A3_missing_days_time_compression::2::true_kcal|.0f}} kcal/day was reported as {{A:A3_missing_days_time_compression::2::v1_app_mean_kcal|.0f}}.
4. **The "95% confidence interval" contained the truth about {{S:coverage_pooled_base::original|.0%}} of the time** in simulation, and the app showed a precise number for any 7 readings.
5. **The anomaly step flagged about 10% of readings by construction** ({{A:A4_isolation_forest_on_error_free_data::mean_fraction_flagged_on_error_free_data|.1%}} on error-free data) and overwrote them with lagging smoothed values.

Others, briefly: smoothing shrank the slope (a noise-free {{A:A1_ewma_attenuation_noise_free_line::0::true_kcal|.0f}} kcal/day trend was reported as {{A:A1_ewma_attenuation_noise_free_line::0::v1_research_kcal|.0f}} at 14 days); the headline number averaged the whole history; the "daily balance" bar chart was two-reading noise; the "on track" message was inverted; the coach treated the planned 5 kg change as the goal weight; fictional readings were pre-filled; the coach made hormone claims and gave one protein target to everyone, including users who selected kidney disease; the forecaster was evaluated with one normalisation and deployed with another.

Be fair to the original: on its own data (90 days, independent noise, no gaps) it was accurate, with a median error of {{A:A11_circular_validation_of_readme_claim::median_abs_error_vs_TRUE_simulated_balance_kcal|.0f}} kcal/day against the true balance. The problems were the way it was checked, and what happens outside that easy setting.

## 3. What I corrected

- One shared estimation module (`src/energy_balance.py`) used by the app, the coach, the experiments and the tests.
- Regression on raw readings against real dates; missed days supported.
- Huber robust regression instead of smoothing plus Isolation Forest; no reading is deleted or replaced.
- An autocorrelation-aware confidence interval, compared against three alternatives.
- "Not enough data" below 10 readings / 14 days; no deficit or surplus label below 25 days or when the interval does not support one.
- A trailing 28-day "current" estimate, with the whole-period figure shown separately.
- Input validation; no pre-filled data; demo data only on request and labelled.
- Coach and app text limited to what was computed; inverted and wrong-quantity logic fixed.
- Forecast: one shared function, a prediction interval, a default chosen by measured error.

## 4. Why these corrections were necessary

- A tool that reports a deficit must not be biased by how often the user weighs in. The missed-days defect made the answer depend on logging habits.
- An interval that is right 30% of the time is worse than none, because it signals confidence that is not there.
- Without known ground truth and a non-circular comparison, no accuracy statement about the estimator could be defended.
- Two implementations meant the tested code was not the code users saw.
- Health-adjacent software should not state mechanisms or targets it has no basis for.

## 5. New methods implemented

| Method | One-line explanation | Why |
|---|---|---|
| Huber regression | A straight-line fit in which readings far from the line count for less | Robust to gross errors without discarding data; equals least squares when data are clean |
| AR(1) sandwich interval | The slope's standard error computed allowing each day's deviation to resemble the previous day's | Water weight persists for days; the classical formula assumes it does not |
| Minimum-history and interval-based labelling | A direction is named only when the whole interval is on one side of zero and at least 25 days are covered | Chosen on development simulations as the shortest history with at most 5% wrong-direction calls |
| Reliability indicator | High / moderate / low from interval width, lowered for many missed days, large scatter or sensitivity to unusual readings | Tells a non-technical user how far to trust the number; no invented percentage |
| Scenario simulator | Two different generating models, six scenarios, ten conditions, stored truth | Makes accuracy measurable and tests the 7,700 assumption |

Comparison methods also implemented: least squares, Theil-Sen, EWMA-then-regression, Newey-West, block bootstrap.

## 6. Experiments conducted

- Development and evaluation users are disjoint (different seed bases). Every design choice used development data only.
- The original code runs unchanged (a test checks it is byte-identical to the ZIP) on exactly the same readings as the new methods.
- {{S:headline::base, last 28 d::n}} simulated users in the base condition, {{S:headline::clean, last 28 d::n}} in the clean condition, 480 per stress condition; windows of 14, 28 and 56 days.
- Metrics: MAE, RMSE, bias, three-way classification, interval coverage and width, error by reliability level, tracking error after a change of behaviour, forecast error and prediction-interval coverage.
- Paired differences per user with bootstrap confidence intervals and Wilcoxon tests.

## 7. What the results demonstrate

{{T:headline_before_after}}

Mean absolute error in kcal/day; simulated, unseen users.

What you can claim:

- With missed days or gross errors the corrected estimator is more accurate, by a wide margin when many days are missing ({{S:headline::missing 45%, last 28 d::legacy_app|.0f}} to {{S:headline::missing 45%, last 28 d::huber|.0f}}).
- In the base condition at 28 days the paired change in absolute error against the original app is {{S:paired::base, last 28 d vs legacy_app::diff|+.0f}} kcal/day (95% CI {{S:paired::base, last 28 d vs legacy_app::lo|+.0f}} to {{S:paired::base, last 28 d vs legacy_app::hi|+.0f}}; negative means the corrected method is closer).
- Interval coverage rose from {{S:coverage_pooled_base::original|.0%}} to {{S:coverage_pooled_base::huber_ar1|.0%}}.
- With 28 days of data the corrected app's stated direction was wrong for {{S:decisive::28::definite labels pointing the WRONG WAY|.1%}} of labelled users, against {{S:legacy_wrong_sign::last 28 d|.0%}} for the original app's printed sign.
- The reliability level orders real error: median {{S:reliability::high::median|.0f}} / {{S:reliability::moderate::median|.0f}} / {{S:reliability::low::median|.0f}} kcal/day.

What you must not claim:

- That the new method is more accurate on clean, complete logs. It is not ({{S:headline::clean, last 28 d::legacy_app|.0f}} vs {{S:headline::clean, last 28 d::huber|.0f}} kcal/day).
- That it is better at 14 days. The original research module has lower average error there ({{S:headline::base, last 14 d::legacy_research|.0f}} vs {{S:headline::base, last 14 d::huber|.0f}}), because its smoothing shrinks estimates towards zero. Nothing is useful at 14 days.
- That the interval is a true 95% interval. It covers about {{S:coverage_pooled_base::huber_ar1|.0%}}.
- Anything about real users, clinical accuracy or calorie intake.

## 8. Limitations that remain

1. Simulation only; both simulation models were written by me.
2. The interval under-covers; fixing it with a simulation-derived multiplier was deliberately not done.
3. The 7,700 kcal/kg factor is wrong for individuals by an unknown amount; in the second simulation model the median error from this alone is {{S:mechB_conversion::median_excess|+.0%}}.
4. Water shifts at the start or end of a diet look like energy balance.
5. A straight line assumes a constant balance inside the window.
6. Four weeks of consistent readings are needed before the tool says anything definite; even then it commits for under half of users.
7. Forecasting is secondary and no method wins on both simulators.
8. The LLM coach path and the guideline sources were not verified.

## 9. Further development

- Collect or obtain real longitudinal weight logs, ideally with an independent reference (doubly labelled water, or carefully logged intake in a controlled setting), and repeat section 6 on them.
- A state-space model (local linear trend with a water-weight component) to follow a changing balance and give better-calibrated intervals.
- Per-person or composition-aware conversion instead of a fixed 7,700.
- Explicit, documented shrinkage or a Bayesian prior, once the distribution of real users' balances is known.
- Missing-not-at-random logging (people who skip the scale after a heavy day).
- A proper user study of whether people understand the reliability indicator.

## Likely questions

**"Why remove Isolation Forest? It is the ML part."** It flagged a fixed 10% of readings whether or not anything was wrong, replaced measurements with lagging values, and gave no accuracy benefit in any tested condition. Huber regression does the job it was meant to do and can be explained in one sentence.

**"Is this still machine learning?"** The core is robust statistical estimation with quantified uncertainty. The project's contribution is the evaluation: a simulator with ground truth, a fair comparison, and an honest account of where the method fails.

**"Why is your interval not 95%?"** With two to eight weeks of autocorrelated data, the standard remedies (Newey-West, block bootstrap) did worse in my tests ({{S:coverage_pooled_base::ols_hac|.0%}} and {{S:coverage_pooled_base::huber_block_bootstrap|.0%}}). The adopted method reached {{S:coverage_pooled_base::huber_ar1|.0%}}. I report the shortfall instead of tuning it away on my own simulator.

**"Your method is no better on clean data. So what improved?"** Correctness under realistic logging (gaps, errors), honest uncertainty, refusing to answer when the data cannot, and following a change in behaviour. On ideal data there was little left to gain.

**"How do you know the simulator is realistic?"** I do not. That is why there are two different models, why the second one breaks the estimator's own assumption, and why real data is the first item of future work.
