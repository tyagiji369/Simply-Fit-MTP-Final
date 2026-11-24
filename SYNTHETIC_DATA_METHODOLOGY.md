# Synthetic Data Methodology

**Everything described here is simulated.** No series produced by this project is a real person's measurements, and no result obtained on it is real-world or clinical validation. The simulator exists because no suitable real longitudinal body-weight dataset was available for this project.

Code: `src/simulation.py`. Parameters: `config/scenarios.json` (edit that file to change any value; nothing below is hard-coded in the simulator). Tests: `tests/test_simulation.py`.

## 1. Why the internship generator was not enough

`src/data_generator.py` (kept, still used to train the optional LSTM and for the NHANES demographic check) has these properties, measured in `results/audit/audit_original.json`:

| Property of the original generator | Consequence |
|---|---|
| Day-to-day scatter is independent (lag-1 autocorrelation of detrended readings ≈ −0.01) | Real water weight persists for days. Independent noise makes any trend estimator, and any confidence interval, look better than it would be. |
| No missed days, no gross measurement errors | The anomaly step and the "list position = day" shortcut were never exercised. |
| One trajectory shape (a near-straight line for 90 days) | Nothing tests behaviour when a diet stops or reverses. |
| The true daily energy balance is not stored | Accuracy could not be scored against truth. The internship notebook scored the estimator against a straight-line fit of the same noisy readings instead. |
| Tissue changes by exactly balance ÷ 7700 | The data embody the estimator's own assumption, so the conversion error is invisible. |

## 2. What every simulated series contains

For each simulated user and each day *t* = 0 … *n*−1 (default *n* = 90):

| Field | Unit | Meaning |
|---|---|---|
| `observed[t]` | kg | The scale reading, or missing |
| `tissue[t]` | kg | True tissue weight (fat + lean), free of water noise and measurement error |
| `balance[t]` | kcal/day | **Ground truth**: the true energy balance during day *t*, i.e. what moved `tissue[t−1]` to `tissue[t]` |
| `is_outlier[t]` | – | Whether a gross measurement error was injected |

Two ground-truth quantities are derived for any window of days [*a*, *b*]:

- **True mean balance** = mean of `balance[a+1 … b]` (kcal/day). This is the scientific target.
- **Tissue-trend equivalent** = (`tissue[b]` − `tissue[a]`) ÷ (*b* − *a*) × 7700 (kcal/day). This is the most any weight-only method could report: the true tissue slope converted with the assumed factor. In mechanism A the two are identical; in mechanism B they differ, and the difference is the error of the 7700 assumption itself.

## 3. Two data-generating mechanisms

Two mechanisms are used so that a method cannot look good merely because the data were built from its own equations.

### Mechanism A: "ledger" (shares the estimator's conversion assumption)

For *t* ≥ 1:

```
balance[t] = plan[t] + e_t                       e_t ~ Normal(0, s_b),  s_b ~ U(100, 300) kcal/day
tissue[t]  = tissue[t-1] + balance[t] / 7700
water[t]   = phi * water[t-1] + u_t              stationary AR(1), SD s_w ~ U(0.20, 0.60) kg, phi ~ U(0.30, 0.80)
reading[t] = tissue[t] + c*water[t] + c*m_t      m_t ~ Normal(0, s_m), s_m ~ U(0.05, 0.15) kg
```

`plan[t]` is the scenario's planned balance, `c` is the noise multiplier of the observation condition. Because actual daily balance varies around the plan, tissue weight follows a random walk around its trend; the ground truth is the realised balance, not the plan.

Purpose: isolates how well a method handles persistent noise, gaps and gross errors when its own model of tissue change is correct.

### Mechanism B: "two-compartment, adaptive" (does not share it)

State: fat mass *F*, lean mass *L*, adaptive-thermogenesis term *AT*, fast glycogen-plus-water pool *G*. Scenario values are **planned intake offsets** from initial maintenance, not balances.

```
intake_off[t] = plan[t] + weekly[t] + feast[t] + Normal(0, s_i)
AT[t]         = AT[t-1] + (0.14 * plan[t] - AT[t-1]) / 14
dEE[t]        = 3.2*(F-F0) + 22*(L-L0) + 7*((F+L)-(F0+L0)) + 0.10*intake_off[t] + AT[t]
balance[t]    = intake_off[t] - dEE[t]                        (kcal/day, ground truth)
p             = C / (C + F),   C = 10.4 * 1800 / 9400         (Forbes partition)
F            += (1 - p) * balance[t] / 9400
L            +=      p  * balance[t] / 1800
G            += (g * intake_off[t] / 1000 - G) / 3
tissue[t]     = F + L
reading[t]    = tissue[t] + G[t] + c*(water[t] + cycle[t]) + c*m_t
```

| Element | Value | Why it is in the model |
|---|---|---|
| Energy density of fat / lean tissue | 9400 / 1800 kcal/kg | A kilogram of weight change is then worth roughly 6000–8000 kcal depending on body fat, not 7700. Tests the conversion assumption. |
| Forbes partition, constant 10.4 kg | share of an imbalance going to lean tissue falls as fat mass rises | Makes the conversion factor differ between simulated people. |
| Expenditure response | 3.2 kcal/kg fat, 22 kcal/kg lean, 7 kcal/kg body weight for activity, 10% thermic effect of intake, adaptive thermogenesis 14% of the planned offset with a 14-day lag | A constant diet produces a shrinking deficit, so the true balance drifts even when behaviour does not. |
| Initial body-fat fraction | Deurenberg (1991) equation from BMI, age and sex, clipped to 10–50% | Gives each user a starting composition. |
| Glycogen-plus-water pool | 0.3–0.9 kg per 1000 kcal/day of intake offset, 3-day time constant | Starting or stopping a diet moves water weight within days. This is the main known confounder of weight-based inference. |
| Weekly rhythm | weekend intake +0 to +400 kcal/day, weekdays reduced to keep the weekly mean | Periodic structure the linear model does not contain. |
| Occasional large meals | 8% of days, +600 to +1200 kcal, mean-centred | Skewed intake, feeding the water pool the next day. |
| Water noise | AR(1), SD 0.20–0.55 kg, phi 0.40–0.85, Student-t (4 d.f.) innovations | Heavier tails and longer memory than mechanism A. |
| Slow cycle | 30% of users: sinusoid, period 24–35 days, amplitude 0.3–0.8 kg | A non-energy fluctuation that looks like a trend over two weeks. |
| Missed days | in runs of 2–6 consecutive days | Travel-like gaps rather than independent misses. |
| Gross errors | 70% upward | Clothed or after-meal weighings are more common than underweighings. |

The glycogen pool's own energy content is neglected (it is small next to the fat and lean terms); this is a documented simplification.

The numerical constants are typical values of the kind used in published dynamic models of human body-weight change (Hall and colleagues) and the Forbes relationship. They were written from general knowledge of that literature and **were not re-verified against the source papers for this project**; they should be treated as plausible assumptions, and the exact figures checked before being cited in a thesis. Mechanism B is a deliberately different stress model, not a validated physiological simulator.

### Demographics (both mechanisms)

Sex, age, height and starting weight are drawn from the distributions the internship project fitted to CDC NHANES 2017-2018 (`data/public/nhanes_adults.csv`), reused unchanged from `src/data_generator.py`.

## 4. Scenarios

Each scenario defines `plan[t]`. Ranges are sampled once per user.

| Scenario | Planned value (kcal/day) |
|---|---|
| `stable` | 0 throughout |
| `sustained_deficit` | constant, −700 to −250 |
| `sustained_surplus` | constant, +200 to +500 |
| `deficit_to_maintenance` | −700 to −300 until a change day drawn from days 35–75, then 0 |
| `deficit_to_surplus` | −700 to −300 until the change day, then +200 to +500 |
| `gradual_drift` | linear from (−700 to −350) on day 0 to (−100 to +100) on the last day |

In mechanism A these are balances; in mechanism B they are intake offsets and the realised balance is smaller in magnitude and drifts.

## 5. Observation conditions

Applied after the true trajectory is generated, from a separate random stream. **The same user (same seed, mechanism, scenario) has the identical true trajectory under every condition**, so conditions are compared like-for-like.

| Condition | Noise multiplier | Missed days | Gross errors |
|---|---|---|---|
| `clean` | 1.0 | 0% | 0% |
| `base` | 1.0 | 10% | 2% |
| `noise_low` / `noise_high` | 0.5 / 1.6 | 10% | 2% |
| `missing_0` / `missing_25` / `missing_45` | 1.0 | 0% / 25% / 45% | 2% |
| `outliers_0` / `outliers_5` / `outliers_10` | 1.0 | 10% | 0% / 5% / 10% |

Gross errors add or subtract 1–3 kg to a single reading. All readings are rounded to 0.1 kg, as a bathroom scale displays. `clean` is the condition most favourable to the original pipeline and is always reported next to `base`.

## 6. Seeds and the development / evaluation split

Each series is generated from `SeedSequence([seed_base, mechanism, scenario, user, stream])`, so every run is exactly reproducible.

| Use | `seed_base` | Used for |
|---|---|---|
| Development | 100000 | Every design decision: choice of estimator, interval method, default window, reliability thresholds |
| Evaluation | 900000 | Reported results only; never looked at while designing |

The two bases produce disjoint simulated users. The frozen original pipeline has no tunable parameters in this project, so it is only run on evaluation data.

## 7. Limitations of this simulator

1. It is not real data. Agreement between a method and simulated truth shows the method is correct for the simulator's assumptions, nothing more.
2. Noise magnitudes and persistence are assumed ranges, not fitted to measured weight logs.
3. Mechanism B's constants were not re-verified against their sources (see above).
4. Real confounders that are absent: illness, medication changes, menstrual-cycle patterns beyond a generic sinusoid, constipation, alcohol, travel across time zones, changes of scale, deliberate dehydration, growth of muscle from new training.
5. Missingness is unrelated to weight. Real users may skip weighing exactly when weight is up, which would bias any method.
6. Both mechanisms are designed by the same author as the estimator. Using two mechanisms reduces, but cannot remove, the risk that the evaluation favours the estimator's assumptions.
7. Because simulated "truth" in the scenarios is spread roughly symmetrically around zero, methods that shrink estimates towards zero are rewarded in average-error terms. This is discussed in `EXPERIMENT_RESULTS.md`.

## 8. Reproducing

```bash
python -c "from src.simulation import simulate_series; s = simulate_series('B', 'deficit_to_maintenance', 'base', user=0, seed_base=900000); print(s.observed[:10], s.true_mean_balance(62, 89))"
pytest tests/test_simulation.py -q
```
