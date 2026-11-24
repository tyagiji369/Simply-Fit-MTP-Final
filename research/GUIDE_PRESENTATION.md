# Presenting the Simply-Fit M.Tech extension

Allow 8–10 minutes plus questions. Open with the evidence gap before displaying any performance number.

| Slide | What to show | What to say |
|---|---|---|
| 1 · Research question | A dated weight log and the two targets from report §2 | “I investigated when weight alone supports a useful estimate of average tissue change. I had no real-world longitudinal reference labels.” |
| 2 · What I inherited | Three-version table, report §1 | “The existing MTP already fixed the calendar clock and implemented robust regression and AR intervals. My contribution begins with independent stress testing.” |
| 3 · Independent simulator | A tissue path, decaying water pulses, temporary offset and a missed measurement described in §3 | Explain why skipping a weigh-in after a water increase creates informative missingness. The exact ranges are assumptions. |
| 4 · Fair comparison | Split table and target formula | Say the same log goes to each method, the truth covers identical endpoints and selection occurred before final evaluation. |
| 5 · Does a Kalman model help? | estimators.png and paired interval | “It can track changing velocity. Its average trend did not beat Huber by a resolved margin, so I retained Huber.” |
| 6 · Which intervals work? | intervals.png | Compare measured coverage and width together. Describe how held-aside validation residuals set the calibration multiplier. |
| 7 · When should the app stay silent? | selective_risk.png | “580 of 960 users received an estimate. Their mean error was 82.3 kcal/day. The other users are part of the result.” |
| 8 · Failure cases | Default 28-day and long-water coverage; same-scale example | “Overall coverage hides undercoverage in a common app window. Two identical weight histories can imply −308 or −110 kcal/day.” |
| 9 · Demonstration and conclusion | Tracker and report using a labeled demo or synthetic log | Show the interval, withholding behavior and distinct forecast band. Close with the scope of evidence and what a future longitudinal study would need. |

### Questions a guide may ask

**Why keep Huber after building a more complex model?** The robust filter's held-out MAE advantage was 0.16 kcal/day, with a paired interval spanning both signs. It added research value by exposing model-based overconfidence and providing a disagreement signal.

**What is the clinical meaning of 95%?** It is a nominal level for a latent simulated window target among exchangeable logs. Across reported 28-day logs it covered 87.7%. Clinical accuracy remains unknown.

**Why 75%?** I chose an operating preference on development data to demonstrate the reporting trade-off. There was no externally specified cost of withholding versus error. The plotted curve allows that decision to be revisited.

**Why can a narrower interval cover more often?** Width allocation matters. An AR-based range may be very wide for some users and too narrow for others. Compare full widths on identical logs.

**What would make the next validation persuasive?** Collect timestamped standardized weigh-ins, repeat measurements for noise, information on temporary fluid shifts and an independent reference appropriate to the target. Preregister the estimand, calibration population and acceptable miss rate, then test the frozen policy prospectively.
