| interval | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original 'kcal_per_day_ci95' | 0.29 | 0.32 | 0.31 | 0.28 | 0.27 | 0.26 | 0.31 | 0.25 |
| Classical OLS | 0.80 | 0.76 | 0.73 | 0.79 | 0.71 | 0.63 | 0.59 | 0.62 |
| Newey-West HAC (OLS) | 0.74 | 0.77 | 0.81 | 0.79 | 0.65 | 0.67 | 0.69 | 0.67 |
| AR(1) sandwich (OLS) | 0.88 | 0.89 | 0.91 | 0.90 | 0.84 | 0.83 | 0.83 | 0.82 |
| Huber, independence | 0.79 | 0.73 | 0.67 | 0.74 | 0.66 | 0.60 | 0.54 | 0.58 |
| Huber AR(1), no small-sample correction | 0.85 | 0.85 | 0.88 | 0.86 | 0.75 | 0.79 | 0.81 | 0.78 |
| Huber AR(1) sandwich (recommended) | 0.87 | 0.87 | 0.89 | 0.89 | 0.80 | 0.85 | 0.83 | 0.84 |
| Huber, block bootstrap | 0.64 | 0.73 | 0.78 | 0.75 | 0.56 | 0.63 | 0.68 | 0.62 |
