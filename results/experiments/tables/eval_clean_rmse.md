| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | 311 | 143 | 64 | 147 | 356 | 200 | 89 | 219 |
| Original: app (IF-replaced raw) | 310 | 133 | 57 | 132 | 399 | 193 | 92 | 213 |
| Original steps, real day index | 311 | 143 | 64 | 147 | 356 | 200 | 89 | 219 |
| EWMA then regression (no IF) | 305 | 135 | 60 | 135 | 367 | 201 | 90 | 218 |
| Least squares on raw readings | 317 | 133 | 57 | 129 | 433 | 200 | 95 | 216 |
| Theil-Sen | 331 | 138 | 61 | 131 | 440 | 207 | 96 | 222 |
| Huber regression (recommended) | 323 | 136 | 59 | 130 | 436 | 203 | 96 | 219 |
