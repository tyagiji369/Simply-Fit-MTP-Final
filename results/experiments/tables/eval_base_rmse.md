| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | 371 | 182 | 90 | 189 | 438 | 247 | 116 | 277 |
| Original: app (IF-replaced raw) | 362 | 163 | 80 | 168 | 503 | 234 | 118 | 276 |
| Original steps, real day index | 336 | 164 | 72 | 173 | 389 | 217 | 94 | 219 |
| EWMA then regression (no IF) | 338 | 156 | 66 | 159 | 402 | 215 | 94 | 217 |
| Least squares on raw readings | 355 | 148 | 62 | 145 | 468 | 216 | 99 | 222 |
| Theil-Sen | 341 | 142 | 64 | 137 | 460 | 216 | 100 | 223 |
| Huber regression (recommended) | 333 | 141 | 62 | 136 | 454 | 210 | 98 | 221 |
