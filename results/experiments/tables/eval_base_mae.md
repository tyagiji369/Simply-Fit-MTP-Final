| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | 272 | 134 | 68 | 140 | 312 | 184 | 86 | 208 |
| Original: app (IF-replaced raw) | 276 | 126 | 63 | 129 | 365 | 180 | 87 | 206 |
| Original steps, real day index | 251 | 122 | 54 | 129 | 282 | 162 | 70 | 166 |
| EWMA then regression (no IF) | 253 | 119 | 51 | 121 | 297 | 164 | 71 | 166 |
| Least squares on raw readings | 269 | 116 | 48 | 112 | 359 | 166 | 75 | 169 |
| Theil-Sen | 261 | 110 | 49 | 105 | 349 | 162 | 75 | 169 |
| Huber regression (recommended) | 253 | 110 | 48 | 104 | 347 | 158 | 74 | 166 |
