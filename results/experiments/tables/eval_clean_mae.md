| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | 239 | 111 | 50 | 116 | 272 | 152 | 67 | 169 |
| Original: app (IF-replaced raw) | 236 | 103 | 45 | 102 | 304 | 150 | 69 | 160 |
| Original steps, real day index | 239 | 111 | 50 | 116 | 272 | 152 | 67 | 169 |
| EWMA then regression (no IF) | 233 | 104 | 47 | 107 | 278 | 153 | 68 | 167 |
| Least squares on raw readings | 242 | 105 | 45 | 99 | 331 | 155 | 72 | 162 |
| Theil-Sen | 252 | 105 | 48 | 100 | 332 | 157 | 73 | 167 |
| Huber regression (recommended) | 245 | 107 | 47 | 100 | 331 | 154 | 73 | 164 |
