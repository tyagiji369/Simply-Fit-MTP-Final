| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | -15 | -9 | -24 | +2 | -35 | +15 | +9 | -67 |
| Original: app (IF-replaced raw) | -1 | +3 | -8 | -20 | -13 | +18 | +19 | -73 |
| Original steps, real day index | -18 | -8 | -14 | +28 | -37 | +7 | +10 | -32 |
| EWMA then regression (no IF) | -15 | -5 | -10 | +21 | -31 | +9 | +12 | -32 |
| Least squares on raw readings | +4 | +8 | +6 | +1 | -8 | +12 | +22 | -38 |
| Theil-Sen | +3 | +7 | +6 | +0 | -13 | +7 | +22 | -38 |
| Huber regression (recommended) | +1 | +8 | +7 | +0 | -7 | +11 | +23 | -38 |
