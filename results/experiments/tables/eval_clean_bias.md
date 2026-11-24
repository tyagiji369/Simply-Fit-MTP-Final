| method | A last 14 d | A last 28 d | A last 56 d | A first 28 d | B last 14 d | B last 28 d | B last 56 d | B first 28 d |
|---|---|---|---|---|---|---|---|---|
| Original: research module (EWMA + IF + EWMA) | -6 | -3 | -12 | +21 | -24 | +15 | +11 | -47 |
| Original: app (IF-replaced raw) | +13 | +7 | +1 | +6 | -17 | +13 | +19 | -45 |
| Original steps, real day index | -6 | -3 | -12 | +21 | -24 | +15 | +11 | -47 |
| EWMA then regression (no IF) | -1 | -1 | -8 | +16 | -25 | +14 | +14 | -45 |
| Least squares on raw readings | +21 | +11 | +5 | +0 | -16 | +13 | +22 | -42 |
| Theil-Sen | +23 | +13 | +6 | +1 | -24 | +13 | +22 | -41 |
| Huber regression (recommended) | +21 | +13 | +6 | +1 | -20 | +13 | +22 | -42 |
