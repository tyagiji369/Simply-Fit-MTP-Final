| method | MAE (kg), all forecasts | MAE (kg), model A | MAE (kg), model B | MAE (kg), subset with 14 complete days |
|---|---|---|---|---|
| Recent level: mean of last 3 readings (app default) | 0.490 | 0.480 | 0.501 | 0.476 |
| Persistence (last reading) | 0.518 | 0.521 | 0.514 | 0.498 |
| Linear trend, last 14 days | 0.579 | 0.527 | 0.634 | 0.573 |
| Original fallback (2-point) | 0.692 | 0.696 | 0.689 | 0.652 |
| Optional LSTM, window-centred (needs 14 complete days) |  |  |  | 0.477 |
