| method | MAE (kg), all forecasts | MAE (kg), model A | MAE (kg), model B | MAE (kg), subset with 14 complete days |
|---|---|---|---|---|
| Recent level: mean of last 3 readings (app default) | 0.501 | 0.503 | 0.500 | 0.490 |
| Persistence (last reading) | 0.522 | 0.523 | 0.521 | 0.516 |
| Linear trend, last 14 days | 0.595 | 0.572 | 0.619 | 0.604 |
| Original fallback (2-point) | 0.693 | 0.685 | 0.703 | 0.691 |
| Optional LSTM, window-centred (needs 14 complete days) |  |  |  | 0.491 |
